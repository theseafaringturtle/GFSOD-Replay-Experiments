import json
import logging
import math
import os
import random
import shutil
import time
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, List

import torch
from detectron2.modeling.poolers import ROIPooler
from detectron2.data import MetadataCatalog
from detectron2.utils.env import seed_all_rng

from defrcn.dataloader import build_detection_train_loader, get_detection_dataset_dicts, DatasetMapper
from defrcn.dataloader.finite_sampler import FiniteTrainingSampler
from defrcn.evaluation.archs import resnet101
from samplers.utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class BaseSampler(metaclass=ABCMeta):
    """Base class that takes care of dataset loading and saving. Will not modify novel samples."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.class_samples = {cls_id: set() for cls_id in
                              range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}  # class_id to [file_name]
        self.sample_labels = {} # file_name to label tensor
        self.class_instance_counts = {cls_id: 0 for cls_id in
                                      range(cfg.MODEL.ROI_HEADS.NUM_CLASSES)}  # class_id to instance_count
        self.imagenet_model = self.build_model()
        self.roi_pooler = ROIPooler(output_size=(1, 1), scales=(1 / 32,), sampling_ratio=(0), pooler_type="ROIAlignV2")

        # Max Proportion of extra base instances tolerated across pool
        self.BASE_INSTANCE_LIMIT_PROP = 2
        # Max Proportion of instances tolerated per sample w.r.t size of pool, 0.1 = any sample can't contribute more than 10%.
        self.BASE_PER_SAMPLE_LIMIT_PROP = 0.1

    def build_model(self):
        logger.info("Loading ImageNet Pre-train Model from {}".format(self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    @time_perf(logger)
    def gather_sample_pool(self, req_pool_size: int):
        """Load data and call process_image_entry on each item"""
        logger.info("Gathering samples...")
        # Ensure starting pool will be the same for same seed
        seed_all_rng(self.cfg.SEED)
        # Section adapted from class instance histogram code in dataloader/build.py
        # Get class IDs and how many total instances we have per class across dataset
        dataset: List[Dict] = get_detection_dataset_dicts(self.cfg.DATASETS.TRAIN)
        class_ids = set()
        for sample in dataset:
            sample_ids = [anno['category_id'] for anno in sample['annotations']]
            for sample_id in sample_ids:
                class_ids.add(sample_id)
        class_ids = sorted(list(class_ids))
        # List by class index -> List of entries -> Dict entries
        dataset_class_samples: List[List[Dict]] = [[] for cls_id in
                                                   class_ids]  # class_id to [file_name]
        dataset_class_instances = {cls_id: 0 for cls_id in
                                   class_ids}  # class_id to instance_count
        for entry in dataset:
            annos = entry["annotations"]
            classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
            classes = [class_ids.index(cat_id) for cat_id in classes]
            for instance_cid in set(classes):
                dataset_class_samples[instance_cid].append(entry)
                dataset_class_instances[instance_cid] += classes.count(instance_cid)

        for cid in class_ids:
            random.shuffle(dataset_class_samples[cid])
            # print(f"Name: {self.base_class_id_to_name(cid)} \t Instances: {dataset_class_instances[cid]}")

        # For logging
        filled_classes = set()
        newly_filled_classes = set()

        # Start from classes with the fewest instances
        ranked_class_ids = class_ids
        ranked_class_ids.sort(key=lambda cid: len(dataset_class_samples[cid]))
        for class_id in ranked_class_ids:
            # We have enough samples to start ranking them, stop going through dataset
            if len(self.class_samples[class_id]) >= req_pool_size:
                continue

            for entry in dataset_class_samples[class_id]:
                if len(self.class_samples[class_id]) >= req_pool_size:
                    break
                file_name = entry['file_name']
                gt_classes = torch.IntTensor(
                    [e["category_id"] for e in entry["annotations"] if not e.get("iscrowd", 0)])

                # For checks
                has_too_many_instances = False

                # Count unique labels
                unique_classes, unique_counts = gt_classes.unique(return_counts=True)
                # Number of instances checks
                for i, instance_cid in enumerate(unique_classes):
                    # Take note of the number of instances we've gathered for that class
                    current_class_instances = self.class_instance_counts[int(instance_cid)]
                    # Only allow a 200% overshoot for the pool, and a single sample should not contribute more than 10% to total instances
                    if current_class_instances + unique_counts[i] > req_pool_size * self.BASE_INSTANCE_LIMIT_PROP \
                            or unique_counts[i] > math.ceil(req_pool_size * self.BASE_PER_SAMPLE_LIMIT_PROP):
                        has_too_many_instances = True
                        break
                if not has_too_many_instances:
                    # Add to pool
                    for i, instance_cid in enumerate(unique_classes):
                        self.class_samples[int(instance_cid)].add(file_name)
                        self.class_instance_counts[int(instance_cid)] += unique_counts[i]
                        self.sample_labels[file_name] = gt_classes.tolist()
                        # for logging
                        if len(self.class_samples[int(instance_cid)]) >= req_pool_size:
                            newly_filled_classes.add(int(instance_cid))
            for newly_filled_cid in newly_filled_classes:
                if newly_filled_cid not in filled_classes:
                    filled_classes.add(newly_filled_cid)
                    logger.info(f"Sample pool for {self.base_class_id_to_name(newly_filled_cid)} has been filled")
            newly_filled_classes.clear()
        # Just leaving dataloader here in case we decide to run this in parallel and gather tensors
        # The easiest way would be to set IMS_PER_BATCH = num_gpus and gather dictionaries as part of class method
        logger.info("Finished gathering samples, processing...")
        pool_file_names = set()
        for cid in class_ids:
            pool_file_names = pool_file_names.union(self.class_samples[cid])
        pool_dataset = [d for d in dataset if d["file_name"] in pool_file_names]
        memory_cfg = self.cfg.clone()
        memory_cfg.defrost()
        memory_cfg.DATALOADER.SAMPLER_TRAIN = "FiniteTrainingSampler"
        # To obtain 1 image per GPU
        memory_cfg.SOLVER.IMS_PER_BATCH = 1
        mapper = DatasetMapper(memory_cfg, True)
        sampler = FiniteTrainingSampler(len(pool_dataset))
        total_batch_size = memory_cfg.SOLVER.IMS_PER_BATCH
        memory_loader = build_detection_train_loader(pool_dataset, mapper=mapper, sampler=sampler,
                                                     total_batch_size=total_batch_size)
        memory_iter = iter(memory_loader)
        num_processed = 0
        for input in memory_iter:
            num_processed += 1
            entry = input[0]
            if num_processed in list(range(0, len(pool_dataset), len(pool_dataset) // 10)):
                logger.info(f"Samples processed: {round(num_processed / len(pool_dataset) * 100)}%")
            self.process_image_entry(entry)

    @abstractmethod
    def process_image_entry(self, input):
        """Store / collate the information of each detectron2 dataset entry however you wish"""
        pass

    @abstractmethod
    def process_post(self):
        """Further processing with the fully collected images"""
        pass

    @abstractmethod
    def select_samples(self, samples_needed: int) -> Dict:
        """Filter collected information and return mapping of class id to array of filenames with len samples_needed"""
        pass

    def base_class_id_to_name(self, class_id: int):
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if 'voc' in train_set_name:
            base_classes = MetadataCatalog.get(train_set_name).get("base_classes", None)
            return base_classes[class_id]
        elif 'coco' in train_set_name:
            # cid_to_contiguous = MetadataCatalog.get(train_set_name).get("base_dataset_id_to_contiguous_id")
            # contiguous_to_cid = {v: k for k, v in cid_to_contiguous.items()}
            return MetadataCatalog.get(train_set_name).get("base_classes")[class_id]
        else:
            raise Exception(
                "You need to specify a class ID mapping for base classes, or add your dataset to this function")

    def is_base_id(self, class_id: int):
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if 'voc' in train_set_name:
            return class_id < 15
        elif 'coco' in train_set_name:
            return class_id in MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get(
                "base_dataset_id_to_contiguous_id").keys()
        else:
            raise Exception(
                "You need to specify a class ID mapping for base classes, or add your dataset to this function")

    def create_dirs(self, prev_seed, new_seed) -> Tuple[str, str]:
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        if "voc" in train_set_name:
            base_dir = os.path.join('../datasets', "vocsplit")
        elif "coco" in train_set_name:
            base_dir = os.path.join('../datasets', "cocosplit")
        else:
            raise Exception("Specify a split directory for your dataset")
        # Get previous directory for novel classes
        prev_seed_dir = os.path.join(base_dir, f"seed{prev_seed}")
        # Create new directory to store new data as a new seed, starting from whichever number was provided in config
        new_seed_dir = os.path.join(base_dir, f"seed{new_seed}")
        os.makedirs(new_seed_dir, exist_ok=True)
        return prev_seed_dir, new_seed_dir

    def save(self, filenames_per_base_class: Dict, samples_needed: int, prev_seed: int, new_seed: int):
        """Create a few-shot dataset for either VOC or COCO"""
        logger.info("Saving split to disk...")
        # Copy novel class files verbatim, since changing those would change the benchmark
        train_set_name = self.cfg.DATASETS.TRAIN[0]
        prev_seed_dir, new_seed_dir = self.create_dirs(prev_seed, new_seed)
        if 'voc' in train_set_name:
            txt_files = os.listdir(prev_seed_dir)
            # Get novel class names
            novel_classes = MetadataCatalog.get(train_set_name).get("novel_classes", None)
            if not novel_classes:
                raise Exception(
                    f"Dataset {train_set_name} has no novel_classes set, check builtin_meta.py for an example on how to set them")
            novel_txt_files = []
            for class_name in novel_classes:
                for file in txt_files:
                    if os.path.isfile(f"{prev_seed_dir}/{file}") \
                            and class_name in file and f"_{samples_needed}shot" in file:
                        novel_txt_files.append(file)
            if not novel_txt_files:
                raise Exception(f"No novel class txt files found under {prev_seed_dir}")
            # Copy the existing novel class txt files
            for file in novel_txt_files:
                shutil.copy(f"{prev_seed_dir}/{file}", f"{new_seed_dir}")
            # Copy our new base class txt files
            for class_id, file_names in filenames_per_base_class.items():
                # Note: instance filtering is performed later in meta_voc.py
                class_name = self.base_class_id_to_name(class_id)
                with open(f"{new_seed_dir}/box_{samples_needed}shot_{class_name}_train.txt", 'w') as text_file:
                    text_file.write('\n'.join(file_names) + '\n')
        elif 'coco' in train_set_name:
            data_path = "../datasets/cocosplit/datasplit/trainvalno5k.json"
            with open(data_path, "r") as data_file:
                data = json.load(data_file)
            logger.info("Loading trainvalno5k.json ...")
            new_all_cats = []
            for cat in data["categories"]:
                new_all_cats.append(cat)

            # Extract relevant images and annotations in a single pass for all classes
            # base_classes = MetadataCatalog.get(train_set_name).get("base_classes", None)
            all_filenames = sum(filenames_per_base_class.values(), [])
            all_filenames = [os.path.basename(f) for f in all_filenames]
            id2img = {}
            filename2img = {}
            for img in data["images"]:
                if img["file_name"] in all_filenames:
                    id2img[img["id"]] = img
                    filename2img[img["file_name"]] = img
            all_annotations = []
            for anno in data["annotations"]:
                if anno["image_id"] in id2img.keys() and self.is_base_id(anno["category_id"]) \
                        and not anno.get("is_crowd", 0):
                    all_annotations.append(anno)
            # Match images to their respective classes, annotations to their images, save to json
            new_base_data = {}
            for class_id, file_names in filenames_per_base_class.items():
                file_names = [os.path.basename(f) for f in file_names]
                class_images = [filename2img[filename] for filename in file_names]
                class_image_ids = [img["id"] for img in class_images]
                annotations = [anno for anno in all_annotations if anno["image_id"] in class_image_ids]
                new_base_data[class_id] = {
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "categories": data["categories"],
                    "images": class_images,
                    "annotations": annotations,
                }
            logger.info("Data collated...")
            # Create base files
            for class_id, base_data in new_base_data.items():
                base_class_name = self.base_class_id_to_name(class_id)
                # Oddly enough, if the first file wasn't removed then it wouldn't be properly overwritten by json.dump()
                # but a truncated part of the old file remained. This was an intermittent but reproducible issue on the hpc
                if os.path.exists(f"{new_seed_dir}/full_box_{samples_needed}shot_{base_class_name}_trainval.json"):
                    os.remove(f"{new_seed_dir}/full_box_{samples_needed}shot_{base_class_name}_trainval.json",)
                with open(f"{new_seed_dir}/full_box_{samples_needed}shot_{base_class_name}_trainval.json",
                          'w') as json_file:
                    json.dump(base_data, json_file)
            # Copy novel files verbatim
            logger.info("Copying novel classes...")
            novel_classes = MetadataCatalog.get(train_set_name).get("novel_classes", None)
            if not novel_classes:
                raise Exception(
                    f"Dataset {train_set_name} has no novel_classes set, check builtin_meta.py for an example on how to set them")
            for novel_class_name in novel_classes:
                source = f"{prev_seed_dir}/full_box_{samples_needed}shot_{novel_class_name}_trainval.json"
                dest = f"{new_seed_dir}/full_box_{samples_needed}shot_{novel_class_name}_trainval.json"
                shutil.copy(source, dest)
                logger.info(f"Copied {novel_class_name}")
        else:
            raise Exception(
                f"You need to implement data split saving for {train_set_name}, check this function and https://github.com/ucbdrive/few-shot-object-detection/tree/master/datasets")
        logger.info(f"Split saved to {new_seed_dir}! ({samples_needed}-shot)")
