import logging
import os.path
import pickle
import time
import re
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size, get_rank  # , get_rank, reduce_dict
from iopath import PathManager
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.hooks import RemovableHandle
import torch.distributed as dist
from DeFRCNTrainer import DeFRCNTrainer
from GPM import FeatureMap, get_representation_matrix, update_GPM, register_feature_map_hooks, \
    determine_conv_output_sizes
from MemoryTrainer import MemoryTrainer

logger = logging.getLogger("defrcn").getChild(__name__)


def reduce_dict(input_dict, average=True):
    """
    Same as detectron's reduce_dict, but without stacking the values, which only worked for them in the loss use case
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        if isinstance(input_dict, OrderedDict):
            for k in input_dict.keys():
                names.append(k)
                input_dict[k] = input_dict[k].cuda()
                values.append(input_dict[k])
        else:
            for k in sorted(input_dict.keys()):
                names.append(k)
                input_dict[k] = input_dict[k].cuda()
                values.append(input_dict[k])
        # values = torch.stack(values, dim=0)
        dist.barrier()
        for i in range(len(values)):
            # print(names[i])
            dist.all_reduce(values[i], op=dist.ReduceOp.AVG)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class FasterRCNNFeatureMap(FeatureMap):
    def __init__(self, multi_gpu=False, batch_size: int = 8):
        super().__init__()
        self.layer_names = [f'backbone.res4.{i}.conv1' for i in range(23)] + \
                           [f'backbone.res4.{i}.conv2' for i in range(23)] + \
                           [f'backbone.res4.{i}.conv3' for i in range(23)] + ['proposal_generator.rpn_head.conv']
        self.layer_names = sorted(self.layer_names)
        # Workaround for documented behaviour: https://github.com/pytorch/pytorch/issues/9176
        if multi_gpu:
            self.layer_names = ["module." + name for name in self.layer_names]
        self.samples = {name: batch_size for name in self.layer_names}
        self.threshold = {name: 0.97 for name in self.layer_names}


def create_random_sample(max_size: Tuple[int, int, int]) -> dict:
    sample = dict()
    random_tensor = torch.rand(max_size)
    sample['image'] = random_tensor
    sample['width'] = max_size[1]
    sample['height'] = max_size[2]
    sample['image_id'] = f'777777{torch.rand(1).item() * 100:.0f}'
    sample['file_name'] = sample['image_id'] + '.jpg'
    sample['instances'] = Instances(max_size[1:])
    return sample


def get_base_ds_name(train_set_name: str):
    """ Get base name of dataset, used for caching features across experiments
        Assuming the usual format for TFA and later experiments: basename_(novel/base/all)_kshot_seedx"""
    match = re.match('.*(_novel|_base|_all)', train_set_name)
    if not match:
        raise Exception("GPM: Could not get base dataset name for " + train_set_name)
    return train_set_name[:train_set_name.index(match.groups()[0])]


class GPMTrainer(MemoryTrainer):
    # Adapted from Gradient Projection Memory by Saha et al.

    def cache_proj_matrix(self, features, save_name: str):
        os.makedirs("./gpm_features/", exist_ok=True)
        file_name = f"./gpm_features/{save_name}_gpm.pickle"
        if os.path.exists(file_name):
            logger.info("Overwriting " + file_name)
        with open(file_name, "wb") as f:
            pickle.dump(features, f)
        logger.info(f"File {file_name} saved")

    def get_cached_proj_matrix(self, save_name: str) -> [torch.Tensor]:
        file_name = f"./gpm_features/{save_name}_gpm.pickle"
        with open(file_name, "rb") as f:
            features = pickle.load(f)
            logger.info(f"Cached GPM at {file_name} loaded")
        return features

    def resume_or_load(self, resume=True):
        # Load checkpoint
        super().resume_or_load(resume)

        self.device = torch.device(self.cfg.MODEL.DEVICE)

        self.model.fmap = FasterRCNNFeatureMap(
            isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel),
            self.cfg.SOLVER.IMS_PER_BATCH)

        USE_GPM_CACHE = hasattr(self.cfg, 'GPM_CACHE') and self.cfg.GPM_CACHE

        gpm_cache_loaded = False
        if USE_GPM_CACHE:
            try:
                # Load cached data
                self.feature_mat = self.get_cached_proj_matrix(get_base_ds_name(self.cfg.DATASETS.TRAIN[0]))
                # Move it to the GPU associated with this process
                for i in range(len(self.feature_mat)):
                    self.feature_mat[i] = self.feature_mat[i].to(self.device)
                gpm_cache_loaded = True
            except FileNotFoundError:
                logger.warning("GPM cache not found, calculating from scratch")
        if not USE_GPM_CACHE or not gpm_cache_loaded:
            hooks: [RemovableHandle] = register_feature_map_hooks(self.model)
            self.model.fmap.getting_conv_size = True

            # Create random data according to detectron2 format. Minimum image size.
            random_samples: [dict] = [create_random_sample((3, 640, 640))]

            determine_conv_output_sizes(self.model, random_samples)
            self.model.fmap.getting_conv_size = False

            self.model.fmap.clear_activations()
            self.calculate_activations()
            mat_dict = get_representation_matrix(self.model, self.model.device)
            features = update_GPM(mat_dict, self.model.fmap.threshold, features=dict())

            for hook_handle in hooks:
                hook_handle.remove()

            self.feature_mat = []
            # Projection Matrix Precomputation
            for layer_name in self.model.fmap.layer_names:
                Uf = torch.matmul(features[layer_name], features[layer_name].transpose(1, 0)).to(self.device)
                # print('Layer {} - Projection Matrix shape: {}'.format(layer_name, Uf.shape))
                self.feature_mat.append(Uf)
            # Average gradients across GPUs, reduce variance between samples of different mini-batches
            if get_world_size() > 1:
                for i in range(len(self.feature_mat)):
                    tensor = self.feature_mat[i].clone()
                    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                    self.feature_mat[i] = tensor
            # If on main process, cache the projection matrix
            if get_world_size() == 1 or get_rank() == 0:
                self.cache_proj_matrix(self.feature_mat, get_base_ds_name(self.cfg.DATASETS.TRAIN[0]))
        self.update_filter(load_base=False, load_novel=True)
        self.model.train()

    def step(self, _: [dict], novel_data: [dict]):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"

        loss_dict = self.model(novel_data)
        losses = sum(loss_dict.values())
        losses.backward()
        # GPM optimisation, from Saha et al. 2021
        feature_max_index = 0

        for layer_index, (m, params) in enumerate(self.model.named_parameters()):
            is_feature_layer = next(filter(lambda name: m.startswith(name), self.model.fmap.samples), None) is not None
            is_feature_weight = len(params.size()) != 1
            if is_feature_layer and is_feature_weight:
                sz = params.grad.data.size(0)
                layer_gradient = params.grad.data.to(self.device)
                params.grad.data = layer_gradient - torch.mm(layer_gradient.view(sz, -1),
                                                             self.feature_mat[feature_max_index]).view(params.size())
                feature_max_index += 1

        self.add_metrics(loss_dict)


    def calculate_activations(self):
        # Run model but ignore output, we only care about catching the activations through the capture_activation hook
        clock_start = time.perf_counter()
        # Sample check
        num_images_seen = 0
        # Make sure at least y instances are predicted through RPN, otherwise number of samples passing through ROI heads will be 0
        roi_activations = 0
        min_activations = min(self.model.fmap.samples.values())
        num_act_iterations = 0
        iterator = next(self.get_all_fs_base_samples())
        while (num_images_seen < min_activations or roi_activations < min_activations) and num_act_iterations < 200:
            logger.debug(f"{roi_activations}/{min_activations} activations found, continuing")
            try:
                mem_sample = next(iterator)
            except StopIteration:
                raise StopIteration("Error: reached the end of the dataset without extracting relevant proposals")
            example_out = self.model([mem_sample])
            num_images_seen += len(mem_sample)
            for example_image_results in example_out:
                roi_activations += len(example_image_results['instances'])
        clock_end_inf = time.perf_counter()
        logger.debug(f"Representation inference time: {clock_end_inf - clock_start}")
