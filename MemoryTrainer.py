import re

import torch
from detectron2.data import MetadataCatalog
from torch import Tensor

from DeFRCNTrainer import DeFRCNTrainer
from defrcn.data.builtin_meta import coco_contiguous_id_to_class_id, voc_contiguous_id_to_class_id


class MemoryTrainer(DeFRCNTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        memory_config = self.cfg.clone()
        memory_config.defrost()
        # Assuming only 1 dataset at a time, as per usual, e.g. (voc_2007_trainval_novel1_2shot_seed0,)
        train_set_name = memory_config.DATASETS.TRAIN[0]
        name_and_shots, seed = train_set_name.split("_seed")
        # Use only base classes, add 10 to seed, so if we're running novel 0-9 we'll get base memory from 10-19
        # It's a quick way to make sure we're using different images for base memory, just like in CFA
        new_train_set_name = f"{re.sub('novel|all', 'base', name_and_shots)}_seed{int(seed)}"
        memory_config.DATASETS.TRAIN = [new_train_set_name]
        print(f"Using {new_train_set_name} instead of {train_set_name} for memory")
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)
        self.memory_config = memory_config

        # Numerical stability param for A-GEM and CFA, taken from VanDeVen's A-GEM implementation
        self.eps_agem = 1e-7

    def get_memory_batch(self):
        memory_data = next(self._memory_loader_iter)
        self.adjust_batch_ids(self.memory_config.DATASETS.TRAIN[0], memory_data)
        return memory_data

    def get_current_batch(self):
        current_data = next(self._data_loader_iter)
        self.adjust_batch_ids(self.cfg.DATASETS.TRAIN[0], current_data)
        return current_data

    def adjust_batch_ids(self, train_set_name, data):
        # Adjust memory data IDs
        # VOC does not need adjustment, since first 15 contiguous class IDs are the base ones
        if "voc" in train_set_name:
            for sample in data:
                classes: Tensor = sample['instances'].get('gt_classes')  # CPU tensor containing class IDs for instances
                sample['instances'].set('gt_classes', voc_contiguous_id_to_class_id(train_set_name, classes))
            return data
        elif "coco" in train_set_name:
            for sample in data:
                classes: Tensor = sample['instances'].get('gt_classes')  # ditto
                sample['instances'].set('gt_classes', coco_contiguous_id_to_class_id(train_set_name, classes))
            return data
        else:
            raise NotImplementedError("For custom datasets, specify here how a contiguous ID is mapped to a class ID")

    def get_gradient(self, model):
        gradient = []
        for p in model.parameters():
            if p.requires_grad:
                gradient.append(p.grad.view(-1))
        return torch.cat(gradient)

    def update_gradient(self, model, new_grad):
        index = 0
        for p in model.parameters():
            if p.requires_grad:
                n_param = p.numel()  # number of parameters in [p]
                p.grad.copy_(new_grad[index:index + n_param].view_as(p))
                index += n_param
