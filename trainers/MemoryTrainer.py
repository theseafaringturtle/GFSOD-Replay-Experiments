import re
import time
from random import shuffle
from typing import Dict, Iterator

import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import get_event_storage
from torch import Tensor

from .DeFRCNTrainer import DeFRCNTrainer


class MemoryTrainer(DeFRCNTrainer):
    """
    Note:
        This class sets up training with separate novel and memory batches, for easier comparison with CL methods.
        Optional methods get_gradient and update_gradient are offered for gradient manipulation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.minibatch_size = self.cfg.SOLVER.IMS_PER_BATCH // get_world_size()
        self.batches_seen = 0
        self.update_filter(load_base=True, load_novel=True)

    def update_filter(self, load_base: bool, load_novel: bool):
        self.load_base = load_base
        self.load_novel = load_novel
        """Update batch filter. See whether base, novel batches or both need to be filled"""
        if self.load_base and self.load_novel:
            condition = lambda novel_data, mem_data: len(novel_data) < self.minibatch_size or len(
                mem_data) < self.minibatch_size
        elif self.load_base:
            condition = lambda novel_data, mem_data: len(mem_data) < self.minibatch_size
        elif self.load_novel:
            condition = lambda novel_data, mem_data: len(novel_data) < self.minibatch_size
        else:
            raise Exception(
                "MemoryTrainer must load a batch, so one of these must be true: load_base, load_novel")
        # Add just-in-case safety catch if the dataset doesn't have enough data
        max_batch_limit = 500
        self.batch_condition = lambda novel_data, mem_data: \
            condition(novel_data, mem_data) and self.batches_seen < max_batch_limit

    def is_novel(self, sample):
        classes = sample.get("instances").get("gt_classes").tolist()
        if 'voc' in self.cfg.DATASETS.TRAIN[0]:
            return all([c >= 15 for c in classes])
        else:
            for c in classes:
                if c in MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get(
                        "base_dataset_id_to_contiguous_id").keys():
                    return False
            return True

    def run_step(self):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"
        start = time.perf_counter()

        # Fill up batches from `all` dataset
        novel_data = []
        mem_data = []

        while self.batch_condition(novel_data, mem_data):
            data = next(self._data_loader_iter)
            if self.load_novel and len(novel_data) < self.minibatch_size:
                novel_data += [d for d in data if self.is_novel(d)]
            if self.load_base and len(mem_data) < self.minibatch_size:
                mem_data += [d for d in data if not self.is_novel(d)]
            self.batches_seen += 1
        self.batches_seen = 0
        # print(f"Lengths: {len(mem_data)}, {len(novel_data)}")
        novel_data = novel_data[:self.minibatch_size]
        mem_data = mem_data[:self.minibatch_size]

        data_time = time.perf_counter() - start

        self.metrics_dict = {"data_time": data_time}

        self.optimizer.zero_grad()

        self.step(mem_data, novel_data)

        self.write_metrics(self.metrics_dict)

        self.optimizer.step()

    def step(self, mem_data: [dict], novel_data: [dict]):
        """
        Args:
            mem_data : list of base set samples with len = minibatch size (IMS_PER_BATCH / NUM_GPUS)
            novel_data: list of novel set samples, ditto
        Note:
            Method used for calculating the loss and gradients. Parent class sets up the data batches, and triggers the optimizer after this method
        """
        data = novel_data + mem_data
        # Random shuffle
        shuffle(data)
        # Trim batch to original size
        data = data[:self.minibatch_size]

        # Calculate loss
        loss_dict = self.model(data)
        loss = sum(loss_dict.values())
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

    def add_metrics(self, metrics: Dict):
        """
        Args:
            metrics_dict (dict): dict of values that will be logged at the end of the iteration with the existing ones.
        """
        self.metrics_dict = {**metrics, **self.metrics_dict}

    def write_metrics(self, metrics_dict: Dict, prefix: str = ""):
        """
        Args:
            metrics_dict (dict): dict of values, including scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): log prefix
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in metrics_dict.items() if isinstance(v, Tensor)}

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # Any delays such as data_time can have high variance, as mentioned by det2 authors. Pick the maximum one.
            for key in all_metrics_dict[0].keys():
                if key.endswith("_time"):
                    value = np.max([x.pop(key) for x in all_metrics_dict])
                    storage.put_scalar(key, value)

            # average the rest of the metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            # If memory loss is present, pop those key-values and sum them up
            memory_loss_dict = {k: metrics_dict.pop(k) for k in list(metrics_dict.keys()) if "memory_loss" in k}
            memory_losses_reduced = sum(memory_loss_dict.values())
            # Sum up current loss
            loss_dict = {k: v for k, v in metrics_dict.items() if "loss" in k}
            current_losses_reduced = sum(loss_dict.values())
            if not np.isfinite(current_losses_reduced) or not np.isfinite(memory_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )
            storage.put_scalar("{}total_loss".format(prefix), current_losses_reduced)
            if memory_losses_reduced:
                storage.put_scalar("{}memory_total_loss".format(prefix), memory_losses_reduced)
            # Log any additional metrics other than loss passed to this function
            storage.put_scalars(**{k: v for k, v in metrics_dict.items() if "loss" not in k})

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

    def get_all_fs_base_samples(self) -> Iterator:
        """
        Notes:
            Returns a sequential iterator over base samples in the set. Should be called on a single process.
            This function currently processes images one at a time, filtering by base classes. No images must be skipped.
            A good alternative would be to use a sync queue for this.
        """
        memory_cfg = self.cfg.clone()
        memory_cfg.defrost()
        memory_cfg.DATALOADER.SAMPLER_TRAIN = "FiniteTrainingSampler"
        memory_cfg.SOLVER.IMS_PER_BATCH = comm.get_world_size()
        memory_loader = self.build_train_loader(memory_cfg)
        iterator = iter(memory_loader)
        dataset_length = len(memory_loader.dataset.dataset)
        for i in range(dataset_length):
            # Since we've set IMS_PER_BATCH = 1, pick first
            samples = next(iterator)
            samples = [sample for sample in samples if not self.is_novel(sample)]
            # Empty after filtering
            if not samples:
                continue
            else:
                # print([sample.get("file_name") for sample in samples])
                yield samples
