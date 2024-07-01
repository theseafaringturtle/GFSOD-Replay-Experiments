import random
import time
import re
import torch
from detectron2.utils.comm import get_world_size

import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer
from MemoryTrainer import MemoryTrainer


class AlterTrainer(MemoryTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.minibatch_size = cfg.SOLVER.IMS_PER_BATCH // get_world_size()

    def run_step(self):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"
        start = time.perf_counter()

        # Alternate between base and novel batches, simple as that
        if self.iter % self.minibatch_size * 2> self.minibatch_size:
            # Calculate current gradients
            self.optimizer.zero_grad()

            data = self.get_current_batch()
            data_time = time.perf_counter() - start

            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            losses.backward()
            self._write_metrics(loss_dict, data_time)
        else:
            # Calculate memory gradients
            self.optimizer.zero_grad()

            memory_data = self.get_memory_batch()
            data_time = time.perf_counter() - start

            memory_loss_dict = self.model(memory_data)
            memory_losses = sum(memory_loss_dict.values())
            memory_losses.backward()
            self._write_metrics(memory_loss_dict, data_time)

        self.optimizer.step()
