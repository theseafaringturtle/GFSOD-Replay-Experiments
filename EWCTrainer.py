import logging
import time

import torch
from detectron2.utils.comm import get_world_size
from detectron2.utils.events import EventStorage

from detectron2.utils import comm
from MemoryTrainer import MemoryTrainer

logger = logging.getLogger("defrcn").getChild(__name__)


class EWCTrainer(MemoryTrainer):

    # EWC from Kirkpatrick et al. 2016, implementation taken from ContinualAI

    def __init__(self, cfg):
        super().__init__(cfg)
        # Memory only used once after base training to compute Fisher Information Matrix
        self.fisher_dict = {}
        self.optpar_dict = {}
        self.ewc_lambda = 0.4

    def resume_or_load(self, resume=True):
        # Load checkpoint
        super().resume_or_load(resume)

        logger.info("Retrieving FIM...")
        start = time.perf_counter()

        self.update_filter(load_base=True, load_novel=False)

        # Accumulate base gradients
        num_samples_run = 0
        with EventStorage() as storage:
            for base_samples in self.get_all_fs_base_samples():
                loss_dict = self.model(base_samples)
                losses = sum(loss_dict.values())
                losses.backward()
                num_samples_run += 1
                logger.info(f"Samples parsed on GPU {comm.get_rank()}: {num_samples_run}")
            # Scatter accumulated gradients to other machines
        if get_world_size() > 1:
            logger.info(f"Gathering from GPU {comm.get_rank()}")
            grad = self.get_gradient(self.model)
            grads_list = comm.all_gather(grad)
            dist_accumulated_grads = sum(grads_list)
            self.update_gradient(self.model, dist_accumulated_grads)

        # Calculate Fisher Information Matrix
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.optpar_dict[name] = param.data.clone()
            self.fisher_dict[name] = param.grad.data.clone().pow(2)

        logger.info(f"Found {len(self.fisher_dict)} parameter entries")
        # Clear accumulated gradients
        self.optimizer.zero_grad()
        # Use only novel data
        self.update_filter(load_base=False, load_novel=True)

        end = time.perf_counter() - start
        logger.info(f"FIM retrieved in {end}s on {num_samples_run} samples")

    def step(self, _: [dict], novel_data: [dict]):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"

        loss_dict = self.model(novel_data)
        losses = sum(loss_dict.values())
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            fisher = self.fisher_dict[name]
            optpar = self.optpar_dict[name]
            losses += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda
        losses.backward()
        self.write_metrics(loss_dict)
