import logging
import time
from detectron2.utils.events import EventStorage

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


        with EventStorage() as storage:
            for i in range(len(self.data_loader.dataset.dataset)):
                loss_dict = self.model(self.get_memory_batch())
                losses = sum(loss_dict.values())
                losses.backward()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.optpar_dict[name] = param.data.clone()
            self.fisher_dict[name] = param.grad.data.clone().pow(2)

        logger.info(f"Found {len(self.fisher_dict)} parameter entries")
        self.optimizer.zero_grad()

    def run_step(self):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = self.get_current_batch()
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            fisher = self.fisher_dict[name]
            optpar = self.optpar_dict[name]
            losses += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda

        losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()

