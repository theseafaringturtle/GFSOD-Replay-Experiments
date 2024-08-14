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

    def resume_or_load(self, resume=True):
        # Load checkpoint
        super().resume_or_load(resume)

        self.update_filter(load_base=True, load_novel=False)

        with EventStorage() as storage:
            for base_sample in self.get_all_fs_base_samples():
                loss_dict = self.model([base_sample])
                losses = sum(loss_dict.values())
                losses.backward()

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
