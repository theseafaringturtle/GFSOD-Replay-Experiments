import logging

from MemoryTrainer import MemoryTrainer

logger = logging.getLogger("defrcn").getChild(__name__)


class ERTrainer(MemoryTrainer):
    # Experience Replay by Riemer et al.
    def __init__(self, cfg):
        cfg.defrost()
        cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER // 2
        cfg.SOLVER.STEPS = [x // 2 for x in cfg.SOLVER.STEPS]
        super().__init__(cfg)

    def step(self, mem_data, novel_data):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"

        self.optimizer.zero_grad()

        loss_dict = self.model(novel_data)
        loss = sum(loss_dict.values())

        memory_loss_dict = self.model(mem_data)
        memory_loss = sum(memory_loss_dict.values())

        total_loss = loss + memory_loss

        total_loss.backward()

        self.add_metrics({"memory_" + k: v for k, v in memory_loss_dict.items()})
        self.add_metrics(loss_dict)
