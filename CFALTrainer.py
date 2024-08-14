import torch

from MemoryTrainer import MemoryTrainer


class CFALTrainer(MemoryTrainer):
    # An attempt at combining CFA and MEGA-I

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_agem = 1e-7

    def step(self, mem_data, novel_data):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"

        # Calculate current gradients
        loss_dict = self.model(novel_data)
        losses = sum(loss_dict.values())
        losses.backward()

        self.current_gradient = self.get_gradient(self.model)

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_loss_dict = self.model(mem_data)
        memory_losses = sum(memory_loss_dict.values())
        memory_losses.backward()

        self.memory_gradient = self.get_gradient(self.model)

        if self.iter < self.cfg.SOLVER.WARMUP_ITERS:
            ratio = 0.5
        else:
            x = losses / (losses + memory_losses)
            a = 2
            ratio = torch.tanh(a * (x - 0.5)) * 0.1 + 0.5

        # Angle check and projection, following CFA Algo 1
        dot_prod = (self.current_gradient * self.memory_gradient).sum().detach()  # gb . gn
        if dot_prod >= 0:
            grad_avg = (1 - ratio) * self.memory_gradient + (ratio) * self.current_gradient
            self.update_gradient(self.model, grad_avg)
        else:
            gb_mag_sq = (self.memory_gradient * self.memory_gradient).sum().item()
            gn_mag_sq = (self.current_gradient * self.current_gradient).sum().item()
            # Average the projections of gn-on-gb and gb-on-gn, formula provided in paper's algo is simplified to
            # g = (gn - (gn.gb) / (gb.gb) * gb +  gb - (gb.gn) / (gn.gn) * gn ) / 2
            # =  ( gn * (1 - (gb.gn) / (gn.gn) + gb * (1 - (gn.gb) / (gb.gb)) / 2
            grad_proj = (1 - ratio) * (1 - (dot_prod / (gb_mag_sq + self.eps_agem))) * self.memory_gradient \
                        + ratio * (1 - (dot_prod / (gn_mag_sq + self.eps_agem))) * self.current_gradient
            self.update_gradient(self.model, grad_proj)
