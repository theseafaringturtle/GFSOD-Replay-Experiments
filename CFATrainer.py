import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer
from MemoryTrainer import MemoryTrainer


class CFATrainer(MemoryTrainer):

    def run_step(self):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = self.get_current_batch()
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        losses.backward()

        self.current_gradient = self.get_gradient(self.model)

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_data = self.get_memory_batch()
        memory_loss_dict = self.model(memory_data)
        memory_losses = sum(memory_loss_dict.values())
        memory_losses.backward()

        self.memory_gradient = self.get_gradient(self.model)

        # Angle check and projection, following CFA Algo 1
        dot_prod = (self.current_gradient * self.memory_gradient).sum().item()  # gb . gn
        if dot_prod >= 0:
            grad_avg = (self.memory_gradient + self.current_gradient) / 2.0
            self.update_gradient(self.model, grad_avg)
        else:
            gb_mag_sq = (self.memory_gradient * self.memory_gradient).sum().item()
            gn_mag_sq = (self.current_gradient * self.current_gradient).sum().item()
            # Average the projections of gn-on-gb and gb-on-gn, formula provided in paper's algo is simplified to
            # g = (gn - (gn.gb) / (gb.gb) * gb +  gb - (gb.gn) / (gn.gn) * gn ) / 2
            # =  ( gn * (1 - (gb.gn) / (gn.gn) + gb * (1 - (gn.gb) / (gb.gb)) / 2
            grad_proj = 0.5 * (1 - (dot_prod / (gb_mag_sq + self.eps_agem))) * self.memory_gradient \
                        + 0.5 * (1 - (dot_prod / (gn_mag_sq + self.eps_agem))) * self.current_gradient
            self.update_gradient(self.model, grad_proj)

        self._write_metrics(loss_dict, data_time)

        self.optimizer.step()

    def vec_angle(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        mag1 = torch.linalg.norm(v1)
        mag2 = torch.linalg.norm(v2)
        dot_prod = torch.dot(v1, v2)
        return torch.acos(dot_prod / (mag1 * mag2))
