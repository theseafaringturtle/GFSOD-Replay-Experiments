import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer
from MemoryTrainer import MemoryTrainer


class CFATrainer(MemoryTrainer):

    def run_step(self):
        assert self.model.training, "[CFATrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        losses.backward()

        self.current_gradient = self.get_gradient(self.model)

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_data = next(self._memory_loader_iter)
        # TODO fix coco's base classes since they're not contiguous unlike VOC
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
            gb_mag = (self.memory_gradient * self.memory_gradient).sum().item()
            gn_mag = (self.current_gradient * self.current_gradient).sum().item()
            # Average the projections of gn-on-gb and gb-on-gn, formula provided in paper's algo is simplified to
            # g = (gn - (gn.gb) / (gb.gb) * gb +  gb - (gb.gn) / (gn.gn) * gn ) / 2
            # =  ( gn * (1 - (gb.gn) / (gn.gn) + gb * (1 - (gn.gb) / (gb.gb)) / 2
            grad_proj = 0.5 * (1 - (dot_prod / (gb_mag + self.eps_agem))) * self.memory_gradient \
                        + 0.5 * (1 - (dot_prod / (gn_mag + self.eps_agem))) * self.current_gradient
            self.update_gradient(self.model, grad_proj)


        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

