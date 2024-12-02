import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from .MemoryTrainer import MemoryTrainer


class CAGTrainer(MemoryTrainer):
    # From: Conflict-Averse Gradient Descent for Multitask Learning (CAGrad) by Liu et al.

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

        grad_res = self.cagrad()

        self.update_gradient(self.model, grad_res)

        self.add_metrics({"memory_" + k: v for k, v in memory_loss_dict.items()})
        self.add_metrics(loss_dict)

    def cagrad(self, c=0.5):
        avg_gradient = (self.memory_gradient + self.current_gradient) / 2

        gb_mag_sq = self.memory_gradient.dot(self.memory_gradient).item()
        dot_prod = self.memory_gradient.dot(self.current_gradient).item()
        gn_mag_sq = self.current_gradient.dot(self.current_gradient).item()

        g0_norm = 0.5 * np.sqrt(gb_mag_sq + gn_mag_sq + 2 * dot_prod + 1e-4)

        # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
        coef = c * g0_norm

        def obj(x):
            # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
            # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
            return coef * np.sqrt(
                x ** 2 * (gb_mag_sq + gn_mag_sq - 2 * dot_prod) + 2 * x * (dot_prod - gn_mag_sq) + gn_mag_sq + 1e-4) + \
                   0.5 * x * (gb_mag_sq + gn_mag_sq - 2 * dot_prod) + (0.5 + x) * (dot_prod - gn_mag_sq) + gn_mag_sq

        res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
        x = res.x

        gw = x * self.memory_gradient + (1 - x) * self.current_gradient
        gw_norm = np.sqrt(x ** 2 * gb_mag_sq + (1 - x) ** 2 * gn_mag_sq + 2 * x * (1 - x) * dot_prod + 1e-4)

        lmbda = coef / (gw_norm + 1e-4)
        g = avg_gradient + lmbda * gw
        return g / (1 + c)
