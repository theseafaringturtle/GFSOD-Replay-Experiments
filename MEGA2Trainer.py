import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer
from MemoryTrainer import MemoryTrainer


class MEGA2Trainer(MemoryTrainer):

    def run_step(self):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = self.get_current_batch()
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        self.current_loss = sum(loss_dict.values())
        self.current_loss.backward()

        self.current_gradient = self.get_gradient(self.model)

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_data = self.get_memory_batch()
        memory_loss_dict = self.model(memory_data)
        self.memory_loss = sum(memory_loss_dict.values())
        self.memory_loss.backward()

        self.memory_gradient = self.get_gradient(self.model)

        # MEGA-II
        sensitivity = 1e-10
        #
        self.deno1 = (torch.norm(self.current_gradient) * torch.norm(self.memory_gradient))
        self.num1 = (self.memory_gradient * self.current_gradient).sum()
        self.angle_tilda = torch.acos(self.num1 / self.deno1)

        thetas = []
        objectives = []

        for _ in range(3):
            # both thetas and objectives are random (0, pi)
            thetas.append((torch.rand(1) * torch.pi / 2).squeeze())
            objectives.append((torch.rand(1) * torch.pi / 2).squeeze())

        self.ratio = self.memory_loss / self.current_loss

        # Find an angle ˜θ that maximises: current_loss * cos(beta) + memory_loss * cos(theta − β).
        # MEGA-II's implementation is to sample 3 random ones then adjust them
        for idx in range(3):
            steps = 0
            # Adjust each theta[idx] 11 times
            while steps <= 10:  # note: this code runs 11 times in TF version as well
                theta = thetas[idx]
                theta = theta + (1 / (1 + self.ratio)) * (
                        -torch.sin(theta) + self.ratio * torch.sin(self.angle_tilda - theta))
                theta = torch.clamp(theta, min=0.0, max=0.5 * torch.pi)
                thetas[idx] = theta
                steps += 1

            objectives[idx] = self.current_loss * torch.cos(thetas[idx]) + self.memory_loss * torch.cos(
                self.angle_tilda - thetas[idx])

        objectives = torch.tensor(objectives)
        max_idx = torch.argmax(objectives)
        self.theta = thetas[max_idx]

        tr = (self.current_gradient * self.memory_gradient).sum()
        tt = (self.current_gradient * self.current_gradient).sum()
        rr = (self.memory_gradient * self.memory_gradient).sum()

        def compute_g_tilda(tr, tt, rr, flat_task_grads, flat_ref_grads):
            a = (rr * tt * torch.cos(self.theta) - tr * torch.norm(flat_task_grads) * torch.norm(
                flat_ref_grads) * torch.cos(self.angle_tilda - self.theta)) / self.deno
            b = (-tr * tt * torch.cos(self.theta) + tt * torch.norm(flat_task_grads) * torch.norm(
                flat_ref_grads) * torch.cos(self.angle_tilda - self.theta)) / self.deno
            return a * flat_task_grads + b * flat_ref_grads

        self.deno = tt * rr - tr * tr

        if self.deno >= sensitivity:
            g_tilda = compute_g_tilda(tr, tt, rr, self.current_gradient, self.memory_gradient)
            self.update_gradient(self.model, g_tilda)
        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
