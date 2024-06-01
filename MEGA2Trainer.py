import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer


class MEGA2Trainer(DeFRCNTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        memory_config = self.cfg.clone()
        memory_config.defrost()
        # Assuming only 1 dataset at a time, as per usual, e.g. (voc_2007_trainval_all1_2shot_seed0,)
        train_set_name = memory_config.DATASETS.TRAIN[0]
        name_and_shots, seed = train_set_name.split("_seed")
        # Use only base classes, add 10 to seed, so if we're running novel 0-9 we'll get base memory from 10-19
        # It's a quick way to make sure we're using different images for base memory, just like in CFA
        memory_config.DATASETS.TRAIN = [f"{re.sub('novel|all', 'base', name_and_shots)}_seed{int(seed) + 10}"]
        print(f"Using {memory_config.DATASETS.TRAIN} instead of {train_set_name} for memory")
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)

        # Numerical stability param for A-GEM and CFA, taken from VanDeVen's A-GEM implementation
        self.eps_agem = 1e-7

    def run_step(self):
        assert self.model.training, "[MEGA2Trainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        self.current_loss = sum(loss_dict.values())
        self.current_loss.backward()

        self.current_gradient = self.get_gradient(self.model)

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_data = next(self._memory_loader_iter)
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


    def get_gradient(self, model):
        gradient = []
        for p in model.parameters():
            if p.requires_grad:
                gradient.append(p.grad.view(-1))
        return torch.cat(gradient)

    def update_gradient(self, model, new_grad):
        index = 0
        for p in model.parameters():
            if p.requires_grad:
                n_param = p.numel()  # number of parameters in [p]
                p.grad.copy_(new_grad[index:index + n_param].view_as(p))
                index += n_param
