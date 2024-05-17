import time

import torch

from DeFRCNTrainer import DeFRCNTrainer


class CFATrainer(DeFRCNTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        memory_config = self.cfg.clone()
        memory_config.defrost()
        # Assuming only 1 dataset at a time, as per usual, e.g. (voc_2007_trainval_novel1_2shot_seed0,)
        train_set_name = memory_config.DATASETS.TRAIN[0]
        name_and_shots, seed = train_set_name.split("_seed")
        # Use only base classes, add 10 to seed, so if we're running novel 0-9 we'll get base memory from 10-19
        # It's a quick way to make sure we're using different images for base memory, just like in CFA
        memory_config.DATASETS.TRAIN = [f"{name_and_shots.replace('novel', 'base')}_seed{int(seed) + 10}"]
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)

        # Numerical stability param for A-GEM and CFA, taken from VanDeVen's A-GEM implementation
        self.eps_agem = 1e-7

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
