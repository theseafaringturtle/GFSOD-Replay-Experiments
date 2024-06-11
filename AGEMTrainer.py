import time
import re
import torch

from DeFRCNTrainer import DeFRCNTrainer


class AGEMTrainer(DeFRCNTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        memory_config = self.cfg.clone()
        memory_config.defrost()
        # Assuming only 1 dataset at a time, as per usual, e.g. (voc_2007_trainval_novel1_2shot_seed0,)
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

        # Inequality check. If the angle between gradients is >90, they are pointing in different directions.
        # Dot product is a shorthand to see if the vectors are in a different sector, without computing the angle.
        dot_prod = (self.current_gradient * self.memory_gradient).sum().item()  # gb . gn
        if dot_prod < 0.0:
            # Project current gradient onto memory gradient
            length_rep = (self.memory_gradient * self.memory_gradient).sum()
            grad_proj = self.current_gradient - (dot_prod / (length_rep + self.eps_agem)) * self.memory_gradient
            self.update_gradient(self.model, grad_proj)
        else:
            self.update_gradient(self.model, self.current_gradient)

        self._write_metrics(loss_dict, data_time)

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
