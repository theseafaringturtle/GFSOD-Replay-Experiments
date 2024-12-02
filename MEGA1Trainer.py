import torch
import math

from MemoryTrainer import MemoryTrainer


class MEGA1Trainer(MemoryTrainer):

    def step(self, mem_data, novel_data):
        assert self.model.training, f"[{self.__class__}] model was changed to eval mode!"

        # Calculate current gradients
        loss_dict = self.model(novel_data)
        loss = sum(loss_dict.values())
        loss.backward()

        self.current_gradient = self.get_gradient(self.model)

        loss = loss.detach()

        # Calculate memory gradients
        self.optimizer.zero_grad()

        memory_loss_dict = self.model(mem_data)
        memory_loss = sum(memory_loss_dict.values())
        memory_loss.backward()

        self.memory_gradient = self.get_gradient(self.model)

        memory_loss = memory_loss.detach() # avoid backpropagation on next formulae

        # MEGA-I
        sensitivity = 1e-10

        if loss > sensitivity:  # proper implementation?
            # alpha_1 is 1, alpha_2 is the loss ratio
            new_grad = self.current_gradient + self.memory_gradient * (memory_loss / loss).item()
            self.update_gradient(self.model, new_grad)
        else:
            # alpha_1 is 0, alpha_2 is 1
            self.update_gradient(self.model, self.memory_gradient)

        self.add_metrics({"memory_" + k: v for k, v in memory_loss_dict.items()})
        self.add_metrics(loss_dict)
