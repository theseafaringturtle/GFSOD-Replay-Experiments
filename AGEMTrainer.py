from MemoryTrainer import MemoryTrainer


class AGEMTrainer(MemoryTrainer):

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

        # Inequality check. If the angle between gradients is >90, they are pointing in different directions.
        # Dot product is a shorthand to see if the vectors are in a different sector, without computing the angle.
        dot_prod = (self.current_gradient * self.memory_gradient).sum().item()  # gb . gn
        if dot_prod < 0.0:
            # Project current gradient onto memory gradient
            memory_mag_squared = (self.memory_gradient * self.memory_gradient).sum()
            grad_proj = self.current_gradient - (dot_prod / (memory_mag_squared + self.eps_agem)) * self.memory_gradient
            self.update_gradient(self.model, grad_proj)
        else:
            self.update_gradient(self.model, self.current_gradient)
