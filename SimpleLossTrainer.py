import time

from MemoryTrainer import MemoryTrainer


class SimpleLossTrainer(MemoryTrainer):

    def run_step(self):
        assert self.model.training, "[SimpleLossTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = self.get_current_batch()
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        memory_data = self.get_memory_batch()
        memory_loss_dict = self.model(memory_data)
        memory_losses = sum(memory_loss_dict.values())

        total_loss = losses + memory_losses

        total_loss.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
