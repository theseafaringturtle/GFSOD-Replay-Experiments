import time

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
        memory_config.DATASETS.TRAIN = [f"{name_and_shots.replace('novel', 'base')}_{int(seed) + 10}"]
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)

    def run_step(self):
        assert self.model.training, "[CFATrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        memory_data = next(self._data_loader_iter)

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
