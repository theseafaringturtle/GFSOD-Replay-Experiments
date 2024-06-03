import time
import re
from typing import Tuple

import numpy as np
import torch
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.hooks import RemovableHandle

from DeFRCNTrainer import DeFRCNTrainer
from GPM import FeatureMap, get_representation_matrix, update_GPM, register_feature_map_hooks, \
    determine_conv_output_sizes


class FasterRCNNFeatureMap(FeatureMap):
    def __init__(self, multi_gpu=False, minibatch_size: int = 8):
        super().__init__()
        # Detector layers only for now
        # self.layer_names = ['backbone.res4.10.conv1', 'backbone.res4.10.conv2', 'backbone.res4.10.conv3',
        #                     'backbone.res4.11.conv1', 'backbone.res4.11.conv2', 'backbone.res4.11.conv3',
        #                     'backbone.res4.12.conv1', 'backbone.res4.12.conv2', 'backbone.res4.12.conv3']
        self.layer_names = ['backbone.res4.0.conv1', 'backbone.res4.0.conv2', 'backbone.res4.0.conv3',
                            'proposal_generator.rpn_head.conv']

        # Workaround for documented behaviour: https://github.com/pytorch/pytorch/issues/9176
        if multi_gpu:
            self.layer_names = ["module." + name for name in self.layer_names]
        self.samples = {name: minibatch_size for name in self.layer_names}


def create_random_sample(max_size: Tuple[int, int, int]) -> dict:
    sample = dict()
    random_tensor = torch.rand(max_size)
    sample['image'] = random_tensor
    sample['width'] = max_size[1]
    sample['height'] = max_size[2]
    sample['image_id'] = f'777777{torch.rand(1).item() * 100:.0f}'
    sample['file_name'] = sample['image_id'] + '.jpg'
    sample['instances'] = Instances(max_size[1:])
    return sample


class GPMTrainer(DeFRCNTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        # While it's the same code as memory methods,
        # this is for getting the base samples to build initial representation matrix.
        # Memory is not used in optimisation loop
        memory_config = self.cfg.clone()
        memory_config.defrost()
        train_set_name = memory_config.DATASETS.TRAIN[0]
        name_and_shots, seed = train_set_name.split("_seed")
        memory_config.DATASETS.TRAIN = [f"{re.sub('novel|all', 'base', name_and_shots)}_seed{int(seed) + 10}"]
        print(f"Using {memory_config.DATASETS.TRAIN} instead of {train_set_name} for memory")
        # Use same number of shots to be the same as k used in normal config, but different base images
        self.memory_loader = self.build_train_loader(memory_config)
        self._memory_loader_iter = iter(self.memory_loader)

        self.device = torch.device(self.cfg.MODEL.DEVICE)

        self.model.fmap = FasterRCNNFeatureMap(
            isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel),
            self.cfg.SOLVER.IMS_PER_BATCH // get_world_size()
        )

        hooks: [RemovableHandle] = register_feature_map_hooks(self.model)

        # Create random data according to detectron2 format
        random_samples: [dict] = [create_random_sample((3, 1300, 800))]

        # next(self._memory_loader_iter)
        determine_conv_output_sizes(self.model, [next(self._memory_loader_iter)[0]])
        self.model.fmap.clear_activations()

        threshold = {name: 0.99 for name in self.model.fmap.layer_names}

        mat_dict = get_representation_matrix(self.model, next(self._memory_loader_iter))
        features = update_GPM(self.model, mat_dict, threshold, features=dict())

        for hook_handle in hooks:
            hook_handle.remove()

        self.feature_mat = []
        # Projection Matrix Precomputation
        for layer_name in self.model.fmap.layer_names:
            Uf = torch.matmul(features[layer_name], features[layer_name].transpose(1, 0)).to(self.device)
            print('Layer {} - Projection Matrix shape: {}'.format(layer_name, Uf.shape))
            self.feature_mat.append(Uf)
        self.model.train()

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

        # self.current_gradient = self.get_gradient(self.model)

        # Gradient projections
        feature_max_index = 0

        for layer_index, (m, params) in enumerate(self.model.named_parameters()):
            is_feature_layer = next(filter(lambda name: m.startswith(name), self.model.fmap.samples), None) is not None
            is_feature_weight = len(params.size()) != 1
            if is_feature_layer and is_feature_weight:
                sz = params.grad.data.size(0)
                layer_gradient = params.grad.data.to(self.device)
                params.grad.data = layer_gradient - torch.mm(layer_gradient.view(sz, -1),
                                                             self.feature_mat[feature_max_index]).view(params.size())
                feature_max_index += 1
            # Fill biases and BN with 0s when fine-tuning
            elif not is_feature_weight:
                params.grad.data.fill_(0)

        # self.update_gradient(self.model, grad_proj)

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
