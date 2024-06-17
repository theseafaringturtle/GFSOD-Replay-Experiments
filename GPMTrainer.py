import time
import re
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from detectron2.structures import Instances
from detectron2.utils.comm import get_world_size  # , get_rank, reduce_dict
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.hooks import RemovableHandle
import torch.distributed as dist
from DeFRCNTrainer import DeFRCNTrainer
from GPM import FeatureMap, get_representation_matrix, update_GPM, register_feature_map_hooks, \
    determine_conv_output_sizes
from MemoryTrainer import MemoryTrainer


def reduce_dict(input_dict, average=True):
    """
    Same as detectron's reduce_dict, but without stacking the values, which only worked for them in the loss use case
    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        if isinstance(input_dict, OrderedDict):
            for k in input_dict.keys():
                names.append(k)
                input_dict[k] = input_dict[k].cuda()
                values.append(input_dict[k])
        else:
            for k in sorted(input_dict.keys()):
                names.append(k)
                input_dict[k] = input_dict[k].cuda()
                values.append(input_dict[k])
        # values = torch.stack(values, dim=0)
        dist.barrier()
        for i in range(len(values)):
            print(names[i])
            dist.all_reduce(values[i], op=dist.ReduceOp.AVG)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class FasterRCNNFeatureMap(FeatureMap):
    def __init__(self, multi_gpu=False, minibatch_size: int = 8):
        super().__init__()
        self.layer_names = [f'backbone.res4.{i}.conv1' for i in range(23)] + \
                           [f'backbone.res4.{i}.conv2' for i in range(23)] + \
                           [f'backbone.res4.{i}.conv3' for i in range(23)]
        self.layer_names = sorted(self.layer_names)
        # Workaround for documented behaviour: https://github.com/pytorch/pytorch/issues/9176
        if multi_gpu:
            self.layer_names = ["module." + name for name in self.layer_names]
        self.samples = {name: minibatch_size for name in self.layer_names}
        self.threshold = {name: (0.8 if not name.endswith('conv3') else 0.7) for name in self.layer_names}
        # self.threshold['proposal_generator.rpn_head.conv'] = 0.6


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


class GPMTrainer(MemoryTrainer):

    def __init__(self, cfg):
        # While it's the same code as memory methods, this is for getting the base samples to build initial representation matrix.
        # Memory is not used in optimisation loop
        super().__init__(cfg)

    def resume_or_load(self, resume=True):
        # Load checkpoint
        super().resume_or_load(resume)

        self.device = torch.device(self.cfg.MODEL.DEVICE)

        self.model.fmap = FasterRCNNFeatureMap(
            isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel),
            self.cfg.SOLVER.IMS_PER_BATCH // get_world_size()
        )

        hooks: [RemovableHandle] = register_feature_map_hooks(self.model)

        # Create random data according to detectron2 format
        random_samples: [dict] = [create_random_sample((3, 1300, 800))]

        # next(self._memory_loader_iter)
        # determine_conv_output_sizes(self.model, [next(self._memory_loader_iter)[0]])
        determine_conv_output_sizes(self.model, random_samples)
        self.model.fmap.clear_activations()
        self.calculate_activations()
        mat_dict = get_representation_matrix(self.model)
        features = update_GPM(self.model, mat_dict, self.model.fmap.threshold, features=dict())

        for hook_handle in hooks:
            hook_handle.remove()

        self.feature_mat = []
        # Projection Matrix Precomputation
        for layer_name in self.model.fmap.layer_names:
            Uf = torch.matmul(features[layer_name], features[layer_name].transpose(1, 0)).to(self.device)
            print('Layer {} - Projection Matrix shape: {}'.format(layer_name, Uf.shape))
            self.feature_mat.append(Uf)
        # Average gradients across GPUs
        if get_world_size() > 0:
            for i in range(len(self.feature_mat)):
                tensor = self.feature_mat[i].clone()
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                self.feature_mat[i] = tensor
        self.model.train()

    def run_step(self):
        assert self.model.training, "[CFATrainer] model was changed to eval mode!"
        start = time.perf_counter()

        # Calculate current gradients
        self.optimizer.zero_grad()

        data = self.get_current_batch()
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

        # self.update_gradient(self.model, grad_proj)

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

    def calculate_activations(self):
        # Run model but ignore output, we only care about catching the activations through the capture_activation hook
        clock_start = time.perf_counter()
        # Sanity check
        # Make sure at least y instances are predicted through RPN, otherwise number of samples passing through ROI heads will be 0
        roi_activations = 0
        min_activations = min(self.model.fmap.samples.values())
        num_act_iterations = 0
        while roi_activations < min_activations and num_act_iterations < 200:
            print(f"{roi_activations}/{min_activations} activations found, continuing")
            example_out = self.model(self.get_memory_batch())
            for example_image_results in example_out:
                roi_activations += len(example_image_results['instances'])
        clock_end_inf = time.perf_counter()
        print(f"Representation inference time: {clock_end_inf - clock_start}")
