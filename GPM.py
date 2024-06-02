import time
from abc import ABC
from collections import OrderedDict
from functools import partial
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

backbone_layers = ['backbone.stem.conv1', 'backbone.res2.0.shortcut', 'backbone.res2.0.conv1', 'backbone.res2.0.conv2',
                   'backbone.res2.0.conv3', 'backbone.res2.1.conv1', 'backbone.res2.1.conv2', 'backbone.res2.1.conv3',
                   'backbone.res2.2.conv1', 'backbone.res2.2.conv2', 'backbone.res2.2.conv3',
                   'backbone.res3.0.shortcut', 'backbone.res3.0.conv1', 'backbone.res3.0.conv2',
                   'backbone.res3.0.conv3', 'backbone.res3.1.conv1', 'backbone.res3.1.conv2', 'backbone.res3.1.conv3',
                   'backbone.res3.2.conv1', 'backbone.res3.2.conv2', 'backbone.res3.2.conv3', 'backbone.res3.3.conv1',
                   'backbone.res3.3.conv2', 'backbone.res3.3.conv3', 'backbone.res4.0.shortcut',
                   'backbone.res4.0.conv1', 'backbone.res4.0.conv2', 'backbone.res4.0.conv3',
                   'backbone.res4.1.conv1', 'backbone.res4.1.conv2', 'backbone.res4.1.conv3',
                   'backbone.res4.2.conv1', 'backbone.res4.2.conv2', 'backbone.res4.2.conv3', 'backbone.res4.3.conv1',
                   'backbone.res4.3.conv2', 'backbone.res4.3.conv3', 'backbone.res4.4.conv1', 'backbone.res4.4.conv2',
                   'backbone.res4.4.conv3', 'backbone.res4.5.conv1', 'backbone.res4.5.conv2', 'backbone.res4.5.conv3',
                   'backbone.res4.6.conv1', 'backbone.res4.6.conv2', 'backbone.res4.6.conv3', 'backbone.res4.7.conv1',
                   'backbone.res4.7.conv2', 'backbone.res4.7.conv3', 'backbone.res4.8.conv1', 'backbone.res4.8.conv2',
                   'backbone.res4.8.conv3', 'backbone.res4.9.conv1', 'backbone.res4.9.conv2', 'backbone.res4.9.conv3',
                   'backbone.res4.10.conv1', 'backbone.res4.10.conv2', 'backbone.res4.10.conv3',
                   'backbone.res4.11.conv1', 'backbone.res4.11.conv2', 'backbone.res4.11.conv3',
                   'backbone.res4.12.conv1', 'backbone.res4.12.conv2', 'backbone.res4.12.conv3',
                   'backbone.res4.13.conv1', 'backbone.res4.13.conv2', 'backbone.res4.13.conv3',
                   'backbone.res4.14.conv1', 'backbone.res4.14.conv2', 'backbone.res4.14.conv3',
                   'backbone.res4.15.conv1', 'backbone.res4.15.conv2', 'backbone.res4.15.conv3',
                   'backbone.res4.16.conv1', 'backbone.res4.16.conv2', 'backbone.res4.16.conv3',
                   'backbone.res4.17.conv1', 'backbone.res4.17.conv2', 'backbone.res4.17.conv3',
                   'backbone.res4.18.conv1', 'backbone.res4.18.conv2', 'backbone.res4.18.conv3',
                   'backbone.res4.19.conv1', 'backbone.res4.19.conv2', 'backbone.res4.19.conv3',
                   'backbone.res4.20.conv1', 'backbone.res4.20.conv2', 'backbone.res4.20.conv3',
                   'backbone.res4.21.conv1', 'backbone.res4.21.conv2', 'backbone.res4.21.conv3',
                   'backbone.res4.22.conv1', 'backbone.res4.22.conv2', 'backbone.res4.22.conv3']


def compute_conv_output_size(input_size: Tuple[int, int], kernel_size: Tuple[int, int],
                             stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    conv_out_size = []
    for dim in range(2):
        size_dim = int(np.floor((input_size[dim] + 2 * padding[dim] - dilation[dim] * (kernel_size[dim] - 1) - 1)))
        conv_out_size.append(size_dim)
    return conv_out_size


def register_feature_map_hooks(model: nn.Module) -> [RemovableHandle]:
    def capture_activation(layer_name: str, module: nn.Module, input: Tensor):
        model.fmap.set_activation_for_layer(layer_name, input[0])

    hook_handles = []
    for name, module in model.named_modules():
        if name in model.fmap.layer_names:
            handle = module.register_forward_pre_hook(partial(capture_activation, name))
            hook_handles.append(handle)
            model.fmap.set_module(name, module)
    model.fmap.activations_hooks_registered = True
    return hook_handles


def determine_conv_output_sizes(model: nn.Module, random_samples):
    """Run a single sample through network, get output size of all the convolutions we've marked.
    This is useful in case we don't just pick sequential convolution in the network
    :param: model: full model, with layer_names already marked in model.fmap
    :param: random_samples: data in the format the model expects. Can be a tensor of tensors (classification) or a list (detectron2).
    random_samples should use the maximum input image size to the network, including RGB channels
    """
    assert hasattr(model, 'fmap'), "Model needs feature map storage"
    assert model.fmap.activations_hooks_registered is True, "Need to register hooks to sizes of convolutional layers across the model"
    out_sizes: Dict[str, (int, int)] = {}

    model.fmap.clear_activations()
    model.eval()
    with torch.no_grad():
        # According to pytorch, mini-batch stats are used in training mode, and in eval mode when buffers are None.
        # since we're not tracking stats as per GPM, we need at least 2 samples
        model(random_samples)
    for layer_name in model.fmap.layer_names:
        output_size = model.fmap.get_activation_for_layer(layer_name).shape
        if model.fmap.is_conv_layer(layer_name):
            bsize, channels, w, h = output_size
            out_sizes[layer_name] = (w, h)
        else:
            bsize, linear_out = output_size
            out_sizes[layer_name] = linear_out
    model.fmap.clear_activations()
    # min_size currently unused
    for layer_name in model.fmap.layer_names:
        model.fmap.set_max_input_size(layer_name, out_sizes[layer_name])


class FeatureMap(ABC):
    def __init__(self):
        self.layer_names = ['your', 'layers', 'here']
        self.samples = {
            'your': 16,
            'layers': 16,
            'here': 16
        }
        self.final_proj_samples = 16
        self.modules = OrderedDict()
        self.act = OrderedDict()
        self.max_input_sizes = OrderedDict()
        self.activations_hooks_registered = False

    def get_samples(self, module_name: str):
        return self.samples[module_name]

    def set_module(self, module_name: str, module: nn.Module):
        assert module_name in self.layer_names, f"Module {module_name} not in layers, allowed: {list(self.layer_names)}"
        self.modules[module_name] = module

    def get_module(self, module_name: str):
        return self.modules[module_name]

    def set_activation_for_layer(self, module_name: str, output: Tensor):
        assert module_name in self.layer_names, f"Module {module_name} not in layers, allowed: {list(self.layer_names)}"
        self.act[module_name] = output

    def get_activation_for_layer(self, module_name: str):
        return self.act[module_name]

    def clear_activations(self):
        self.act = OrderedDict()

    def get_all_activations(self) -> [str, Tensor]:
        return list(self.act.items())

    def set_max_input_size(self, layer_name, size: Tuple[int, int]):
        self.max_input_sizes[layer_name] = size

    def get_max_input_size(self, layer_name):
        return self.max_input_sizes[layer_name]

    def is_conv_layer(self, module_name: str):
        return hasattr(self.get_module(module_name), 'kernel_size')

    def get_kernel_size(self, module_name: str):
        return self.get_module(module_name).kernel_size

    def get_in_channel(self, module_name: str):
        module = self.get_module(module_name)
        # Could add assert here
        return module.in_channels


def get_representation_matrix(net, example_data) -> Dict[str, Tensor]:
    # Ignoring output, we only care about activations
    clock_start = time.perf_counter()
    _ = net(example_data)
    clock_end_inf = time.perf_counter()
    print(f"Representation inference time: {clock_end_inf - clock_start}")
    mats = dict()
    for layer_name in tqdm(net.fmap.layer_names):
        bsz = net.fmap.samples.get(layer_name)
        k = 0
        if net.fmap.is_conv_layer(layer_name):
            kernel_size = net.fmap.get_kernel_size(layer_name)
            in_channel = net.fmap.get_in_channel(layer_name)
            conv_width, conv_height = compute_conv_output_size(net.fmap.get_max_input_size(layer_name),
                                                               net.fmap.get_kernel_size(layer_name))
            mat = np.zeros((kernel_size[0] * kernel_size[1] * in_channel, conv_width * conv_height * bsz))
            act = net.fmap.get_activation_for_layer(layer_name).detach().cpu().numpy()
            for kk in range(bsz):
                for w_index in range(conv_width):
                    for h_index in range(conv_height):
                        patch = act[kk, :, w_index:kernel_size[0] + w_index, h_index:kernel_size[1] + h_index]
                        if patch.shape[0] != in_channel or patch.shape[1] != kernel_size[1] or patch.shape[2] != \
                                kernel_size[0]:
                            padded = np.zeros((in_channel, kernel_size[1], kernel_size[0]))
                            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                            mat[:, k] = padded.reshape(-1)
                        else:
                            mat[:, k] = patch.reshape(-1)
                        k += 1
            mats[layer_name] = mat
        else:
            act = net.fmap.get_activation_for_layer(layer_name).detach().cpu().numpy()
            activation = act[0:bsz].transpose(1, 0)
            mats[layer_name] = activation
    clock_end_comp = time.perf_counter()
    print(f"Representation matrix computation time: {clock_end_comp - clock_end_inf}")
    print('-' * 30)
    print('Representation Matrix')
    print('-' * 30)
    for layer_name in mats.keys():
        print('Layer {} : {}'.format(layer_name, mats[layer_name].shape))
    print('-' * 30)
    return mats


def update_GPM(model, mat_dict, threshold, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
    print('Threshold: ', threshold)
    if not features:
        # After First Task
        for layer_name in mat_dict.keys():
            activation = mat_dict[layer_name]
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold[layer_name])  # +1
            features[layer_name] = U[:, 0:r]
    else:
        for layer_name in mat_dict.keys():
            activation = mat_dict[layer_name]
            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()
            # Projected Representation (Eq-8)
            act_hat = activation - np.dot(np.dot(features[layer_name], features[layer_name].transpose(1, 0)),
                                          activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold[layer_name]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print('Skip Updating GPM for layer: {}'.format(layer_name))
                continue
            # update GPM
            Ui = np.hstack((features[layer_name], U[:, 0:r]))
            if Ui.shape[1] > Ui.shape[0]:
                features[layer_name] = Ui[:, 0:Ui.shape[0]]
            else:
                features[layer_name] = Ui

    print('-' * 40)
    print('Gradient Constraints Summary')
    print('-' * 40)
    for layer_name in features:
        print('Layer {} : {}/{}'.format(layer_name, features[layer_name].shape[1], features[layer_name].shape[0]))
    print('-' * 40)
    return features
