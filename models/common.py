from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .downsampler import Downsampler


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def act(act_fun="LeakyReLU"):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "ELU":
            return nn.ELU()
        elif act_fun == "none":
            return nn.Sequential()
        elif act_fun == "ReLU":
            return nn.ReLU()
        else:
            raise AssertionError()
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(
    in_f,
    out_f,
    kernel_size,
    stride=1,
    disable_dropout=False,
    bias=True,
    pad="zero",
    downsample_mode="stride",
    dropout_mode=None,
    dropout_p=0.2,
    iterator=1,
    string="deeper",
):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            downsampler = Downsampler(
                n_planes=out_f,
                factor=stride,
                kernel_type=downsample_mode,
                phase=0.5,
                preserve_size=True,
            )
        else:
            raise AssertionError()

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    dropout = None
    if dropout_mode == "2d" and not disable_dropout:
        dropout = nn.Dropout2d(p=dropout_p)
    elif dropout_mode == "1d" and not disable_dropout:
        dropout = nn.Dropout(p=dropout_p)

    layers = filter(lambda x: x is not None, [padder, convolver, dropout, downsampler])

    ordered_layers = OrderedDict(
        [(f"{layer._get_name()}_{string}_{iterator}", layer) for layer in layers]
    )

    return nn.Sequential(ordered_layers)
