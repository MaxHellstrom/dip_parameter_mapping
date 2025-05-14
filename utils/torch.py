from collections import OrderedDict

import numpy as np
import torch


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var, drop_c0=True):
    """Converts an image in torch.Tensor format to np.array."""
    if drop_c0:
        return img_var.detach().cpu().numpy()[0]

    return img_var.detach().cpu().numpy()


def rename_modules(sequential, string, number):
    _modules = OrderedDict()
    for key in sequential._modules:
        module = sequential._modules[key]
        if len(key) == 1:
            _module_name = f"{module._get_name()}_{string}_{number}"
            # now it's only equipped for two same modules
            if _module_name not in _modules:
                _modules[_module_name] = module
            else:
                _modules[_module_name + "_1"] = module
        else:
            _modules[key] = module
    sequential._modules = _modules
    return number + 1


def get_noise(input_depth, method, spatial_size, noise_type="u", var=1.0 / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x
    `spatial_size[1]`) initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard
        deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1),
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        raise AssertionError()

    return net_input


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        raise AssertionError()
