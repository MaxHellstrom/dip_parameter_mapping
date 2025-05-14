import numpy as np
import torch


def spgr_signal(S0, T1, FA, TR, mask=None, B1_corr=None, mode="numpy", device=None):
    # for a single voxel
    if mode == "voxel":
        exp_term = np.exp(-TR / T1)
        s = (
            S0
            * np.sin(B1_corr * FA)
            * (1 - exp_term)
            / (1 - np.cos(B1_corr * FA) * exp_term)
        )

        return s

    # for torch tensors
    if mode == "torch":
        # create result tensor with shape (nFA, nx, ny)
        s = torch.zeros(len(FA), mask.size()[0], mask.size()[1]).to(device)

        # apply mask and append trailing singleton dim for broadcasting
        # resulting shape: (mask.sum(), 1)
        S0 = S0[mask][:, None]
        T1 = T1[mask][:, None]
        B1_corr = B1_corr[mask][:, None]

        # shape (mask.sum(), 1)
        exp_term = torch.exp(-TR / T1)

        # calc spgr and broadcast over all FA. Resulting shape: (mask.sum(), nFA)
        s_masked = (
            S0
            * torch.sin(FA * B1_corr)
            * (1 - exp_term)
            / (1 - torch.cos(FA * B1_corr) * exp_term)
        )

        # expand the mask to (nFA, nx, ny), select those pixels, and apply s
        # the transpose is required for column major order flattening ("F style").
        s[torch.stack(len(FA) * [mask])] = s_masked.transpose(1, 0).flatten()

        # check for infs and nans
        if s.isnan().sum().item() > 0:
            print("nan found in s")

        if s.isinf().sum().item() > 0:
            print.critical("infs found in s")

        return s

    # for numpy arrays
    if mode == "numpy":
        # create result ndarray with shape (nFA, nx, ny)
        s = np.zeros((len(FA), mask.shape[0], mask.shape[1]))

        # apply mask and append trailing singleton dim for broadcasting
        # resulting shape: (mask.sum(), 1)
        S0 = np.expand_dims(S0[mask], axis=-1)
        T1 = np.expand_dims(T1[mask], axis=-1)
        B1_corr = np.expand_dims(B1_corr[mask], axis=-1)

        # shape (mask.sum(), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_term = np.exp(-TR / T1)

        # calc spgr and broadcast over all FA. Resulting shape: (mask.sum(), nFA)
        s_masked = (
            S0
            * np.sin(B1_corr * FA)
            * (1 - exp_term)
            / (1 - np.cos(B1_corr * FA) * exp_term)
        )

        # expand the mask to (nFA, nx, ny), select those pixels, and apply s
        s[np.stack(len(FA) * [mask])] = s_masked.flatten(order="F")

        return s


def spin_echo_signal(S0, T2, TE, mask=None, mode="numpy", device=None):
    # for a single voxel
    if mode == "voxel":
        s = S0 * np.exp(-TE / T2)

        return s

    # for torch tensors
    if mode == "torch":
        # create result tensor with shape (nFA, nx, ny)
        s = torch.zeros(len(TE), mask.size()[0], mask.size()[1]).to(device)

        # apply mask and append trailing singleton dim for broadcasting
        # resulting shape: (mask.sum(), 1)
        S0 = S0[mask][:, None]
        T2 = T2[mask][:, None]

        # calc SE and broadcast over all TE. Resulting shape: (mask.sum(), nTE)
        s_masked = S0 * torch.exp(-TE / T2)

        # expand the mask to (nFA, nx, ny), select those pixels, and apply s
        # the transpose is required for column major order flattening ("F style").
        s[torch.stack(len(TE) * [mask])] = s_masked.transpose(1, 0).flatten()

        return s

    # for numpy arrays
    if mode == "numpy":
        # create result ndarray with shape (nFA, nx, ny)
        s = np.zeros((len(TE), mask.shape[0], mask.shape[1]))

        # apply mask and append trailing singleton dim for broadcasting
        # resulting shape: (mask.sum(), 1)
        S0 = np.expand_dims(S0[mask], axis=-1)
        T2 = np.expand_dims(T2[mask], axis=-1)

        # calc SE and broadcast over all TE. Resulting shape: (mask.sum(), nTE)
        with np.errstate(divide="ignore", invalid="ignore"):
            s_masked = S0 * np.exp(-TE / T2)

        # expand the mask to (nFA, nx, ny), select those pixels, and apply s
        s[np.stack(len(TE) * [mask])] = s_masked.flatten(order="F")

        return s


def diffusion_signal(S0, ADC, b_values, mask=None, mode="numpy", device=None):
    # for a single voxel
    if mode == "voxel":
        s = S0 * np.exp(-b_values * ADC)

        return s

    # for torch tensors
    if mode == "torch":
        s = torch.zeros((len(b_values), mask.shape[0], mask.shape[1])).to(device)

        for i in range(len(b_values)):
            s[i, :, :][mask] = S0[mask] * torch.exp(-b_values[i] * ADC[mask])

        return s

    # for numpy arrays
    if mode == "numpy":
        if mask is None:
            mask = np.ones(shape=S0.shape, dtype=bool)

        s = np.zeros((len(b_values), mask.shape[0], mask.shape[1]))

        for i in range(len(b_values)):
            s[i, :, :][mask] = S0[mask] * np.exp(-b_values[i] * ADC[mask])

        return s


def simulate_complex_noise(image, noise_std):
    """Adds complex noise to 2D image

    Parameters
    ----------
    img :
        2D magnitude image
    noise_std :
        standard deviation of noise
    """

    if len(image.shape) == 2:
        nx = image.shape[0]
        ny = image.shape[1]

        r1 = np.random.randn(nx, ny)
        r2 = np.random.randn(nx, ny)

        res = np.abs(image + noise_std * r1 + 1j * noise_std * r2)

        return res

    if len(image.shape) == 3:
        nc = image.shape[0]
        nx = image.shape[1]
        ny = image.shape[2]

        r1 = np.random.randn(nc, nx, ny)
        r2 = np.random.randn(nc, nx, ny)

        res = np.zeros((nc, nx, ny))

        for channel in range(nc):
            tmp = image[channel, :, :]
            res[channel, :, :] = np.abs(
                tmp + noise_std * r1[channel, :, :] + 1j * noise_std * r2[channel, :, :]
            )

        return res
