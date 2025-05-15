import numpy as np
from bm3d import gaussian_kernel
from joblib import Parallel, delayed
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.optimize import least_squares
from scipy.signal import fftconvolve


def masked_ssim(ref, est, mask, L=None, K1=0.01, K2=0.03):
    x = ref[mask]
    y = est[mask]

    if x.size == 0:
        raise ValueError("Masken täcker inga pixlar")

    # Sätt L automatiskt om det inte är specificerat
    if L is None:
        L = max(np.max(x), np.max(y))

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2)

    return numerator / denominator


def masked_nrmse(ref, est, mask):
    x = ref[mask]
    y = est[mask]

    if x.size == 0:
        raise ValueError("Masken täcker inga pixlar")

    mse = np.mean((x - y) ** 2)
    rmse = np.sqrt(mse)

    scale = np.max(x) - np.min(x)
    if scale == 0:
        return 0.0

    return rmse / scale


def estimate_sigma2_from_residuals(r):
    # input: residuals (res) with shape (num_settings * num_voxels)
    num_settings = r.shape[0]

    r2 = np.square(r)
    sigma_2 = 1 / (num_settings - 2) * r2.sum(axis=0)

    return sigma_2


def T1_VFA_NLLS_estimator_parallel(y, FA_values, TR, B1_corr, mask, bounds, n_jobs=-1):
    def initial_guess_from_linear_fit(y_voxel, FA_voxel):
        eps = 1e-6
        sin_fa = np.sin(FA_voxel) + eps
        tan_fa = np.tan(FA_voxel) + eps

        x = y_voxel / tan_fa
        z = y_voxel / sin_fa

        A = np.vstack([x, np.ones_like(x)]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b = coeffs

            if a <= 0 or a >= 1:
                raise ValueError("Value of 'a' is out of valid range (0, 1)")

            T1_init = -TR / np.log(a)
            S0_init = b / (1 - a)
        except ValueError:
            T1_init = 1.0
            S0_init = 1.0
        except Exception:
            T1_init = 1.0
            S0_init = 1.0

        # Clamp till bounds
        S0_init = np.clip(S0_init, bounds[0][0], bounds[1][0])
        T1_init = np.clip(T1_init, bounds[0][1], bounds[1][1])

        return [S0_init, T1_init]

    def voxel_fit(y_voxel, FA_voxel, B1_voxel):
        FA_eff = FA_voxel * B1_voxel

        def model(S0, T1):
            E1 = np.exp(-TR / T1)
            return S0 * np.sin(FA_eff) * (1 - E1) / (1 - np.cos(FA_eff) * E1)

        def residuals(params):
            return y_voxel - model(*params)

        init_guess = initial_guess_from_linear_fit(y_voxel, FA_eff)
        sol = least_squares(residuals, x0=init_guess, bounds=(bounds[0], bounds[1]))
        return sol.x[0], sol.x[1]

    S0 = np.zeros(mask.shape)
    T1 = np.zeros(mask.shape)

    if np.sum(mask) == 0:
        return np.stack((S0, T1), axis=0)

    y_masked = y[:, mask]
    B1_masked = B1_corr[mask]

    results = Parallel(n_jobs=n_jobs)(
        delayed(voxel_fit)(y_masked[:, i], FA_values, B1_masked[i])
        for i in range(y_masked.shape[1])
    )

    S0[mask], T1[mask] = zip(*results, strict=False)

    return np.stack((S0, T1), axis=0)


def get_experiment_noise(noise_type: str, noise_var: float, sz: tuple):
    """
    Generate noise for experiment with specified kernel, variance, seed and size.
    Return noise and relevant parameters.
    The generated noise is non-circular.
    :param noise_type: Noise type, see get_experiment_kernel for list of accepted types.
    :param noise_var: Noise variance of the resulting noise
    :param realization: Seed for the noise realization
    :param sz: image size -> size of resulting noise
    :return: noise, PSD, and kernel
    """
    # np.random.seed(realization)

    # Get pre-specified kernel
    kernel = get_experiment_kernel(noise_type, noise_var, sz)

    # Create noisy image
    half_kernel = np.ceil(np.array(kernel.shape) / 2)

    if len(sz) == 3 and half_kernel.size == 2:
        half_kernel = [half_kernel[0], half_kernel[1], 0]
        kernel = np.atleast_3d(kernel)

    half_kernel = np.array(half_kernel, dtype=int)

    # Crop edges
    noise = fftconvolve(
        np.random.normal(size=(sz + 2 * half_kernel)), kernel, mode="same"
    )
    noise = np.atleast_3d(noise)[
        half_kernel[0] : -half_kernel[0], half_kernel[1] : -half_kernel[1], :
    ]

    psd = abs(fft2(kernel, (sz[0], sz[1]), axes=(0, 1))) ** 2 * sz[0] * sz[1]

    return noise, psd, kernel

def get_experiment_kernel(
    noise_type: str, noise_var: float, sz: tuple = np.array((101, 101))
):
    """
    Get kernel for generating noise from specific experiment from the paper.
    :param noise_type: Noise type string, g[0-4](w|)
    :param noise_var: noise variance
    :param sz: size of image, used only for g4 and g4w
    :return: experiment kernel with the l2-norm equal to variance
    """
    # if noiseType == gw / g0
    kernel = np.array([[1]])
    noise_types = ["gw", "g0", "g1", "g2", "g3", "g4", "g1w", "g2w", "g3w", "g4w"]
    if noise_type not in noise_types:
        raise ValueError("Noise type must be one of " + str(noise_types))

    if noise_type != "g4" and noise_type != "g4w":
        # Crop this size of kernel when generating,
        # unless pink noise, in which
        # if noiseType == we want to use the full image size
        sz = np.array([101, 101])
    else:
        sz = np.array(sz)

    # Sizes for meshgrids
    sz2 = -(1 - (sz % 2)) * 1 + np.floor(sz / 2)
    sz1 = np.floor(sz / 2)
    uu, vv = np.meshgrid(
        [i for i in range(-int(sz1[0]), int(sz2[0]) + 1)],
        [i for i in range(-int(sz1[1]), int(sz2[1]) + 1)],
    )

    beta = 0.8

    if noise_type[0:2] == "g1":
        # Horizontal line
        kernel = np.atleast_2d(16 - abs(np.linspace(1, 31, 31) - 16))

    elif noise_type[0:2] == "g2":
        # Circular repeating pattern
        scale = 1
        dist = uu**2 + vv**2
        kernel = np.cos(np.sqrt(dist) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == "g3":
        # Diagonal line pattern kernel
        scale = 1
        kernel = np.cos((uu + vv) / scale) * gaussian_kernel((sz[0], sz[1]), 10)

    elif noise_type[0:2] == "g4":
        # Pink noise
        dist = uu**2 + vv**2
        n = sz[0] * sz[1]
        spec = np.sqrt((np.sqrt(n) * 1e-2) / (np.sqrt(dist) + np.sqrt(n) * 1e-2))
        kernel = fftshift(ifft2(ifftshift(spec)))

    else:  # gw and g0 are white
        beta = 0

    # -- Noise with additional white component --

    if len(noise_type) > 2 and noise_type[2] == "w":
        kernel = kernel / np.sqrt(np.sum(kernel**2))
        kalpha = np.sqrt((1 - beta) + beta * abs(fft2(kernel, (sz[0], sz[1]))) ** 2)
        kernel = fftshift(ifft2(kalpha))

    kernel = np.real(kernel)
    # Correct variance
    kernel = kernel / np.sqrt(np.sum(kernel**2)) * np.sqrt(noise_var)

    return kernel
