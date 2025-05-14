import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares


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
