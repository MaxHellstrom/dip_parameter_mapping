import numpy as np


def check_stopping_criteria(metrics, idx_iter, param):
    # stop by mse criterion

    if param["training"]["enable_stop_optim_at_mse"]:
        bool_stop = stop_by_moving_average(
            vec=metrics["loss"],
            windowlen=param["training"]["stop_optim_at_mse_window"],
            threshold=param["training"]["stop_optim_at_mse"],
            idx_iter=idx_iter,
        )

        if bool_stop:
            print(f"mse limit reached at iter {idx_iter}, terminating optim")
            return True

    if param["training"]["enable_stop_optim_at_slope"]:
        bool_stop = stop_by_slope(
            vec=metrics["hp_fraction"],
            windowlen=param["training"]["stop_optim_at_slope_window"],
            threshold=param["training"]["stop_optim_at_slope"],
            idx_iter=idx_iter,
            iter_burnin=param["training"]["stop_optim_at_slope_burnin"],
        )

        if bool_stop:
            print(f"hp slope limit reached at iter {idx_iter}, terminating optim")
            return True

    if stop_by_max_num_iter(
        idx_iter=idx_iter, iter_max=param["training"]["max_num_iter_optim"]
    ):
        print(f"max_num_iter reached at iter {idx_iter}, terminating optim")
        return True

    return False


def stop_by_moving_average(vec, windowlen, threshold, idx_iter):
    if idx_iter > windowlen and np.average(vec[-windowlen:]) < threshold:
        return True
    return False


def stop_by_max_num_iter(idx_iter, iter_max):
    return idx_iter == iter_max - 1


def stop_by_slope(vec, windowlen, threshold, idx_iter, iter_burnin):
    if idx_iter < iter_burnin:
        return False
    if len(vec) == 1:
        return False

    if len(vec) < windowlen:
        return False

    x = np.arange(len(vec))[-windowlen:]
    y = vec[-windowlen:]

    k, _ = np.polyfit(x=x, y=y, deg=1)
    if k > threshold:
        return True

    return False
