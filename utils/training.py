import time

import numpy as np
import torch

from hyperparameters import get_hyperparameters
from utils.physics import diffusion_signal, spgr_signal, spin_echo_signal
from utils.plotting import plot_estimate_param, plot_training_metrics
from utils.stopcriteria import check_stopping_criteria
from utils.torch import get_noise, torch_to_np


def calculate_qmri_loss(
    nw_output: torch.Tensor, param: dict, idx_iter: int
) -> torch.Tensor:
    """implement dip denoising for qmri parameter mapping by mapping nw output through
    the signal equation."""

    if param["training"]["export_output_during_training"]:
        np.save(
            param["path"]["stored_outputs"] / f"iter_{idx_iter}.npy",
            nw_output.detach().cpu().numpy(),
        )
    mask = param["data"]["mask"]

    p1 = nw_output[0, 0, ...]
    p2 = nw_output[0, 1, ...]

    p1 = (
        torch.clamp(
            p1,
            min=param["network"]["clamp_lims"][0][0],
            max=param["network"]["clamp_lims"][0][1],
        )
        if param["network"]["clamp_p1"]
        else p1
    )

    p2 = (
        torch.clamp(
            p2,
            min=param["network"]["clamp_lims"][1][0],
            max=param["network"]["clamp_lims"][1][1],
        )
        if param["network"]["clamp_p2"]
        else p2
    )

    p1 = torch.exp(-p1) if param["network"]["train_p1_as_neg_log"] else p1
    p2 = torch.exp(-p2) if param["network"]["train_p2_as_neg_log"] else p2

    # calculate mri signal from network output
    # T1 Variable Flip Angle (VFA)
    if param["general"]["name_application"] == "T1_VFA":
        s = spgr_signal(
            S0=p1,
            T1=p2,
            FA=param["physics"]["fa"],
            TR=param["physics"]["tr"],
            mask=mask,
            B1_corr=param["physics"]["b1_corr"],
            mode="torch",
            device=param["general"]["device"],
        )
    # T2 Multi Echo Spin Echo
    elif param["general"]["name_application"] == "T2_MESE":
        s = spin_echo_signal(
            S0=p1,
            T2=p2,
            TE=param["physics"]["te"],
            mask=mask,
            mode="torch",
            device=param["general"]["device"],
        )
    # Apparent Diffusion Coefficient Diffusion Weighted Imaging
    elif param["general"]["name_application"] == "ADC_DWI":
        s = diffusion_signal(
            S0=p1,
            ADC=p2,
            b_values=param["physics"]["b"],
            mask=mask,
            mode="torch",
            device=param["general"]["device"],
        )

    # calculate residuals
    r = param["data"]["y"] - s
    # select only the residuals that are within the mask
    r = r[torch.stack(s.shape[0] * [param["data"]["mask"]], axis=0)]

    training_metrics = dict()

    # loss metric
    loss = torch.mean(torch.pow(r, 2))
    training_metrics["loss"] = loss

    return loss, training_metrics


def generate_order(N, mode="sequential"):
    if mode == "sequential":
        return list(range(N))

    elif mode == "sequential_rev":
        return list(reversed(range(N)))

    elif mode == "inwards":
        return [x for i in range(N // 2) for x in (i, N - 1 - i)]

    elif mode == "outwards":
        mid_left = (N - 1) // 2
        mid_right = N // 2
        return [x for i in range(N // 2) for x in (mid_left - i, mid_right + i)]

    elif mode == "random":
        arr = np.arange(N)
        np.random.shuffle(arr)
        return list(arr)

    else:
        raise ValueError(f"unknown mode: {mode}")


def get_tr_params(all_parameters_to_update, lims_version) -> dict:
    training_updates = dict()
    if "PARAMS_TRAINING" in all_parameters_to_update:
        training_updates = all_parameters_to_update["PARAMS_TRAINING"]
    _, training_params = get_hyperparameters(
        training_updates=training_updates, lims_version=lims_version
    )
    return training_params


def get_nw_params(name_architecture: str, all_parameters_to_update: dict) -> dict:
    network_updates = dict()
    if "PARAMS_NW" in all_parameters_to_update:
        network_updates = all_parameters_to_update["PARAMS_NW"]

    all_params_nw, _ = get_hyperparameters(network_updates=network_updates)
    selected_architecture_params = all_params_nw[name_architecture]
    general_params = all_params_nw["GENERAL"]
    general_params.update(selected_architecture_params)
    return general_params


class Timer:
    """single timer class to time the estimation process"""

    def __init__(self, export_path="."):
        self.export_path = export_path

    def start(self):
        self.start = time.time()

    def stop(self, export=True):
        self.stop = time.time()

        self.elapsed_time = self.stop - self.start

        if export:
            np.save(self.export_path / "elapsed_time_s.npy", self.elapsed_time)

    def print_elapsed_time(self):
        print(f"elapsed time: {np.round(self.elapsed_time, 2)} s")


def check_progress(actions, idx_iter, metrics, net_input_saved, noise, param, net):
    # check if any of the stopping criterias are reached
    optim_completed = check_stopping_criteria(metrics, idx_iter, param)
    # if optimization completed, set action to true to continue with estimation
    actions[idx_iter] = True if optim_completed else actions[idx_iter]

    save_output = (
        param["training"]["save_network_input_and_state"] if optim_completed else False
    )

    # see set_action_interval in utils.training
    if not actions[idx_iter]:
        return

    str_progress = get_progress_string(
        idx_slice=param["data"]["idx_slice"],
        idx_iter=idx_iter,
        loss=metrics["loss"][-1],
    )
    print(str_progress)

    # dict to store results
    results = dict(p_est=None, p_std=None)

    # create list to store samples
    p_samples = []

    # monte carlo dropout
    with torch.no_grad():
        net_input = perform_noise_regularization(param, net_input_saved, noise)

        # for each MC sample
        num_iter = (
            param["network"]["num_iter_mc_dropout"]
            if not param["network"]["disable_mcdropout"]
            else 1
        )

        for _ in range(num_iter):
            # forward pass
            out = net(net_input)
            # transform -log(p2) to p2
            if param["network"]["train_p2_as_neg_log"]:
                out[0, 1, :, :] = torch.exp(-out[0, 1, :, :])
            p_samples.append(torch_to_np(out))

    p_samples = np.asarray(p_samples)

    # store results
    results["p_est"] = np.mean(p_samples, axis=0)
    results["p_std"] = np.std(p_samples, axis=0)
    # results["p_samples"] = p_samples

    # plot estimation progress
    if not param["general"]["no_plot_flag"]:
        plot_estimate_param(
            p_est=results["p_est"],
            p_ref=param["data"]["p_nlls"],
            mask=param["data"]["mask"].detach().cpu().numpy(),
            lims_p=param["training"]["plot_lims_p"],
            lims_diff=param["training"]["plot_lims_p_bias"],
            name_est="OURS",
            name_ref="p_nlls",
            path_export=param["path"]["progress_plots"] / f"est_vs_nlls_{idx_iter}.png",
            param_names=param["data"]["param_names"],
            param_units=param["data"]["param_units"],
            dpi=param["training"]["plot_dpi"],
        )

        plot_training_metrics(
            metrics=metrics,
            enable_stop_optim_at_mse=param["training"]["enable_stop_optim_at_mse"],
            stop_optim_at_mse=param["training"]["stop_optim_at_mse"],
            path_export=param["path"]["progress_plots"] / f"metrics_{idx_iter}.png",
            dpi=param["training"]["plot_dpi"],
        )

    # save estimation results, loss list, network state, and network input
    save_results(
        idx_iter=idx_iter,
        param=param,
        results=results,
        metrics=metrics,
        net=net,
        net_input=net_input,
        stopping_criterion_reached=save_output,
    )
    return save_output


def get_progress_string(idx_slice: int, idx_iter: int, loss: float) -> str:
    str_slice = f"slice: {idx_slice:02d} "
    str_iter = f"iter: {idx_iter:05d}, "
    str_loss = f"mse loss: {loss:.5f}"

    return str_slice + str_iter + str_loss


def perform_noise_regularization(
    param: dict, net_input_saved: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """performs noise regularization with std reg_noise_std from param["training"]"""

    if param["training"]["reg_noise_std"] <= 0:
        return net_input_saved

    net_input = net_input_saved + (noise.normal_() * param["training"]["reg_noise_std"])
    return net_input


def save_results(
    idx_iter, param, results, metrics, net, net_input, stopping_criterion_reached
):
    if not stopping_criterion_reached:
        return

    fp = param["path"]["results"]
    print(f"saving results at iter {idx_iter}")

    np.save(file=fp / "p_est.npy", arr=results["p_est"])
    np.save(file=fp / "p_std.npy", arr=results["p_std"])
    # np.save(file=fp / "p_samples.npy", arr=results["p_samples"])

    for metric in metrics:
        np.save(file=fp / f"{metric}.npy", arr=metrics[metric])

    if param["training"]["save_network_input_and_state"]:
        print(f"saving network input and state at iter {idx_iter}")
        torch.save(net_input, fp / "network_input")
        torch.save(net.state_dict(), fp / "network_state")


def get_net_input(param):
    if param["path"]["network_pretrained"]:
        # load the net input of pretrained network
        net_input = torch.load(param["path"]["network_pretrained"] / "network_input")
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

    elif param["general"]["name_architecture"] == "MCDIP":
        net_input = (
            get_noise(
                param["network"]["input_depth"],
                "noise",
                (param["data"]["image_height"], param["data"]["image_width"]),
            )
            .to(param["general"]["device"])
            .detach()
        )

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    return net_input, net_input_saved, noise


def initialize_metrics_dict(param: dict) -> list:
    """creates a dict to store the optimization metrics.
    If a pretrained network is used, load the previous dict instead."""

    if param["path"]["network_pretrained"] is not None:
        fp = param["path"]["network_pretrained"]

        return dict(loss=np.load(fp / "loss.npy").tolist())

    else:
        return dict(loss=[])


def initialize_optimizer(
    net: torch.nn.modules.container.Sequential, net_input: torch.tensor, param: dict
):
    params = get_params(opt_over="net", net=net, net_input=net_input)
    lr = param["training"]["lr"]
    weight_decay = param["training"]["weight_decay"]
    # print(f"returning optimizer with lr={lr}, and weight_decay={weight_decay}")
    return torch.optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    """
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:
        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == "input":
            net_input.requires_grad = True
            params += [net_input]
        else:
            raise AssertionError("")

    return params


def set_action_interval(param: dict) -> list[float]:
    """creates a max_num_iter_optim long list with ones and zeros. This is used by the
    training to know at which iteration the model should perform estimation (ones) and
    when to continue the optimization."""

    n_opt = param["training"]["max_num_iter_optim"]
    n_int = param["training"]["sampling_interval"]

    # set all to zero
    actions = np.zeros(n_opt, dtype=bool)
    # set sampling interval
    actions[::n_int] = True
    # enable last step
    actions[-1] = True  # enable last iteration
    # enable first step if sample_at_iter_zero
    actions[0] = param["training"]["sample_at_iter_zero"]

    return actions
