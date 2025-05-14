import copy
import json
from pathlib import Path

import numpy as np
import torch

from utils.physics import simulate_complex_noise


def initialize_parameter_dict(
    name_architecture: str,
    name_dataset: str,
    params_network: dict,
    params_training: dict,
    name_execution: str,
    idx_slice: int,
    no_plot_flag: bool,
) -> dict:
    name_application = name_dataset.split("__")[0]

    parameter_dict = dict(
        general=dict(
            name_application=name_application,
            name_architecture=name_architecture,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            no_plot_flag=no_plot_flag,
        ),
        data=dict(idx_slice=idx_slice),
        path=dict(
            data=Path(__file__).parent.parent
            / "data"
            / "data_parsed"
            / name_application
            / name_dataset,
            results=params_training["path_results"]
            / name_application
            / name_dataset
            / name_execution
            / str(idx_slice).zfill(3),
            network_pretrained=Path(__file__).parent.parent
            / params_training["path_network_pretrained"]
            if params_training["path_network_pretrained"] is not None
            else None,
        ),
        physics=dict(),
        training=params_training,
        network=params_network,
    )
    return parameter_dict


def export_param_dict_to_results_folder(param: dict):
    """export param object to result path. The saved .json file can then be opened with
    the function load_params_json here in utils.data"""

    path = param["path"]["results"]

    dict = copy.deepcopy(param)
    # remove matrices from params
    for mat in ["mask", "y", "p_ref", "p_nlls"]:
        if mat in dict["data"]:
            dict["data"].pop(mat)

    if dict["general"]["name_application"] == "T1_VFA":
        dict["physics"].pop("b1_corr")

    for key in dict:
        for subkey in dict[key]:
            if dict[key][subkey] not in [str, int, float]:
                dict[key][subkey] = str(dict[key][subkey])

    with open(path / "params.json", "w") as f:
        json.dump(dict, f)


def load_data_to_param_dict(param, data):
    """load data to dictionary with all paramaters, i.e. "params"."""

    # create folders to store estimation results
    param = create_result_folders_and_append_to_param(param)

    device = param["general"]["device"]
    param["data"]["name_application"] = str(data["name_application"])

    for _param in ["image_height", "image_width", "num_slices"]:
        param["data"][_param] = int(data[_param])

    for _param in ["param_names", "param_units"]:
        param["data"][_param] = list(data[_param])

    param["data"]["y"] = torch.from_numpy(data["y"]).to(device)

    # if no specific MSE limit is specified (), use the estimated mse from the data
    # parsing procedure,
    if param["training"]["stop_optim_at_mse"] is None:
        mse_scale = param["training"]["mse_scaling"]
        param["training"]["stop_optim_at_mse"] = mse_scale * np.square(
            data["estimated_noise_std"].item()
        )

    mask = data["mask"]
    param["data"]["mask"] = torch.from_numpy(mask).bool().to(device)
    param["data"]["p_ref"] = data["p_ref"]
    param["data"]["p_nlls"] = data["p_nlls"]

    if param["training"]["randomize_signal_noise"]:
        y_ref = data["y_ref"]
        noise_std = data["noise_std"]

        y = simulate_complex_noise(image=y_ref, noise_std=noise_std)

        param["data"]["y"] = torch.from_numpy(y).to(device)
        np.save(file=param["path"]["results"] / "y.npy", arr=y)

    param = _append_application_specific_data(data, param, device)
    return param


def _append_application_specific_data(data, param, device):
    if data["name_application"] == "T1_VFA":
        for parameter_name in ["tr", "fa"]:
            param["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name].astype(np.float32)
            ).to(device)

        b1_corr = data["b1_corr"]

        param["physics"]["b1_corr"] = torch.from_numpy(b1_corr.astype(np.float32)).to(
            device
        )
    elif data["name_application"] == "T2_MESE":
        # echo time
        for parameter_name in ["te"]:
            param["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name].astype(np.float32)
            ).to(device)

    elif data["name_application"] == "ADC_DWI":
        for parameter_name in ["b"]:
            param["physics"][parameter_name] = torch.from_numpy(
                data[parameter_name]
            ).to(device)

    return param


def create_result_folders_and_append_to_param(param: dict) -> dict:
    """creates folders to store results and append those paths to param"""

    Path(param["path"]["results"]).mkdir(parents=True, exist_ok=True)

    param["path"]["progress_plots"] = param["path"]["results"] / "progress_plots"
    Path(param["path"]["progress_plots"]).mkdir()

    if param["training"]["export_output_during_training"]:
        param["path"]["stored_outputs"] = param["path"]["results"] / "stored_outputs"
        param["path"]["stored_outputs"].mkdir(exist_ok=True)

    return param


def load_dataset(path: Path, idx_slice: int) -> np.lib.npyio.NpzFile:
    """load dataset in npz format"""

    return np.load(path / f"dataset_idx_s_{idx_slice:03d}.npz")
