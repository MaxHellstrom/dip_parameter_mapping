import datetime
from pathlib import Path

from denoise_qmri_parameter_mapping import denoise_qmri_parameter_mapping
from utils.data import initialize_parameter_dict
from utils.training import generate_order, get_nw_params, get_tr_params

timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

name_datasets = ["T1_VFA__Pat_04_shape_222_185_48_fa_2_4_11_13_15_noise_0.02"]
path_results = Path(r"E:\paper3_local_storage\TMP")

order_warmstart = generate_order(N=48, mode="sequential")
order_patient = generate_order(N=name_datasets.__len__())
print(f"order_warmstart: {order_warmstart}")
print(f"order_patient: {order_patient}")


lims_version = "t1_synthetic"
name_execution = timestamp
no_plot_flag = False
name_architecture = "MCDIP"
print("\n")
for idx_pat in order_patient:
    print(f"pat: {name_datasets[idx_pat]}")
    new_parameters_json = ""

    # if we are at the first patient, use no previous network
    if idx_pat == order_patient[0]:
        path_state = None
    else:  # if another patient already has been denoised
        path_state = path_results / name_datasets[idx_pat - 1] / name_execution / "000"

    for idx_slice in order_warmstart:
        print("")
        print(f"slice: {idx_slice:03d}")

        # if we are not at the first slice in the dataset
        if idx_slice != order_warmstart[0]:
            idx_prev_estimate_slice = order_warmstart[idx_slice - 1]

            path_state = (
                path_results
                / name_datasets[idx_pat].split("__")[0]
                / name_datasets[idx_pat]
                / name_execution
                / f"{idx_prev_estimate_slice}".zfill(3)
            )
        if path_state is not None:
            print(f"state: {path_state.relative_to(path_results)}")
        else:
            print("state: None")

        updates = dict(
            PARAMS_NW=dict(
                MCDIP=dict(
                    # ad updates
                ),
                GENERAL=dict(
                    # ad updates
                ),
            ),
            PARAMS_TRAINING=dict(
                path_network_pretrained=path_state,
                enable_stop_optim_at_mse=True,
                stop_optim_at_mse=None,
                sampling_interval=5000,
                max_num_iter_optim=100000,
                path_results=path_results,
            ),
        )

        param = initialize_parameter_dict(
            params_training=get_tr_params(updates, lims_version=lims_version),
            params_network=get_nw_params("MCDIP", updates),
            name_architecture=name_architecture,
            name_dataset=name_datasets[idx_pat],
            name_execution=name_execution,
            no_plot_flag=no_plot_flag,
            idx_slice=idx_slice,
        )

        denoise_qmri_parameter_mapping(param)
    print("\n")
