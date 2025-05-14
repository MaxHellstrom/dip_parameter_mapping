from models import get_architecture
from utils.data import (
    export_param_dict_to_results_folder,
    load_data_to_param_dict,
    load_dataset,
)
from utils.training import (
    Timer,
    calculate_qmri_loss,
    check_progress,
    get_net_input,
    initialize_metrics_dict,
    initialize_optimizer,
    perform_noise_regularization,
    set_action_interval,
)


def denoise_qmri_parameter_mapping(p: dict) -> None:
    p = load_data_to_param_dict(
        data=load_dataset(path=p["path"]["data"], idx_slice=p["data"]["idx_slice"]),
        param=p,
    )

    timer = Timer(export_path=p["path"]["results"])
    actions = set_action_interval(p)

    export_param_dict_to_results_folder(p)

    net = get_architecture(p)
    metrics = initialize_metrics_dict(p)

    net_input, net_input_saved, noise = get_net_input(p)
    optimizer = initialize_optimizer(net=net, net_input=net_input, param=p)

    timer.start()
    for idx_iter in range(actions.__len__()):
        optimizer.zero_grad()

        net_input = perform_noise_regularization(p, net_input_saved, noise)

        nw_output = net(net_input)

        loss, training_metrics = calculate_qmri_loss(nw_output, p, idx_iter)
        loss.backward()

        for metric in training_metrics:
            metrics[metric].append(training_metrics[metric].item())

        stop_reached = check_progress(
            actions=actions,
            idx_iter=idx_iter,
            metrics=metrics,
            net_input_saved=net_input_saved,
            noise=noise,
            param=p,
            net=net,
        )
        if stop_reached:
            break
        optimizer.step()

    timer.stop(export=True)
