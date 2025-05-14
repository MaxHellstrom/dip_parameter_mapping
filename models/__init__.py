import torch

from .skip import skip


def get_architecture(param):
    param_nw = param["network"]

    if param["general"]["name_architecture"] == "MCDIP":
        net = skip(
            num_input_channels=param_nw["input_depth"],
            num_output_channels=param["data"]["p_ref"].shape[0],
            num_channels_down=[param_nw["skip_n33d"]] * param_nw["num_scales"]
            if isinstance(param_nw["skip_n33d"], int)
            else param_nw["skip_n33d"],
            num_channels_up=[param_nw["skip_n33u"]] * param_nw["num_scales"]
            if isinstance(param_nw["skip_n33u"], int)
            else param_nw["skip_n33u"],
            num_channels_skip=[param_nw["skip_n11"]] * param_nw["num_scales"]
            if isinstance(param_nw["skip_n11"], int)
            else param_nw["skip_n11"],
            need_sigmoid=False,
            need_bias=True,
            pad=param_nw["pad"],
            upsample_mode=param_nw["upsample_mode"],
            downsample_mode=param_nw["downsample_mode"],
            act_fun=param_nw["act_fun"],
            disable_dropout=param_nw["disable_mcdropout"],
            dropout_mode_down=param_nw["dropout_mode"],
            dropout_p_down=param_nw["dropout_p"],
            dropout_mode_up=param_nw["dropout_mode"],
            dropout_p_up=param_nw["dropout_p"],
            need_output_act=param_nw["need_output_act"],
        ).to(param["general"]["device"])

    if param["path"]["network_pretrained"] is not None:
        net.load_state_dict(
            torch.load(param["path"]["network_pretrained"] / "network_state")
        )
    else:
        print("training net from scratch...")

    return net
