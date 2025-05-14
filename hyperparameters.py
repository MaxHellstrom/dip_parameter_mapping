def get_hyperparameters(
    network_updates=None, training_updates=None, lims_version="t1_synthetic"
):
    # network parameters
    PARAMS_NW = dict(
        # general parameters
        GENERAL=dict(
            need_output_act=False,
            clamp_p1=False,  # numerical stability
            clamp_p2=True,  # numerical stability
            train_p1_as_neg_log=False,
            train_p2_as_neg_log=True,  # train p2 (e.g. T1, as -log(p2) for stability)
            clamp_lims=((0, 10), (-20, +20)),
        ),
        # MCDIP  parameters
        MCDIP=dict(
            dropout_p=1 * 0.0200,
            disable_mcdropout=False,  # disables mc dropout when True
            num_iter_mc_dropout=128,  # num mc samples in prediction
            num_scales=5,  # 5  U-net depth
            input_depth=32,  # input depth
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            pad="reflection",
            upsample_mode="bilinear",
            act_fun="ReLU",
            downsample_mode="stride",
            dropout_mode="2d",
        ),
    )

    lims = dict(
        t1_invivo=dict(
            p=((0, 10), (0, 10)),
            bias=((-0.1 * 10, +0.1 * 10), (-0.1 * 10, +0.1 * 10)),
            error=((0, 0.1 * 10), (0, 0.1 * 10)),
        ),
        t1_synthetic=dict(
            p=((0, 15), (0, 6)),
            bias=((-0.1 * 15, +0.1 * 15), (-0.1 * 6, +0.1 * 6)),
            error=((0, 0.1 * 15), (0, 0.1 * 6)),
        ),
        t2_synthetic=dict(
            p=((0, 2), (0, 0.4)),
            bias=((-0.1 * 2, +0.1 * 2), (-0.1 * 0.5, +0.1 * 0.5)),
            error=((0, 0.1 * 2), (0, 0.1 * 0.5)),
        ),
        adc_invivo=dict(
            p=((0, 0.5), (0, 3.5)),
            bias=((-0.1 * 1, +0.1 * 1), (-0.1 * 6, +0.1 * 6)),
            error=((0, 0.1 * 1), (0, 0.1 * 5)),
        ),
    )

    # training parameters
    PARAMS_TRAINING = dict(
        path_results=None,
        max_num_iter_optim=100000,  # num optimization iterations
        sampling_interval=100000,
        max_num_iter_optim_new_slice=None,  # uses same as max_num_iter_optim if None
        lr=0.0003,  # learning rate
        reg_noise_std=1 * 0.05,  # noise regularization
        weight_decay=1e-4,
        save_network_input_and_state=True,  # save state dict after optim completed
        path_network_pretrained=None,
        plot_lims_p=lims[lims_version]["p"],
        plot_lims_p_bias=lims[lims_version]["bias"],
        plot_lims_p_error=lims[lims_version]["error"],
        plot_dpi=500,
        # STOP AT MSE LEVEL
        enable_stop_optim_at_mse=True,
        # data["estimated_noise_std"]^2 is used if stop_optim_at_mse=None
        stop_optim_at_mse=None,
        stop_optim_at_mse_window=1,
        mse_scaling=1.0,
        # moving window size for mse threshold
        # STOP AT SLOPE
        enable_stop_optim_at_slope=False,
        stop_optim_at_slope_burnin=0,
        stop_optim_at_slope=1.0e-8,
        stop_optim_at_slope_window=100,
        randomize_signal_noise=False,
        # if true, exports all forward passes during to result results/stored_outputs
        export_output_during_training=False,
        sample_at_iter_zero=False,
    )
    if network_updates is not None:
        for key in PARAMS_NW:
            if key in network_updates:
                PARAMS_NW[key].update(network_updates[key])

    if training_updates is not None:
        PARAMS_TRAINING.update(training_updates)

    return PARAMS_NW, PARAMS_TRAINING
