import matplotlib.pyplot as plt
import numpy as np

from .gridplot import GridPlot


def plot_estimate_mask(mask, path_export=None, width=8, height=8):
    plot = GridPlot()
    plot.add_subplot(
        mask, cbar_delta=0.1, title="mask", cmap=plot.cmaps.binary, cbar_binary=True
    )
    # plot.ticks.remove_all()
    plot.set_size(x=width, y=height)

    if path_export is not None:
        plot.export(path_export, bbox_inches="tight", dpi=300)
        plt.close(plot.fig)


def plot_estimate_b1_corr(
    b1_corr, mask, lims=(None, None), path_export=None, width=8, height=3
):
    plot = GridPlot(ncols=2)
    plot.add_subplot(b1_corr, lims=lims, cbar_delta=0.1)
    plot.ticks.remove_cols(idx=0)
    plot.set_size(x=width, y=height)
    plot.set_spacing(wspace=0.5)
    plot.axs[0, 1].hist(b1_corr[mask], bins=50)
    plot.text.titles_to_row(idx=0, vals=("b1 corr", "b1 corr in mask"))
    if path_export is not None:
        plot.export(path_export, bbox_inches="tight", dpi=300)
        plt.close(plot.fig)


def plot_estimate_param(
    p_est,
    p_ref,
    mask,
    lims_p,
    lims_diff,
    name_est="None",
    name_ref="None",
    path_export=None,
    lims_hist=((None, None), (None, None)),
    param_units=("None", "None"),
    param_names=("None", "None"),
    width=12,
    height=8,
    wspace=0.2,
    dpi=300,
    cbar_delta=0.1,
    sf=0.2,
    nbins=50,
    cmap=None,
):
    nrows = p_est.shape[0]
    ncols = 6
    plot = GridPlot(ncols=ncols, nrows=nrows, width_ratios=(1, 1, sf, 1, sf, 1))

    for idx_p in range(p_est.shape[0]):
        ref = p_ref[idx_p, ...]
        est = p_est[idx_p, ...]

        ref[~mask] = 0
        est[~mask] = 0

        plot.add_subplot(
            row=idx_p,
            col=0,
            mat=ref,
            lims=lims_p[idx_p],
            cbar_ticks=lims_p[idx_p],
            cbar_bool=False,
            cmap=cmap,
        )
        plot.axs[idx_p, 5].hist(
            ref[mask],
            bins=nbins,
            density=True,
            label=name_ref,
            alpha=0.5,
        )

        plot.add_subplot(
            row=idx_p,
            col=1,
            mat=est,
            lims=lims_p[idx_p],
            cbar_ticks=lims_p[idx_p],
            cbar_delta=cbar_delta,
            cmap=cmap,
        )
        plot.axs[idx_p, 5].hist(
            est[mask], bins=nbins, density=True, label=name_est, alpha=0.5
        )

        diff = est - ref

        plot.add_subplot(
            row=idx_p,
            col=3,
            mat=diff,
            lims=lims_diff[idx_p],
            cbar_ticks=lims_diff[idx_p],
            cmap=plot.cmaps.bias,
            cbar_delta=cbar_delta,
        )

    plot.text.titles_to_row(
        idx=0, vals=(name_ref, name_est, None, "difference", None, "voxel vals")
    )
    plot.text.ylabels_to_col(
        idx=0,
        vals=[f"{param_names[i]} [{param_units[i]}]" for i in range(p_est.shape[0])],
    )
    for i in range(p_est.shape[0]):
        plot.axs[i, 5].legend()
        plot.axs[i, 5].set(yticks=())
        if lims_hist != ((None, None), (None, None)):
            plot.axs[i, 5].set(xlim=lims_hist[i])
    plot.set_size(x=width, y=height)
    plot.set_spacing(wspace=wspace)
    plot.ticks.remove_cols(idx=(0, 1, 2, 3, 4))
    plot.lines.remove_cols(cols=(0, 1, 2, 3, 4))
    if path_export is not None:
        plot.export(path_export, bbox_inches="tight", dpi=dpi)
        plt.close(plot.fig)


def plot_training_metrics(
    metrics,
    enable_stop_optim_at_mse,
    stop_optim_at_mse,
    path_export,
    dpi,
):
    num_metrics = 0
    titles = []

    for item in metrics:
        if metrics[item] != []:
            num_metrics += 1
            titles.append(item)
    plot = GridPlot(ncols=num_metrics)
    plot.set_size(x=4 * len(titles), y=4)
    plot.text.titles_to_row(idx=0, vals=titles)

    for i in range(len(titles)):
        endval = metrics[titles[i]][-1]

        endval = np.round(metrics[titles[i]][-1], 6)
        label = f"{endval}"
        plot.axs[0, i].plot(metrics[titles[i]], ".", markersize=1, label=label)

        if titles[i] in ["loss"]:
            plot.axs[0, i].set(yscale="log")

    if enable_stop_optim_at_mse:
        plot.axs[0, titles.index("loss")].plot(
            [0, metrics["loss"].__len__()],
            2 * [stop_optim_at_mse],
            "--r",
            label=f"MSE limit: {np.round(stop_optim_at_mse, 6)}",
        )

    for idx_col in range(plot.ncols):
        plot.axs[0, idx_col].legend()

    plot.export(path_export, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_estimate_signal(
    y,
    mask,
    y_ref=None,
    name_est=None,
    name_ref=None,
    path_export=None,
    lims=(0, 1),
    lims_diff=(-0.05, +0.05),
    dpi=100,
    width=12,
):
    ns_y = y.shape[0]

    ncols = ns_y
    nrows = 3 if y_ref is not None else 1
    row_est = 0 if y_ref is None else 1
    plot = GridPlot(nrows=nrows, ncols=ncols)
    plot.ticks.remove_all()

    plot.set_axs_shapes(y[0, :, :].shape)

    if y_ref is not None:
        ns_y_ref = y_ref.shape[0]
        for idx_s in range(ns_y_ref):
            mat = y_ref[idx_s, :, :]
            mat[~mask] = 0

            plot.add_subplot(
                row=0,
                col=idx_s,
                mat=mat,
                lims=lims,
                cbar_ticks=lims,
                cbar_length=0.8,
                cbar_width=0.05,
                cbar_delta=0.05,
                cbar_bool=idx_s == ns_y_ref - 1,
            )

    for idx_s in range(ns_y):
        mat = y[idx_s, :, :]
        mat[~mask] = 0

        plot.add_subplot(
            row=row_est,
            col=idx_s,
            mat=mat,
            lims=lims,
            cbar_ticks=lims,
            cbar_length=0.8,
            cbar_width=0.05,
            cbar_delta=0.05,
            cbar_bool=idx_s == ns_y - 1,
        )

    if y_ref is not None:
        for idx_col in range(plot.ncols):
            mat = y[idx_col, :, :] - y_ref[idx_col, :, :]
            mat[~mask] = 0
            plot.add_subplot(
                row=2,
                col=idx_col,
                mat=mat,
                lims=lims_diff,
                cmap=plot.cmaps.bias,
                cbar_ticks=lims_diff,
                cbar_length=0.8,
                cbar_width=0.05,
                cbar_delta=0.05,
                cbar_bool=idx_col == min(ns_y - 1, ns_y_ref - 1),
            )

    plot.text.ylabel_to_ax(row=row_est, col=0, val=name_est)
    plot.text.ylabel_to_ax(row=0, col=0, val=name_ref)
    if y_ref is not None:
        plot.text.ylabel_to_ax(row=2, col=0, val="row diff")
    plot.ticks.remove_all()
    plot.set_tight_grid(width=width)
    for idx_row in range(plot.nrows):
        ticks = lims_diff if idx_row == 2 else lims

    if path_export is not None:
        plot.export(path_export, bbox_inches="tight", dpi=dpi)
        plt.close(plot.fig)
