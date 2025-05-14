from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class GridPlot:
    def __init__(
        self,
        nrows=1,
        ncols=1,
        size=None,
        lims=(None, None),
        dpi=None,
        gridspec_kw=None,
        width_ratios=None,
        height_ratios=None,
    ):
        self.nrows, self.ncols = nrows, ncols
        self.size, self.lims = size, lims
        self.lims = lims
        self.cmap = plt.cm.gray
        self.latex = False

        self.idx_clip = None
        self.fig, self.axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=size,
            dpi=dpi,
            gridspec_kw=gridspec_kw,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )
        self.axs = np.reshape(self.axs, (self.nrows, self.ncols))
        self.lines = GridPlot.Lines(self)
        self.ticks = GridPlot.Ticks(self)
        self.cmaps = GridPlot.Cmaps()
        self.cbar = GridPlot.Cbar(self)
        self.text = GridPlot.Text(self)
        self.cbar_params = GridPlot.Cbar_params()

    class Cbar_params:
        def __init__(
            self,
            length=1.0,
            width=0.075,
            delta=0,
            borderpad=0,
            loc="center right",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
        ):
            self.length = length
            self.width = width
            self.bbox_to_anchor = (x0 + width + delta, y0, x1, y1)
            self.borderpad = borderpad
            self.loc = loc

    def set_clip_indices(self, x1, x2, y1, y2):
        self.idx_clip = [x1, x2, y1, y2]

    def combine_subplots(self, rows, cols, name="ax_combined"):
        gs = self.axs[rows[0], cols[0]].get_gridspec()

        for row in range(rows[0], rows[1] + 1):
            for col in range(cols[0], cols[1] + 1):
                self.axs[row, col].remove()

        subgs = gs[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1]
        setattr(self, name, self.fig.add_subplot(subgs))

    def set_spacing(
        self, wspace=None, hspace=None, left=None, right=None, top=None, bottom=None
    ):
        plt.subplots_adjust(
            wspace=wspace, hspace=hspace, left=left, right=right, top=top, bottom=bottom
        )

    def set_size(self, x=None, y=None):
        self.fig.set_size_inches(x, y)

    def set_tight_grid(self, width):
        data_shape = self.axs[0, 0].properties()["images"][0].get_array().data.shape

        height = width * (self.nrows / self.ncols) * (data_shape[0] / data_shape[1])

        self.set_spacing(wspace=0, hspace=0)
        self.set_size(x=width, y=height)

    def set_axs_shapes(self, shape=(100, 100)):
        for row, col in product(range(self.nrows), range(self.ncols)):
            _set_ax_shape(ax=self.axs[row, col], shape=shape)

    def export(self, file_name, bbox_inches=None, dpi=None, pad_inches=None):
        self.fig.savefig(
            fname=file_name, bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches
        )

    def add_subplot(
        self,
        mat,
        row=0,
        col=0,
        lims=(None, None),
        cmap=None,
        xlabel="",
        ylabel="",
        title="",
        cbar_bool=True,
        cbar_length=1.0,
        cbar_width=0.075,
        cbar_delta=0,
        cbar_loc="center right",
        cbar_anchors=(0, 0, 1, 1),
        cbar_borderpad=0,
        cbar_orientation="vertical",
        cbar_ticks=None,
        cbar_label="",
        cbar_title="",
        cbar_binary=False,
        binary_ticklabels=("excl.", "incl."),
    ):
        cmap = self.cmap if cmap is None else cmap
        lims = self.lims if lims == [None, None] else lims

        bbox = (
            cbar_anchors[0] + cbar_width + cbar_delta,
            cbar_anchors[1],
            cbar_anchors[2],
            cbar_anchors[3],
        )

        if self.latex:
            xlabel, ylabel = _tex_str(xlabel), _tex_str(ylabel)
            title, cbar_label = _tex_str(title), _tex_str(cbar_label)
            binary_ticklabels = (
                _tex_str(binary_ticklabels[0]),
                _tex_str(binary_ticklabels[1]),
            )

        if self.idx_clip is not None:
            mat = mat[
                self.idx_clip[0] : -self.idx_clip[1],
                self.idx_clip[2] : -self.idx_clip[3],
            ]

        self.axs[row, col].im = _create_subplot(
            ax=self.axs[row, col],
            mat=mat,
            lims=lims,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        if cbar_bool:
            self.axs[row, col].cbar = _create_colorbar(
                fig=self.fig,
                im=self.axs[row, col].im,
                ax=self.axs[row, col],
                ticks=cbar_ticks,
                label=cbar_label,
                title=cbar_title,
                borderpad=cbar_borderpad,
                loc=cbar_loc,
                bbox_to_anchor=bbox,
                height=cbar_length,
                width=cbar_width,
                binary=cbar_binary,
                binary_ticklabels=binary_ticklabels,
                orientation=cbar_orientation,
            )

    class Text:
        def __init__(self, obj):
            self.obj = obj

        def xlabels_to_row(self, vals, idx=0):
            for idx_row, idx_col in product([idx], range(len(vals))):
                if vals[idx_col] is None:
                    continue

                ax = self.obj.axs[idx_row, idx_col]

                _set_ax_text(
                    ax=ax,
                    xlabel=_tex_str(vals[idx_col]) if self.obj.latex else vals[idx_col],
                )

        def ylabels_to_col(self, vals, idx=0, disable_tex=False):
            tex = self.obj.latex if not disable_tex else False
            for idx_row, idx_col in product(range(len(vals)), [idx]):
                ax = self.obj.axs[idx_row, idx_col]

                _set_ax_text(
                    ax=ax, ylabel=_tex_str(vals[idx_row]) if tex else vals[idx_row]
                )

        def ylabel_to_ax(self, row, col, val):
            _set_ax_text(
                ax=self.obj.axs[row, col],
                ylabel=_tex_str(val) if self.obj.latex else val,
            )

        def titles_to_row(self, vals, idx=0):
            for idx_row, idx_col in product([idx], range(len(vals))):
                ax = self.obj.axs[idx_row, idx_col]

                if vals[idx_col] is None:
                    continue

                _set_ax_text(
                    ax=ax,
                    title=_tex_str(vals[idx_col]) if self.obj.latex else vals[idx_col],
                )

        def labels_to_row(
            self,
            idx,
            vals,
            px=0.1,
            py=0.9,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", ec="black", fc="white"),
            latex_bold=True,
        ):
            for idx_row, idx_col in product([idx], range(len(vals))):
                if vals[idx_col] is None:
                    continue

                _add_label_to_ax(
                    ax=self.obj.axs[idx_row, idx_col],
                    px=px,
                    py=py,
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    bbox=bbox,
                    label=_tex_str(vals[idx_col], bold=latex_bold)
                    if self.obj.latex
                    else vals[idx_col],
                )

        def label_to_ax(
            self,
            row,
            col,
            val,
            px=0.05,
            py=0.95,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", ec="black", fc="white"),
            latex_bold=True,
        ):
            _add_label_to_ax(
                ax=self.obj.axs[row, col],
                px=px,
                py=py,
                ha=ha,
                va=va,
                fontsize=fontsize,
                bbox=bbox,
                label=_tex_str(val, bold=latex_bold) if self.obj.latex else val,
            )

    class Cmaps:
        def __init__(self):
            self.mag = plt.cm.gray
            self.bias = plt.cm.RdBu.reversed()
            self.std = plt.cm.magma
            self.binary = matplotlib.colors.ListedColormap(["black", "white"])

    class Cbar:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            self.obj.axs[row, col].cbar.remove()

        def remove_all(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def remove_rows(self, rows):
            rows = [rows] if isinstance(rows, int) else rows
            for row, col in product(rows, range(self.obj.ncols)):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def remove_cols(self, cols):
            cols = [cols] if isinstance(cols, int) else cols
            for row, col in product(range(self.obj.nrows), cols):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def share_row(
            self,
            idx=0,
            length=0.85,
            width=0.075,
            delta=0.05,
            loc="center right",
            anchors=(0, 0, 1, 1),
            borderpad=0,
            orientation="vertical",
            ticks=None,
            label=None,
            ticklabels=None,
            title=None,
            binary=False,
            binary_ticklabels=["excl.", "incl."],
        ):
            bbox = (anchors[0] + width + delta, anchors[1], anchors[2], anchors[3])

            for idx_row, idx_col in product([idx], range(self.obj.ncols)):
                if hasattr(self.obj.axs[idx_row, idx_col], "cbar"):
                    self.obj.axs[idx_row, idx_col].cbar.remove()

            if self.obj.latex:
                if binary:
                    binary_ticklabels[0] = _tex_str(binary_ticklabels[0])
                    binary_ticklabels[1] = _tex_str(binary_ticklabels[1])

                if label is not None:
                    label = _tex_str(label)

            _create_colorbar(
                fig=self.obj.fig,
                im=self.obj.axs[idx, self.obj.ncols - 1].im,
                ax=self.obj.axs[idx, self.obj.ncols - 1],
                width=width,
                height=length,
                loc=loc,
                bbox_to_anchor=bbox,
                borderpad=borderpad,
                orientation=orientation,
                ticks=ticks,
                label=label,
                ticklabels=ticklabels,
                title=title,
                binary=binary,
                binary_ticklabels=binary_ticklabels,
            )

        def share_col(
            self,
            idx=0,
            length=0.85,
            width=0.075,
            delta=0.05,
            loc="lower center",
            anchors=(0, 0, 1, 1),
            borderpad=0,
            orientation="horizontal",
            ticks=None,
            label=None,
            ticklabels=None,
            title=None,
            binary=False,
            binary_ticklabels=["excl.", "incl."],
        ):
            bbox = (anchors[0], anchors[1] - width - delta, anchors[2], anchors[3])

            for idx_row, idx_col in product(range(self.obj.nrows), [idx]):
                if hasattr(self.obj.axs[idx_row, idx_col], "cbar"):
                    self.obj.axs[idx_row, idx_col].cbar.remove()

            if self.obj.latex:
                if binary:
                    binary_ticklabels[0] = _tex_str(binary_ticklabels[0])
                    binary_ticklabels[1] = _tex_str(binary_ticklabels[1])

                if label is not None:
                    label = _tex_str(label)

            _create_colorbar(
                fig=self.obj.fig,
                im=self.obj.axs[self.obj.nrows - 1, idx].im,
                ax=self.obj.axs[self.obj.nrows - 1, idx],
                width=length,
                height=width,
                loc=loc,
                bbox_to_anchor=bbox,
                borderpad=borderpad,
                orientation=orientation,
                ticks=ticks,
                label=label,
                ticklabels=ticklabels,
                title=title,
                binary=binary,
                binary_ticklabels=binary_ticklabels,
            )

    class Ticks:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_all(self, exceptions=[]):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if (row, col) not in exceptions:
                    _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_rows(self, idx):
            rows = [idx] if isinstance(idx, int) else idx
            for row, col in product(rows, range(self.obj.ncols)):
                _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_cols(self, idx):
            cols = [idx] if isinstance(idx, int) else idx
            for row, col in product(range(self.obj.nrows), cols):
                _remove_ax_ticks(ax=self.obj.axs[row, col])

    class Lines:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_all(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_rows(self, rows):
            rows = [rows] if isinstance(rows, int) else rows
            for row, col in product(rows, range(self.obj.ncols)):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_cols(self, cols):
            cols = [cols] if isinstance(cols, int) else cols
            for row, col in product(range(self.obj.nrows), cols):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def set_color(self, row, col, color):
            _set_ax_line_color(ax=self.obj.axs[row, col], color=color)

        def remove_inner(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if col > 0:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["left"])
                if col < self.obj.ncols - 1:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["right"])
                if row > 0:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["top"])
                if row < self.obj.nrows - 1:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["bottom"])


def _create_colorbar(
    fig,
    im,
    ax,
    width=0.05,
    height=1,
    loc="center right",
    bbox_to_anchor=(0.05, 0, 1, 1),
    borderpad=0,
    orientation="vertical",
    ticks=None,
    label=None,
    binary=False,
    ticklabels=None,
    title=None,
    binary_ticklabels=("excl", "incl"),
):
    axins = inset_axes(
        ax,
        width=f"{100 * width}%",
        height=f"{100 * height}%",
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=borderpad,
    )

    cbar = fig.colorbar(
        im,
        cax=axins,
        ticks=ticks,
        orientation=orientation,
        label=label,
    )

    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels=ticklabels)

    if title is not None:
        cbar.ax.set_title(title)

    if binary:
        cbar.set_ticks(ticks=[1 / 4, 3 / 4])
        cbar.set_ticklabels(ticklabels=binary_ticklabels)

    return cbar


def _create_subplot(ax, mat, lims, cmap, xlabel, ylabel, title):
    im = ax.imshow(mat, vmin=lims[0], vmax=lims[1], cmap=cmap, interpolation=None)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return im


def _remove_ax_lines(ax, dirs=("left", "right", "top", "bottom")):
    for d in dirs:
        ax.spines[d].set_visible(False)


def _remove_ax_ticks(ax):
    ax.set(xticks=[], yticks=[])


def _set_ax_shape(ax, shape, white=True):
    return ax.imshow(
        np.ones(shape), cmap=plt.cm.gray if white else plt.cm.binary, vmin=0, vmax=1
    )


def _set_ax_line_color(ax, color, dirs=["left", "right", "top", "bottom"]):
    for d in dirs:
        ax.spines[d].set_color(color)


def _tex_str(string, bold=False):
    # note: replace \ with \\ in argument string
    if bold:
        return "\\textnormal{" + "\\textbf{" + string + "}}"
    return "\\textnormal{" + string + "}"


def _add_label_to_ax(
    ax,
    label,
    px=0.05,
    py=0.94,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", ec="black", fc="white"),
):
    ax.text(
        px,
        py,
        s=label,
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=ax.transAxes,
        bbox=bbox,
    )


def _set_ax_text(ax, xlabel=None, ylabel=None, title=None):
    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    ax.set_title(title) if title is not None else None
