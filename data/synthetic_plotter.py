import os, sys, time
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

COLORS = ['blue', 'green', 'cyan', 'magenta', 'grey']
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", show_legend=True, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_legend:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_edge_connections(edge1, edge2, color_list = COLORS):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    plot_single_edge(edge1, ax1, show_legend=False)
    plot_single_edge(edge1, ax2, title='Connections after CP')
    plt.tight_layout()
    plt.show()

def plot_edge_connection(edge, color_list = COLORS, title='Connections before CP'):
    fig, ax = plt.subplots()
    plot_single_edge(edge, ax, title=title)
    plt.tight_layout()
    plt.show()

def plot_single_edge(edge, cur_ax, color_list = COLORS, show_legend=True, title='Connections before CP'):
    x = color_list
    y = color_list
    data = 1 - edge
    values = np.array(list("01"))
    norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1], 2, clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: values[::-1][norm(x)])

    im = heatmap(data, y, x, ax=cur_ax, show_legend=show_legend,
                    cmap=plt.get_cmap("tab20", 2), norm=norm,
                    cbar_kw=dict(ticks=[0.25, 0.75], format=fmt),
                    cbarlabel="Edge Type")

    annotate_heatmap(im, valfmt=fmt, size=10, fontweight="bold", threshold=-1,
                     textcolors=["white", "black"])
    cur_ax.set_title(title)
    
    
def plot_x_y(loc, change_point, loc0=None, color_list=COLORS, title='Trajectory', vertical=False):
    # plot changed curve
    if vertical:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("{}: x axis".format(title))
    ax2.set_title("{}: y axis".format(title))
    for i in range(loc.shape[-1]):
        ax1.plot(loc[:, 0, i], color=color_list[i], label=color_list[i])
        if loc0 is not None:
            ax1.plot(loc0[:, 0, i], color=color_list[i], linestyle='dashed', alpha=0.6)
        ax1.axvline(x=change_point, color='red', linestyle='dashed')

    for i in range(loc.shape[-1]):
        if loc0 is not None:
            ax2.plot(loc0[:, 1, i], color=color_list[i], linestyle='dashed', alpha=0.6)
        ax2.plot(loc[:, 1, i], color=color_list[i], label=color_list[i])
        ax2.axvline(x=change_point, color='red', linestyle='dashed')
    plt.legend()
    plt.show()
    
def plot_trajectory(loc, change_point, loc0=None, color_list=COLORS, title='Trajectory'):
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
#     ax.set_xlim([-5., 5.])
#     ax.set_ylim([-5., 5.])
    for i in range(loc.shape[-1]):
        if loc0 is not None:
            ax.plot(loc0[:, 0, i], loc0[:, 1, i], color=color_list[i], linestyle='dashed', alpha=0.6)
        ax.plot(loc[:, 0, i], loc[:, 1, i], color=color_list[i], label=color_list[i])
        ax.plot(loc[0, 0, i], loc[0, 1, i], 'd', color=color_list[i])
        ax.plot(loc[change_point, 0, i], loc[change_point, 1, i], '.', color='r')
    plt.title(title)
    plt.legend()
    plt.figure()