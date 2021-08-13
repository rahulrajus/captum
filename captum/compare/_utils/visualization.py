#!/usr/bin/env python3
from typing import Dict, List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

def visualize_comparison(
    stats_dict: Dict[str, Dict[str, Dict[str, Tensor]]],
    layers1: List[str],
    layers2: List[str],
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    interpolation: str = "nearest",
    cmap: Union[None, str] = "hot",
    show_colorbar: bool = True,
    title: Union[None, str] = None,
    x_label: str = "Layer",
    y_label: str = "Layer",
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
    save_path: str = None,
):
    r"""
    Visualization function for displaying similarity between multiple layers. This function is intended to be used with the output dictionary from the compare() function in ModuleSimilarity based classes.

    Args:
        stats_dict (Dict[str, Dict[str, Dict[str, Tensor]]]): Input structure for
                    generating visualization. Should contain stats for each layer_pair that can be accessed via a dictionary of dictionaries. As an example, to get stats on comparison between conv1 and fc2 -> stats_dict['conv1']['fc2']. This interface assumes that the stats dictionary contains the key "similarity" with the summary score for that layer pair
        layers1 (List[str]): List of layers from the first model to compare
        layers2 (List[str]): List of layers from the second model to compare
        plt_fig_axis (tuple, optional): Tuple of matplotlib.pyplot.figure and axis
                    on which to visualize. If None is provided, then a new figure
                    and axis are created.
        interpolation (str): Interpolation method used if image is upscaled
        cmap (Union[None, str]): String corresponding to desired colormap for
                    visualization. Default is set to 'hot'.
        show_colorbar (bool): Flag for determining whether to show a color bar or not
        title (str): Title string for plot. If None, no title is set.
        x_label (str): Label to place on the x axis
        y_label (str): Label to place on the y axis
        fig_size (tuple, optional): Size of figure created.
        use_pyplot (boolean): If true, uses pyplot to create and show
                    figure and displays the figure after creating. If False,
                    uses Matplotlib object oriented API and simply returns a
                    figure object without showing.
        save_path (str): Location to save plot if desired
    Returns:
        2-element tuple of **figure**, **axis**:
        - **figure** (*matplotlib.pyplot.figure*):
                    Figure object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same figure provided.
        - **axis** (*matplotlib.pyplot.axis*):
                    Axis object on which visualization
                    is created. If plt_fig_axis argument is given, this is the
                    same axis provided.
    Examples::
        >>> # BasicModel_ConvNet takes a single input tensor of images Nx1x10x10,
        >>> net1 = BasicModel_ConvNet()
        >>> net2 = BasicModel_ConvNet()
        >>> net1_layers = list(_model_children(net1))
        >>> net2_layers = list(_model_children(net2))
        >>> # Get all combinations between layers
        >>> # Initialize similarity class
        >>> cka_sim = CKASimilarity(net1, net2)
        >>> _ , cka_dict = cka_sim.compare(
        >>>    inputs=inputs,
        >>>    layer_pairs=pairs,
        >>>    transform=avg_pool_4d,
        >>>    debiased=True,
        >>>    kernel="linear",
        >>>    sigma=0.5,
        >>> )
        >>> visualize_comparison(
        >>>    cka_dict,
        >>>    net1_layers,
        >>>    net2_layers,
        >>>    cmap="magma",
        >>>    title="CKA: Basic Conv 1 vs. Basic Conv 2",
        >>> )
    """

    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    score_mat = np.zeros((len(layers1), len(layers2)))

    for i, l1 in enumerate(layers1):
        for j, l2 in enumerate(layers2):
            score = None
            if l1 in stats_dict:
                score_dict = stats_dict[l1][l2]
                if score_dict and ("similarity" in score_dict):
                    score = score_dict["similarity"].cpu().numpy()
            score_mat[i][j] = score

    plot = plt_axis.imshow(score_mat, cmap=cmap, interpolation="nearest")

    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("right", size="5%", pad=0.2)
        plt_fig.colorbar(plot, orientation="vertical", cax=colorbar_axis)

    if title:
        plt_axis.set_title(title)

    plt_axis.set_xlabel(x_label)
    plt_axis.set_ylabel(y_label)

    plt.xticks(range(len(layers1)), layers1, align='center')
    plt.yticks(range(len(layers2)), layers2, align='center')

    if save_path:
        plt.savefig(save_path)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis
