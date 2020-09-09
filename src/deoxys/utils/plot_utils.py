# -*- coding: utf-8 -*-

__author__ = "Ngoc Huynh Bao"
__email__ = "ngoc.huynh.bao@nmbu.no"


"""
This file contains multiple helper function for plotting diagram and images
using matplotlib
"""


import pandas as pd
import matplotlib.pyplot as plt
from .data_utils import read_csv


def plot_log_performance_from_csv(filepath, output_path):
    """
    Plot and save multiple performance figure using a log file generated from
    tensorflow.keras.callbacks.CSVLogger

    :param filepath: filename of the log file
    :type filepath: str
    :param output_path: path to the folder for saving plotted diagram
    :type output_path: str
    """
    df = read_csv(filepath, index_col='epoch')

    # Plot all data
    _plot_data(df, 'All parameters', df.columns, output_path + '/all.png')

    # Plot train data
    train_keys = [key for key in df.columns if 'val' not in key]
    _plot_data(df, 'Train Performance', train_keys, output_path + '/train.png')

    # Plot val data
    val_keys = [key for key in df.columns if 'val' in key]
    _plot_data(df, 'Validation Performance',
               val_keys, output_path + '/val.png')

    # Plot compare train and val
    compare = [[key, 'val_' + key] for key in train_keys]
    for keys in compare:
        _plot_data(df, '{} Performance'.format(
            keys[0]), keys, output_path + '/{}.png'.format(keys[0]))


def plot_evaluation_performance_from_csv(filepath, output_path):
    # Load data to file
    df = read_csv(filepath, index_col='epoch')

    # Plot evaluation
    _plot_data(df, 'Evaluation Performance', df.columns,
               output_path + '/evaluation.png')


def mask_prediction(output_path, image, true_mask, pred_mask,
                    title='Predicted',
                    mask_levels=None, channel=None):
    """
    Generate and save predicted images with true mask and predicted mask as
    contouring lines

    :param output_path: path to folder for saving the output images
    :type output_path: str
    :param image: a collection of original 2D image data
    :type image: numpy array / collection
    :param true_mask: a collection of true mask data
    :type true_mask: numpy array / collection
    :param pred_mask: a collection of predicted mask data
    :type pred_mask: numpy array / collection
    :param title: title of the diagram, defaults to 'Predicted'
    :type title: str, optional
    :param mask_levels: mask_levels when contouring the images,
    defaults to None
    :type mask_levels: int, or list of int or float, optional
    :param channel: if the original image has multiple channels, this indicates
    which channel to plot the images, defaults to None
    :type channel: int, optional
    """
    if not mask_levels:
        mask_levels = 1
    kwargs = {}
    if not channel:
        if (len(image.shape) == 2
                or (len(image.shape) == 3 and image.shape[2] == 3)):
            image_data = image
        else:
            image_data = image[..., 0]
            kwargs['cmap'] = 'gray'
    else:
        image_data = image[..., channel]
        kwargs['cmap'] = 'gray'

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    if len(true_mask_data.shape) == 3:
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

    plt.figure()
    plt.imshow(image_data, **kwargs)
    true_con = plt.contour(
        true_mask_data, 1, levels=mask_levels, colors='yellow')
    pred_con = plt.contour(
        pred_mask_data, 1, levels=mask_levels, colors='red')

    plt.title(title)
    plt.legend([true_con.collections[0],
                pred_con.collections[0]], ['True', 'Predicted'])
    plt.savefig(output_path)
    plt.close('all')


def plot_images_w_predictions(output_path, image, true_mask, pred_mask,
                              title='Predicted',
                              channel=None):
    """
    Generate and save predicted images with true mask and predicted mask as
    separate images

    :param output_path: path to folder for saving the output images
    :type output_path: str
    :param image: a collection of original 2D image data
    :type image: numpy array / collection
    :param true_mask: a collection of true mask data
    :type true_mask: numpy array / collection
    :param pred_mask: a collection of predicted mask data
    :type pred_mask: numpy array / collection
    :param title: title of the diagram, defaults to 'Predicted'
    :type title: str, optional
    :param channel: if the original image has multiple channels, this indicates
    which channel to plot the images, defaults to None
    :type channel: int, optional
    """
    kwargs = {}
    if not channel:
        if (len(image.shape) == 2
                or (len(image.shape) == 3 and image.shape[2] == 3)):
            image_data = image
        else:
            image_data = image[..., 0]
            kwargs['cmap'] = 'gray'
    else:
        image_data = image[..., channel]
        kwargs['cmap'] = 'gray'

    true_mask_data = true_mask
    pred_mask_data = pred_mask

    if len(true_mask_data.shape) == 3:
        true_mask_data = true_mask[..., 0]
        pred_mask_data = pred_mask[..., 0]

    fig, (img_ax, true_ax, pred_ax) = plt.subplots(1, 3)
    img_ax.imshow(image_data, **kwargs)
    img_ax.set_title('Images')
    true_ax.imshow(true_mask_data)
    true_ax.set_title('True Mask')
    pred_ax.imshow(pred_mask_data)
    pred_ax.set_title('Predicted Mask')

    plt.suptitle(title)
    plt.savefig(output_path)
    plt.close('all')


def _plot_data(dataframe, name, columns, filename):
    ax = dataframe[columns].plot()
    epoch_num = dataframe.shape[0]
    if epoch_num < 20:
        ax.set_xticks(dataframe.index)
    else:
        tick_distance = epoch_num // 10
        ax.set_xticks([tick for i, tick in enumerate(
            dataframe.index) if i % tick_distance == 0])
    ax.set_title(name)
    plt.savefig(filename)
