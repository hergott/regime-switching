'''
This activity function originally produced 4 charts. Some
sections have been commented out to create only the
chart of regime probabilities.
'''

import matplotlib.pyplot as plt  # pylint: disable=import-error
import seaborn as sns  # pylint: disable=import-error
import numpy as np
import math
import itertools
import tempfile
import os
import json
import logging

from azure.storage.blob import BlobServiceClient  # pylint: disable=all


def main(inputs: str) -> str:
    inputs_json = json.loads(inputs)

    res = inputs_json['res']
    container_name = inputs_json['container_name']
    blob_name = inputs_json['blob_name']
    account_url = res['account_url']

    # Set plot environment.
    plt.rcParams['axes.facecolor'] = '#ffffff'
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['axes.edgecolor'] = '#999999'
    plt.rcParams['legend.facecolor'] = '#dddddd'
    plt.rcParams['legend.edgecolor'] = '#aaaaaa'

    # # When using all four charts (see comments below).
    # plt.rcParams['axes.labelsize'] = 6
    # plt.rcParams['axes.titlesize'] = 7
    # plt.rcParams['xtick.labelsize'] = 5
    # plt.rcParams['ytick.labelsize'] = 5
    # plt.rcParams['axes.titlepad'] = 3

    # When displaying one chart of regime probabilities.
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['axes.titlepad'] = 6

    text_color = '#000000'
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    sns.set_palette("bright")

    # # Used when plotting the full set of 4 charts (see comments below).
    # fig, axes = plt.subplots(4, 1, facecolor='#ffffff')
    # blob_name = blob_name + '/charts.png'

    # Used when plotting the single chart of regime probabilities.
    fig, axes = plt.subplots(1, 1, facecolor='#ffffff')
    blob_name = blob_name + '/chart.png'

    nobs = res['nobs']
    n_1000s = math.ceil(nobs/1000.)

    linewidth = max(1.6-0.1*n_1000s, 0.8)

    seaborn_pallete = ['#4976b5', '#c2535a', '#977556', '#de8cc8',
                       '#53b1cb', '#4976b5', '#d68455',  '#8578bb']

    colors = seaborn_pallete[4:]

    regimes_palette = itertools.cycle(seaborn_pallete)
    palette = itertools.cycle(colors)

    def data_dim(data):
        if np.ndim(data) > 1 and data.shape[1] > data.shape[0]:
            y = data.T
        elif np.ndim(data) > 1:
            y = np.squeeze(data, axis=-1)
        else:
            y = data

        l = max(y.shape)
        x = np.arange(start=0, stop=l)

        return x, y

    def add_labels(title, ylabel, ax, yLabel_color='#000000', grid=True, xlabel=None):
        ax.set_ylabel(ylabel, color=yLabel_color)
        ax.set_title(title)

        if xlabel is not None:
            ax.set_xlabel(xlabel, color=yLabel_color)

        if grid:
            ax.grid(color=(.0, .0, .0, 0.1),
                    linestyle='--', linewidth=0.15)

    # Plot regime probabilities.
    #
    # use 'axes[0]' when plotting four charts
    x, y = data_dim(np.asarray(res['smoothed'])*100.)

    regimes = y.shape[1]

    for c in range(regimes):
        color = next(regimes_palette)
        sns.lineplot(x=x, y=y[:, c], ax=axes,
                     color=color, linewidth=linewidth, label=f'regime {c+1}')

        axes.fill_between(x, y[:, c], step="pre", alpha=0.2, color=color)

    # # 4 charts
    # axes.legend(fontsize=5, loc=2)

    # 1 chart
    #
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html
    axes.legend(fontsize=9, loc=7)

    # axes.set_ylim(0, 100)
    # axes[0].set_xlim(0, x.shape[0])

    # # When using 4 charts
    # add_labels('Smoothed regime probabilities', 'Probability (%)', axes)

    # When using 1 chart
    add_labels('Smoothed regime probabilities',
               'Probability (%)', axes, xlabel='Observation #')

    # # The next three charts can be removed temporarily to focus attention on the primary
    # # chart of regime probabilities.
    # #
    # # Plot y vs regime 1.
    # sm = np.asarray(res['smoothed'])
    # x, y = data_dim(sm[0, :]*100.)
    # col = next(palette)
    # sns.lineplot(x=x, y=y, ax=axes[1], color=col, linewidth=linewidth)
    # add_labels('', 'Regime 1 prob. (%)',
    #            axes[1], yLabel_color=col, grid=False)

    # ax2 = axes[1].twinx()

    # x, y = data_dim(np.asarray(res['y']))
    # col = next(palette)
    # sns.lineplot(x=x, y=y, ax=ax2, color=col, linewidth=linewidth)
    # add_labels('Regime 1 probability vs. y values', 'y', ax2, yLabel_color=col)

    # # Plot y-hat
    # x, y = data_dim(np.asarray(res['yhat']))
    # sns.lineplot(x=x, y=y, ax=axes[2], color=next(
    #     palette), linewidth=linewidth)
    # add_labels('Weighted regression predicted y', 'y-hat', axes[2])

    # # Plot residuals
    # x, y = data_dim(np.asarray(res['residuals']))
    # sns.lineplot(x=x, y=y, ax=axes[3], color=next(
    #     palette), linewidth=linewidth)
    # add_labels('Residuals from weighted regression', 'Residual', axes[3])

    # fig.subplots_adjust(bottom=0.04, top=0.97, left=0.1,
    #                     right=0.93, hspace=0.50)

    # Used when plotting only the regime probabilities.
    #
    fig.subplots_adjust(bottom=0.08, top=0.95, left=0.1,
                        right=0.98, hspace=0.50)

    tempFilePath = tempfile.gettempdir()
    file_handle, file_name = tempfile.mkstemp(dir=tempFilePath, suffix='.png')
    plt.savefig(file_name, format='png', dpi=300, orientation='landscape')

    blob_service = BlobServiceClient(account_url=account_url,
                                     credential=os.environ['BlobCredentials'])

    blob = blob_service.get_blob_client(
        container=container_name, blob=blob_name)

    with open(file_name, 'rb') as data:
        blob.upload_blob(data)

    os.close(file_handle)

    if os.path.exists(file_name):
        os.remove(file_name)

    # return None as a string because the Durable Function
    # extension seems to work well with strings.
    return 'None'
