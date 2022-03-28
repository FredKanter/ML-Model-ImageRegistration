import os
import matplotlib.pyplot as plt
import csv
import torch
import numpy as np


def tensors2array(input):
    batch_size = input.shape[0]
    tmp = []
    for ii in range(batch_size):
        tmp.append(input[ii].detach().numpy())
    return np.array(tmp)


def savefig(fn, make_tight=False):
    plot_settings = {'format': 'png', 'dpi': 600, 'bbox_inches': 'tight', 'pad_inches': 0} if make_tight else {'format': 'png', 'dpi': 600}
    plt.savefig(fn + '.png', **plot_settings)


def save_to_disk(fig, filename, path, make_tight=False):
    dir_path = os.path.join(path, filename)

    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(fig.number)
    if make_tight:
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
    savefig(dir_path, make_tight)
    plt.close()


def prepare_results(result_path):
    params = {}
    with open(os.path.join(result_path, 'parameters.txt'), 'r') as pr:
        for line in pr:
            entries = line.split(':')
            if entries[0] == 'Time':
                sum_entries = ''
                for ii in range(1, len(entries)):
                    sum_entries += entries[ii] + ':'
                params[entries[0]] = sum_entries.strip('\n:').strip()
            else:
                params[entries[0]] = entries[1].strip('\n').strip()

    sub_dirs = result_path.split('/')
    params['Date'] = sub_dirs[2]
    params['Run'] = sub_dirs[3]

    files_eval = [npy for npy in os.listdir(result_path) if npy.endswith('.npy')]
    for cf in files_eval:
        tmp_cf = np.load(os.path.join(result_path, cf), allow_pickle=True)
        name = cf.split('.')[0]
        if not len(tmp_cf.shape) == 0:
            if torch.is_tensor(tmp_cf):
                tmp_cf = tensors2array(tmp_cf)
            params['Mean_' + name] = np.round(np.mean(tmp_cf), 4)
        else:
            params[name] = tmp_cf

    return [params]
