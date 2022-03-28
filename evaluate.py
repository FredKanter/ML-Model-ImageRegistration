import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from save_results import save_to_disk


def save_metrics(path, res_dict):
    # save results as arrays
    for indx, key in enumerate(res_dict.keys()):
        res_array = np.array(res_dict[key])
        np.save(os.path.join(path, key), res_array)

        # write file
        if indx == 0:
            file = open(path + '/metrics.txt', 'w')
            file.write(f'{key}: {tensors2array(res_dict[key])}\n')
            file.close()
        else:
            file = open(path + '/metrics.txt', 'a')
            file.write(f'{key}: {tensors2array(res_dict[key])}\n')
            file.close()


def save_images(path, imgs_list, title_list, gray_mode=True, titles=True):
    # gets list of images and stores them to disk using preset of quality settings
    if not len(imgs_list) == len(title_list):
        raise RuntimeError(f'Number titles {len(title_list)} does not match number images {len(imgs_list)}')
    for ii in range(len(imgs_list)):
        fig = plt.figure()
        if gray_mode:
            plt.imshow(imgs_list[ii].detach(), cmap='gray')
        else:
            plt.imshow(imgs_list[ii].detach(), aspect='equal')
        if titles:
            plt.title(title_list[ii])
        plt.axis('off')
        save_to_disk(fig, title_list[ii], path)
