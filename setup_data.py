import torch
import numpy as np
import math
from torch.utils import data
from multiprocessing import Lock
import pickle
import os
import h5py
import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
import random

import image_registration.reg_utils.tools as tools
from image_registration.reg_utils.grids import Grid
from image_registration.reg_utils.deformations import set_deformation
from image_registration.reg_utils.interpolation import set_interpolater
from learn_optimization import object_functions as obFct
from utils.save_results import write_file
from image_registration.reg_utils.data import add_noise, init_generator, normalize_img, resize_image


class OnFlyDataset(data.Dataset):
    """
    Data set to create deformation for image registration using scaling factor for larger deformations
    Data should be created during training using seed point parameter
    Define interval (tuple) with values for parameter used in deformation and [0-max], use scale to construct items
    """
    def __init__(self, num_items, img_path, omega, dfm, dfm_params, noise=True, noise_level=3, invert=False):
        """
        :param BaseImg: image to create reference and be used as template (torch.tensor)
        :param omega: range of image values
        :param num_items: items in dataset (int)
        :param dfm: deformation function (Deformation)
        :param dfm_params: parameter to create transformed image with (dict)
        """
        self.images, _ = tools.load_images(img_path)
        self.scale_fac = dfm_params['scale_fac'] if 'scale_fac' in dfm_params.keys() else 1
        if not self.scale_fac == 1:
            print('Warning tiny images hack enabled')
            self.images = resize_image(self.images, 1 / self.scale_fac)
        self.num_items = num_items
        self.dfm = dfm
        m = torch.tensor(self.images[0].shape)
        self.inter = set_interpolater('linearFAIR', omega, m, Grid(1, omega, m).getSpacing())
        self.noise_value = dfm_params['noise_level'] if 'noise_level' in dfm_params.keys() else noise_level
        dfm_type = dfm_params['type'] if 'type' in dfm_params.keys() else dfm.name
        self.dfm_generator = init_generator(dfm_type, 1, img_path, omega, dfm_params)
        transform_ops = [Normalize()]
        if noise and invert:
            transform_ops = [*transform_ops, AddNoise(self.noise_value, self.images[0].size()), Invert()]
        elif invert:
            transform_ops = [*transform_ops, Invert()]
        elif noise:
            transform_ops = [*transform_ops, AddNoise(self.noise_value, self.images[0].size())]
        self.transform = transforms.Compose(transform_ops)

        # some checks
        if not dfm.name == 'nonpara' and not set(['angle', 'shift', 'shear']) <= set(dfm_params.keys()):
            raise RuntimeError(f'{dfm_params.keys()} is missing one argument of angle, shift or shear'
                               f' for deformation type {dfm.name}')
        if dfm.name == 'nonpara' and not set(['ratio', 'spacing_factor']) <= set(dfm_params.keys()):
            raise RuntimeError(f'{dfm_params.keys()} is missing one argument of ratio or spacing_factor'
                               f' for deformation type {dfm.name}')
        uneven_img_ids = [id for id, img in enumerate(self.images) if np.diff(np.array(img.shape)).sum()]
        if uneven_img_ids:
            raise RuntimeError(f'Image dimensions for image {uneven_img_ids} are uneven.'
                               f'The framework requires even image dimensions')

    def __len__(self):
        # num_items can be arbitrary, since new samples should be create each epoch (could be coupled to some seed)
        return self.num_items

    def __getitem__(self, item):
        """
        :param item: idx given by generator
        :return: target deformation parameters, {reference image, gt deformation parameters, target image}
        """
        BaseImg = self.images[np.random.randint(0, len(self.images))]

        xmin, x0, xc = self.dfm_generator.create_params()

        # transform image for supervised learning
        params = {'R': self.inter.interpolate(BaseImg, xc).squeeze(), 'x*': xmin, 'T': BaseImg}

        if self.transform:
            params = self.transform(params)

        return x0, params


class Normalize(object):
    @staticmethod
    def __normalize__(image):
        return normalize_img(image)

    def __call__(self, params):
        return {'T': self.__normalize__(params['T']), 'R': self.__normalize__(params['R']), 'x*': params['x*']}


class AddNoise(object):
    def __init__(self, level, imgSz):
        self.lvl = level
        self.imgSz = imgSz
        self._add_noise = add_noise

    def __call__(self, params):
        T = params['T']
        T = self._add_noise(T, self.lvl, False)
        return {'T': T, 'R': params['R'], 'x*': params['x*']}


class Invert(object):
    def __invert__(self, image, no_invert):
        if no_invert:
            return image
        else:
            return tools.normalize(image*-1)

    def __call__(self, params):
        T, R = params['T'], params['R']
        no_invert = bool(random.getrandbits(1))
        return {'T': self.__invert__(T, no_invert), 'R': self.__invert__(R,no_invert), 'x*': params['x*']}


class HdfDataset(data.Dataset):
    def __init__(self, path, file, add_noise=False):
        data = h5py.File(os.path.join(path, file), 'r')
        self.data = self.decode_h5(data)
        self.add_noise = add_noise
        self.lock = Lock()

    def __len__(self):
        return len(self.data['x0'])

    def __getitem__(self, item):
        self.lock.acquire()
        params = {'R': self.data['R'][item], 'x*': self.data['x*'][item], 'T': self.data['T'][item]}
        x0 = self.data['x0'][item]
        self.lock.release()
        return x0, params

    @staticmethod
    def decode_h5(fin):
        return {'R': fin['R'][()], 'T': fin['T'][()], 'x*': fin['x*'][()], 'x0': fin['x0'][()]}


def set_generator(name, file, num_items, data_path, omega, dfm, dfm_params, noise=False, invert=False):
    if name == 'onfly':
        print('On fly data generation - single image\n')
        return OnFlyDataset(num_items, data_path, omega, dfm, dfm_params, noise, invert)
    else:
        print(f'On fly data generation disabled. Use fix HDF: {file}\n')
        return HdfDataset(data_path, file)



