import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
import warnings
from scipy.interpolate import Rbf
import scipy

import tools as tools
from grids import Grid, scale_grid, plot_grid_fair, reshape_grid
from deformations import set_deformation
from interpolation import set_interpolater


class DataCreator:
    def __init__(self, name, num_items, path, omega, params):
        super(DataCreator, self).__init__()
        self.name = name
        self.num_items = num_items
        self.omega = omega
        self.params = params
        self.reduction = params['reduction'] if 'reduction' in params.keys() else 1
        self.scale_fac = params['scale_fac'] if 'scale_fac' in params.keys() else 1
        # all images should have same size, so m, h, and grid can be set here
        self.m = (torch.tensor(tools.load_images(path)[0][0].shape) // self.scale_fac).int()
        self.grid = Grid(1, self.omega, self.m)
        self.h = self.grid.getSpacing()
        self.img_dims = len(self.m)

    def create_data(self, path_images, verbose=False, invert=False):
        xmin, x0 = [], []
        imgT, imgR = [], []

        # load image data
        images, _ = tools.load_images(path_images)
        if not self.scale_fac == 1:
            print('Warning tiny images hack enabled')
            images = resize_image(images, 1 / self.scale_fac)

        if not isinstance(images, list):
            raise TypeError('Images should be passed in list to include different images in one data set')

        imgSet = images.copy()

        for cimg in imgSet:
            # normalize img for learning
            cimg = normalize_img(cimg)
            for ii in range(self.num_items):
                min_params, start_params, xc = self.create_params()
                # get minimal shift coherent with different spacings
                x0, xmin = [*x0, start_params], [*xmin, min_params]

                # transform image for supervised learning
                inter = set_interpolater('linearFAIR', self.omega, self.m, self.h)
                imgR = [*imgR, inter.interpolate(cimg, xc).squeeze()]
                # add noise to grid to create two different images/ image grid
                imgT = [*imgT, add_noise(cimg, self.params['noise_level'], invert)]

                if verbose:
                    B, C, HW = xc.shape
                    sxc = xc.reshape(B, C, int(math.sqrt(HW)), int(math.sqrt(HW)))
                    plt.figure(), plt.imshow(cimg, cmap='gray')
                    plot_grid_fair(sxc, imgR[-1], cimg, self.omega), plt.show()

        # convert list to tensors
        return torch.stack(xmin), torch.stack(x0), torch.stack(imgT), torch.stack(imgR)

    def create_params(self):
        # unique for each type of param creation (affine, nonpara)
        reduction = self.reduction
        spacing_factor = self.params['spacing_factor']
        xc = self.grid.getCellCentered().clone()

        m = min(self.m).item()
        max_reduction = math.ceil(math.log2(m)) - 1
        reduction = min(reduction, max_reduction)

        # downscale
        factor = int(math.pow(2, reduction))
        q = (self.m.float() / factor).int()
        small_grid = scale_grid(xc, self.m, q)

        # compute new spacing / m should be enable uneven image dims
        m_new = q.int()
        h = torch.zeros(len(m_new))
        for k in range(0, len(m_new)):
            h[k] = (self.omega[2 * k + 1] - self.omega[2 * k]) / m_new[k].float()

        dims = q.numel()
        xc_np = reshape_grid(small_grid, q).squeeze().permute((dims, *[d for d in np.arange(0, dims)])).detach().numpy()

        # slim version, works dim free except of get_indices subroutine
        HW = [u.item() for u in q]
        spaces = [np.linspace(0, u - 1, u) for u in HW]
        meshes = np.meshgrid(*spaces, indexing='ij')

        while True:
            inds = self.get_indices(HW)
            bases = [xc_np[ii][tuple(inds)] for ii in range(xc_np.shape[0])]

            dd = [base + np.random.normal(0, h.item() * spacing_factor, size=base.size) for base, h in zip(bases, h)]
            with warnings.catch_warnings(record=True):
                # Cause all warnings to always be triggered.
                warnings.simplefilter("error")
                try:
                    # init all Rbf functions and evaluate them for all dimensions
                    rbfs = [Rbf(*inds, d) for d in dd]
                    di = [r(*meshes) for r in rbfs]
                    break
                except np.linalg.LinAlgError:
                    continue
                except scipy.linalg.misc.LinAlgWarning:
                    continue

        # recreate tensor
        xc_gt = torch.stack([torch.tensor(d) for d in di], dim=0).unsqueeze(0).flatten(-dims, -1)

        # rescale to full size and return params
        xc_gt = scale_grid(xc_gt, q, self.m)
        min_params = (xc_gt - xc).flatten(-len(self.m), -1).squeeze()
        start_params = torch.zeros(xc.flatten(-len(self.m), -1).squeeze().shape, dtype=xc.dtype)

        return min_params, start_params, xc_gt.clone()

    def get_indices(self, HW):
        # get coordinates to use for deformation, use hard coded version for 2 or 3 dim (dims as switch)
        # define point indices and hart coded and combine in list
        dims = len(HW)
        nb_points = max(HW) // int(self.params['ratio'])
        if dims == 2:
            H, W = HW[0], HW[1]
            # first fixed corner points + some random points
            x = np.concatenate((np.array([0, H-1, H-1, 0]), np.random.randint(0, H-1, size=nb_points)))
            y = np.concatenate((np.array([0, W-1, 0, W-1]), np.random.randint(0, W-1, size=nb_points)))
            indices = [x, y]
        elif dims == 3:
            H, W, D = HW[0], HW[1], HW[2]
            # first fixed corner points + some random points
            x = np.concatenate((np.array([0, H-1, H-1, 0, 0, H-1, H-1, 0]), np.random.randint(0, H-1, size=nb_points)))
            y = np.concatenate((np.array([0, 0, W-1, W-1, 0, 0, W-1, W-1]), np.random.randint(0, W-1, size=nb_points)))
            z = np.concatenate((np.array([0, 0, 0, 0, D-1, D-1, D-1, D-1]), np.random.randint(0, D-1, size=nb_points)))
            indices = [x, y, z]
        else:
            raise RuntimeError(f'Creating indices for given dimension {dims} not implemented.')
        # inds tuple needed to address numpy array indices and list to use Rbf function
        return indices


class DataCreatorAffine(DataCreator):
    def __init__(self, name, num_items, path, omega, params):
        super(DataCreatorAffine, self).__init__(name, num_items, path, omega, params)

    def create_params(self, dtype=torch.double):
        dfm = set_deformation('affine', self.m.numel())

        angle = np.random.uniform(-self.params['angle'], self.params['angle'])
        shear = self.params['shear']
        shift = self.m / self.params['shift'] * self.h
        xshift = torch.tensor(np.random.uniform(-shift[0], shift[0]), dtype=dtype)
        yshift = torch.tensor(np.random.uniform(-shift[1], shift[1]), dtype=dtype)
        # args default is largest suitable deformation
        w_top = torch.tensor(math.pi * angle / 180, dtype=dtype)

        # deformation matrix
        w1 = torch.cos(w_top) + np.random.uniform(-shear, shear)
        w2 = -torch.sin(w_top) + np.random.uniform(-shear, shear)
        w4 = torch.sin(w_top) + np.random.uniform(-shear, shear)
        w5 = torch.cos(w_top) + np.random.uniform(-shear, shear)
        min_params = torch.tensor([w1, w2, w4, w5, xshift, yshift], dtype=dtype)
        start_param = torch.tensor([[1.005, 0], [0, 1.005], [math.sqrt(2) * self.h[0], math.sqrt(2) * self.h[1]]],
                                   dtype=dtype).flatten()

        xc = dfm.evaluate(min_params.unsqueeze(0), self.grid.getCellCentered(), self.omega.unsqueeze(0))
        return min_params, start_param, xc


def init_generator(name, num_items, path, omega, params):
    if name == 'affine':
        return DataCreatorAffine(name, num_items, path, omega, params)
    elif name == 'nonpara':
        return DataCreator(name, num_items, path, omega, params)
    else:
        print(f'Generator for {name} not implemented. Set to default nonpara.')
        return DataCreator('nonpara', num_items, path, omega, params)


def normalize_img(img):
    norm_fac = torch.std(img)
    return (img - torch.mean(img)) / norm_fac
    # return img / norm_fac


def masking(cimg, invert=False):
    min_val, max_val, med_val = torch.min(cimg), torch.max(cimg), torch.median(cimg)
    mask, threshold = torch.zeros(cimg.size()), min_val + med_val if not invert else max_val - med_val
    if not invert:
        mask[cimg > threshold] = 1
    else:
        mask[cimg < threshold] = 1
    return mask


def add_noise(img, noise_lvl, invert):
    mask = masking(img, invert=invert)
    if noise_lvl == 0:
        noise_lvl = 1
        raise ZeroDivisionError(f'Noise lvl is {noise_lvl}. Set to 1')
    noise_img = img + torch.normal(0, torch.var(img).item() / noise_lvl, size=img.shape) * mask

    # keep background by cut-off normalization and not normalizing through min/max ratio
    noise_img[noise_img < torch.min(img)] = torch.min(img)
    noise_img[noise_img > torch.max(img)] = torch.max(img)
    return noise_img


def resize_image(list_images, scale):
    # function to enable tiny sets to test performance on low resolution with corresponding grid
    dims = len(list_images[0].shape)
    mode = 'bicubic' if dims == 2 else 'trilinear'
    rsi = [F.interpolate(im.unsqueeze(0).unsqueeze(1), scale_factor=scale, mode=mode, align_corners=True).squeeze() for im in list_images]
    return rsi

