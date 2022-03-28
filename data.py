import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
import h5py
import datetime
from PIL import Image
import warnings
from scipy.interpolate import Rbf
import scipy

import image_registration.reg_utils.tools as tools
from image_registration.reg_utils.grids import Grid, scale_grid, plot_grid_fair, reshape_grid
from image_registration.reg_utils.deformations import set_deformation
from image_registration.reg_utils.interpolation import set_interpolater
from image_registration.reg_utils.visualization import construct_large
from utils.save_results import write_file


class DataCreator:
    def __init__(self, name, num_items, path, omega, params):
        super(DataCreator, self).__init__()
        self.name = name
        self.num_items = num_items
        self.omega = omega
        self.params = params
        self.reduction = params['reduction'] if 'reduction' in params.keys() else 1
        # param to create tiny images for ML test, may later be discarded with resize_image
        self.scale_fac = params['scale_fac'] if 'scale_fac' in params.keys() else 1
        # all images should have same size, so m,h and grid can be set here
        # only temporally load images to get dimensions m and then discard them, do not use as attribute
        self.m = (torch.tensor(tools.load_images(path, '.jpg')[0][0].shape) // self.scale_fac).int()
        self.grid = Grid(1, self.omega, self.m)
        self.h = self.grid.getSpacing()
        self.img_dims = len(self.m)

    # pass images and not make them attribute of generator!
    def create_data(self, path_images, image_pairs=False, verbose=False, invert=False):
        xmin, x0 = [], []
        imgT, imgR = [], []

        # load image data
        images, _ = tools.load_images(path_images, '.jpg')
        if not self.scale_fac == 1:
            # Hack remove, create tiny images with grids in corresponding size for testing
            print('Warning tiny images hack enabled')
            images = resize_image(images, 1 / self.scale_fac)

        if not isinstance(images, list):
            raise TypeError('Images should be passed in list to include different images in one data set')

        # add routine to compose different images to set (i.e learn2reg)
        # for multiple images (pairs) diff_on=True
        if image_pairs:
            # get every second element from list [R0, T0, R1, T1, ...], add T num_items times (as many times as R gets
            # deformed)
            for t in images[1::2]:
                imgT.extend([normalize_img(t.squeeze())] * self.num_items)
                imgT.extend([t.squeeze()] * self.num_items)
            imgSet = images[0::2]
        else:
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
                if not image_pairs:
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
        # should be unique for each type of param creation (affine, nonpara)/ default non para
        reduction = self.reduction
        # ratio_random = int(self.params['ratio'])
        spacing_factor = self.params['spacing_factor']
        xc = self.grid.getCellCentered().clone()
        # xc_t = xc.clone()

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
        # TODO Enable small grids for full size images / gutils.scale_grid(w, m_old, m_new) -  only on start-params
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


class DataCreatorBanana(DataCreator):
    def __init__(self, name, num_items, path, omega, params):
        super(DataCreatorBanana, self).__init__(name, num_items, path, omega, params)

    def create_params(self, fac=2, downscale=False):
        grid = Grid(1, self.omega, self.m)

        xc = grid.getCellCentered()
        xc_t = xc.clone()

        m = min(self.m).item()
        max_reduction = math.ceil(math.log2(m)) - 1
        reduction = min(self.reduction, max_reduction)

        # downscale
        if downscale:
            factor = int(math.pow(2, reduction))
            q = (self.m.float() / factor).int()
            small_grid = scale_grid(xc_t, self.m, q)
        else:
            factor, q, small_grid = 1, self.m, xc_t

        # compute new spacing / m should be enable uneven image dims
        m_new = q.int()
        h = torch.zeros(len(m_new))
        for k in range(0, len(m_new)):
            h[k] = (self.omega[2 * k + 1] - self.omega[2 * k]) / m_new[k].float()

        H, W = q[0].item(), q[1].item()
        dims = q.numel()
        xc_np = reshape_grid(small_grid, q).squeeze().permute(2, 0, 1).detach().numpy()

        th = np.linspace(0, H - 1, H)
        tw = np.linspace(0, W - 1, W)
        XI, YI = np.meshgrid(th, tw, indexing='ij')

        # x_space, y_space = H//nb_points, W//nb_points
        x_fac, y_fac = h[0] * (H // fac), h[1] * (W // fac)
        sign_x, sign_y, which_axis = np.random.choice([-1, 1]), np.random.choice([-1, 1]), np.random.choice([0, 1, 2])
        edge_fac, border_fac = np.random.uniform(0.4, 0.75), np.random.uniform(0.1, 0.2)

        # defined line, not middle
        # x = np.concatenate((np.array([0, 0, H - 1, H - 1]), np.arange(x_space, (H+1)-x_space, x_space)))
        # y = np.concatenate((np.array([0, W - 1, 0, W - 1]), np.arange(y_space, (W+1)-y_space, y_space)))
        # fix corner points, center, and four mid points on grid edges
        x = np.concatenate((np.array([0, 0, H - 1, H - 1]), np.array([H // 2, H // 2, H - 1, 0]), np.array([H // 2])))
        y = np.concatenate((np.array([0, W - 1, 0, W - 1]), np.array([0, W - 1, W // 2, W // 2]), np.array([W // 2])))

        dx_base = xc_np[0][x, y]
        dy_base = xc_np[1][x, y]

        # fix edge points are different for x and y axis
        x_shift = np.array(
            [*4 * [sign_x * -edge_fac * x_fac], *2 * [0], *2 * [sign_x * border_fac * x_fac], *[sign_x * x_fac]])
        y_shift = np.array(
            [*4 * [sign_y * -edge_fac * y_fac], *2 * [sign_y * border_fac * y_fac], *2 * [0], *[sign_y * y_fac]])

        dx = dx_base + x_shift
        dy = dy_base + y_shift

        # interpol for both grid dimensions isolated
        rbfi1 = Rbf(x, y, dx, function='thin_plate')
        di_x = rbfi1(XI, YI)

        # matlab column and row swap compared to python/numpy, may be the reason for swaped axis.
        rbfi2 = Rbf(x, y, dy, function='thin_plate')
        di_y = rbfi2(XI, YI)

        # recreate tensor / switch dim to shift points
        if which_axis == 0:
            # shift x1 grid and keep x2 fix
            x1, x2 = torch.tensor(di_x), torch.tensor(xc_np[1])
        elif which_axis == 1:
            # keep x1 grid fix and shift x2
            x1, x2 = torch.tensor(xc_np[0]), torch.tensor(di_y)
        else:
            # shift x1 and x2 grid
            x1, x2 = torch.tensor(di_x), torch.tensor(di_y)
        xc_gt = torch.stack((x1, x2), dim=0).unsqueeze(0).flatten(-dims, -1)

        # rescale to full size and return params
        xc_gt = scale_grid(xc_gt, q, self.m)
        # NPIR performs xc + wk, create gt by subtracting cell-centered grid
        min_params = (xc_gt - xc).flatten(-len(self.m), -1).squeeze()
        start_params = torch.zeros(xc.flatten(-len(self.m), -1).squeeze().shape, dtype=xc.dtype)

        return min_params, start_params, xc_gt.clone()


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


class DataCreatorIncremental(DataCreator):
    def __init__(self, name, num_items, path, omega, params):
        super(DataCreatorIncremental, self).__init__(name, num_items, path, omega, params)

    def create_data(self, path_images, image_pairs=False, verbose=False, invert=False):
        xmin, x0 = [], []
        imgT, imgR = [], []

        # load image data
        images, _ = tools.load_images(path_images, '.jpg')

        if not isinstance(images, list):
            raise TypeError('Images should be passed in list to include different images in one data set')

        # shift is not given in pixel but as a fraction (width,height)/shift
        ranges = {'angle': np.arange(-self.params['angle'], self.params['angle'], self.params['angle']*2/self.num_items),
                  'shift': np.arange(-self.params['shift'], self.params['shift'], self.params['shift']*2/self.num_items)}

        for cimg in images.copy():
            # run over parameter sets and increment values to get slowly changing deformations instead of random
            # slowly change angle and translation/ avoid shear for now, not in random training
            cimg = normalize_img(cimg)
            for param, ax in zip(['angle', 'shift', 'shift'], [0, 0, 1]):
                self.params['angle'], self.params['shift'], self.params['shear'] = 0, self.m[0], 0

                for ii in ranges[param]:
                    if math.isclose(ii, 0, abs_tol=10**-9) and param == 'shift':
                        # add delta to avoid division with 0 / set to m degree rotation or shift_fac m
                        ii = self.m[0]
                    # set chosen parameter to incremental value
                    self.params[param] = ii
                    min_params, start_params, xc = self.create_params(ax=ax)
                    x0, xmin = [*x0, start_params], [*xmin, min_params]

                    # transform image for supervised learning
                    inter = set_interpolater('linearFAIR', self.omega, self.m, self.h)
                    imgR = [*imgR, inter.interpolate(cimg, xc).squeeze()]
                    if not image_pairs:
                        imgT = [*imgT, add_noise(cimg, self.params['noise_level'], invert)]

                    if verbose:
                        B, C, HW = xc.shape
                        sxc = xc.reshape(B, C, int(math.sqrt(HW)), int(math.sqrt(HW)))
                        plot_grid_fair(sxc, imgT[-1], imgR[-1], self.omega), plt.show()

        # convert list to tensors
        return torch.stack(xmin), torch.stack(x0), torch.stack(imgT), torch.stack(imgR)

    def create_params(self, dtype=torch.double, ax=0):
        dfm = set_deformation('affine', self.m.numel())

        # shift steps in axis (switch ax to change axis)
        shift = (self.m / self.params['shift'] * self.h)
        shift[ax] = 0

        w_top = torch.tensor(math.pi * self.params['angle'] / 180, dtype=dtype)

        # deformation matrix
        w1 = torch.cos(w_top) + self.params['shear']
        w2 = -torch.sin(w_top) + self.params['shear']
        w4 = torch.sin(w_top) + self.params['shear']
        w5 = torch.cos(w_top) + self.params['shear']
        min_params = torch.tensor([w1, w2, w4, w5, shift[0], shift[1]], dtype=dtype)
        start_param = torch.tensor([[1.005, 0], [0, 1.005], [math.sqrt(2) * self.h[0], math.sqrt(2) * self.h[1]]],
                                   dtype=dtype).flatten()

        xc = dfm.evaluate(min_params.unsqueeze(0), self.grid.getCellCentered(), self.omega.unsqueeze(0))
        return min_params, start_param, xc


def init_generator(name, num_items, path, omega, params):
    if name == 'affine':
        return DataCreatorAffine(name, num_items, path, omega, params)
    elif name == 'nonpara':
        return DataCreator(name, num_items, path, omega, params)
    elif name == 'incremental':
        return DataCreatorIncremental(name, num_items, path, omega, params)
    elif name == 'banana':
        return DataCreatorBanana(name, num_items, path, omega, params)
    else:
        print(f'Generator for {name} not implemented. Set to default nonpara.')
        return DataCreator('nonpara', num_items, path, omega, params)


def normalize_img(img):
    norm_fac = torch.std(img)
    return (img - torch.mean(img)) / norm_fac
    # for MIDL22 return to 'insufficient' norm by std
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
    # tools.normalize(img + torch.normal(0, torch.mean(img).item() / params['noise_level'], size=img.shape) * mask)
    if noise_lvl == 0:
        noise_lvl = 1
        raise ZeroDivisionError(f'Noise lvl is {noise_lvl}. Set to 1')
    # mean is zero for new normalization, find new way to create noise in dependence on data
    noise_img = img + torch.normal(0, torch.var(img).item() / noise_lvl, size=img.shape) * mask

    # keep background by cut-off normalization and not normalizing through min/max ratio
    noise_img[noise_img < torch.min(img)] = torch.min(img)
    noise_img[noise_img > torch.max(img)] = torch.max(img)
    return noise_img


def select_params(params, idx):
    new_params = {}
    for k in params.keys():
        if k in ['angle', 'shear', 'shift', 'ratio', 'spacing_factor']:
            new_params[k] = params[k][idx]
        else:
            new_params[k] = params[k]
    return new_params


def create_h5_file(paths, sizes, dir_name, image_pairs=False, save_sample=False, invert=False, **kwargs):
    names = ['train', 'test', 'val']
    omega = kwargs['omega']

    # set seed to reproduce data sets
    np.random.seed(kwargs['seed'])

    dir_name = os.path.join(dir_name, kwargs['type'])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for ii in range(len(names)):
        xmin_sum, x0_sum, Timgs_sum, Rimgs_sum = [], [], [], []
        for idx, p in enumerate(paths):
            sz = sizes[idx]
            c_params = select_params(kwargs, idx)

            # set up data generator
            gen_type = kwargs['type'] if 'type' in kwargs.keys() else 'nonpara'
            data_gen = init_generator(gen_type, sz[ii], p, omega, c_params)

            xmin, x0, Timgs, Rimgs = data_gen.create_data(p, image_pairs=image_pairs, invert=invert)

            xmin_sum = [*xmin_sum, xmin]
            x0_sum = [*x0_sum, x0]
            Timgs_sum = [*Timgs_sum, Timgs]
            Rimgs_sum = [*Rimgs_sum, Rimgs]
        print(f'Save dataset {names[ii]} with {len(x0)} samples to disk')
        with h5py.File(os.path.join(dir_name, names[ii] + '.h5'), 'w') as fh:
            ds = fh.create_dataset('T', data=torch.cat(Timgs_sum, 0), compression=None, fletcher32=True)
            ds = fh.create_dataset('R', data=torch.cat(Rimgs_sum, 0), compression=None, fletcher32=True)
            ds = fh.create_dataset('x*', data=torch.cat(xmin_sum, 0), compression=None, fletcher32=True)
            ds = fh.create_dataset('x0', data=torch.cat(x0_sum, 0), compression=None, fletcher32=True)
            ds.attrs['time_created'] = datetime.datetime.now().isoformat().encode('utf-8')
        fh.close()

    # save configurations
    configs = {}
    c_names = [c for c in kwargs.keys()]
    for jj in range(len(c_names)):
        configs[c_names[jj]] = kwargs[c_names[jj]]
    configs['Sizes'] = sizes
    configs['omega'] = omega.detach().numpy()
    write_file(configs, 'Configs', dir_name)

    # save sample images as inputs for FAIR
    if save_sample:
        # idx = np.random.randint(0, Timgs.shape[0])
        for idx in range(Timgs.shape[0]):
            T = Timgs[idx]
            R = Rimgs[idx]
            imgT = Image.fromarray(T.detach().cpu().numpy()).convert("L")
            imgR = Image.fromarray(R.detach().cpu().numpy()).convert("L")
            imgT.save(os.path.join(dir_name, f'T{idx}.jpg'))
            imgR.save(os.path.join(dir_name, f'R{idx}.jpg'))


def plot_set(path, num_items, num_img, invert=False, image_pairs=False, show_all=False, **params):
    np.random.seed(params['seed'])
    params = select_params(params, 0)

    # reduction for nonpara / reduces image dimensions for global deformations, should be one for small image dims
    gen_type = params['type'] if 'type' in params.keys() else 'nonpara'
    data_gen = init_generator(gen_type, num_items, path, params['omega'], params)
    xmin, x0, T, R = data_gen.create_data(path, verbose=False, image_pairs=image_pairs, invert=invert)

    if data_gen.img_dims == 2:
        plot2d(num_img, T, R, show_all)
    else:
        # Error raised preceding pipeline if image dimensions exceed 3 or are lower than 2
        plot3d(num_img, T, R, show_all)


def plot2d(num_img, T, R, show_all=False):
    list_R, list_T = [img for img in R], [img for img in T]
    try:
        img_all = construct_large(list_R, list_T)
    except RuntimeError:
        print('Number images not divided by 5. Show subset')
        if len(list_R) < 5:
            print(f'Too few samples {len(list_R)}. Show random pair')
            items_idx = np.random.choice(np.arange(0, len(list_R)), size=num_img, replace=False)
            plt.figure(), plt.subplot(121), plt.imshow(list_R[items_idx].detach(), cmap='gray'), plt.axis('off')
            plt.subplot(122), plt.imshow(list_T[items_idx].detach(), cmap='gray'), plt.axis('off')
        else:
            mod_nb = len(list_R)
            while not mod_nb % 5 == 0:
                mod_nb = mod_nb - 1
            img_all = construct_large(list_R[:mod_nb], list_T[:mod_nb])

    if show_all:
        for img in img_all:
            plt.figure()
            plt.imshow(img.detach(), cmap='gray'), plt.axis('off')
    else:
        items_idx = np.random.choice(np.arange(0, R.shape[0]), size=num_img, replace=False)
        for item in items_idx:
            plt.figure()
            plt.imshow(R[item], cmap='gray'), plt.title(f'{item}'), plt.axis('off')
    plt.show()


def plot3d(num_img, T, R, show_all=False):
    # does not work as expected, better plot 2d slices (random slice index)/ use plot2d after slicing
    # a = T[0].detach().numpy()
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # cmap = plt.get_cmap("gray")
    # norm = plt.Normalize(a.min(), a.max())
    # ax.voxels(np.ones_like(a), facecolors=cmap(norm(a)), edgecolor="black")
    #
    # plt.show()
    sliced_T, sliced_R = [], []
    for t, r in zip(T, R):
        # slice_idx = np.random.randint(t.shape[-1])
        slice_idx = t.shape[-1]//2
        sliced_T, sliced_R = [*sliced_T, t[:, :, slice_idx]], [*sliced_R, r[:, :, slice_idx]]
    plot2d(num_img, sliced_T, sliced_R, show_all)


def resize_image(list_images, scale):
    # function to enable tiny sets to test performance on low resolution with corresponding grid
    dims = len(list_images[0].shape)
    mode = 'bicubic' if dims == 2 else 'trilinear'
    rsi = [F.interpolate(im.unsqueeze(0).unsqueeze(1), scale_factor=scale, mode=mode, align_corners=True).squeeze() for im in list_images]
    return rsi


def main():
    # data_paths = [os.path.join('/data/image_registration', d) for d in ['Hands', 'MRI', 'learn2Reg/thorax_train',
    # 'fastMRI/images/test']]
    data_paths = [os.path.join('/data/image_registration', d) for d in ['learn2Reg/task3']]
    out_dir = '/data/image_registration/learn2Reg/task3'

    seed = 67
    # TODO 3d npir deformation seems to inflate image sizes, find reason (scale_grid?)/ trilinear interpolation
    # has strong smoothing effect
    # num_items = [[300, 25, 25], [300, 25, 25], [2, 1, 1]]
    num_items = [[1, 1, 1]]
    omega = torch.tensor([-1, 1, -1, 1, -1, 1], dtype=torch.int32)
    show_examples, diff_on, invert = False, False, False
    # bool to determine if image pair is used as (T,R; i.e. learn2reg) or one image but deformed grid
    params = {'seed': seed,
              'type': 'nonpara',
              'angle': [60],
              'shear': [0.12],
              'shift': [10],
              'ratio': [4],
              'spacing_factor': [0.2],
              'omega': omega,
              'noise_level': 100,
              'scale_fac': 4,
              'reduction': 0}

    if show_examples:
        plot_set(data_paths[0], 1, num_img=5, invert=invert, image_pairs=diff_on, show_all=True, **params)
    else:
        create_h5_file(data_paths, num_items, out_dir, image_pairs=diff_on, save_sample=False, invert=invert, **params)


if __name__ == '__main__':
    main()
