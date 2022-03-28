import torch
import torch.nn.functional as F
import numpy as np
# import time
# from scipy import sparse as sp
import matplotlib.pyplot as plt
import math

import image_registration.reg_utils


class Grid:
    def __init__(self, batchsize, omega=None, m=None, dtype=torch.double):
        self.bs = batchsize
        self._omega = omega
        self._m = m
        self.dim = int(len(self._omega)/2) if self._omega is not None else None
        self.dtype = dtype

    @property
    def omega(self):
        return self._omega
    @omega.setter
    def omega(self, v):
        self._omega = v

    @property
    def m(self):
        return self._m
    @m.setter
    def m(self, v):
        self._m = v

    def getSpacing(self):
        omega = self._omega
        m = self._m
        h = torch.zeros(len(m))
        for k in range(0, len(m)):
            h[k] = (omega[2*k+1] - omega[2*k]) / m[k].type(self.dtype)

        return h.type(self.dtype)

    def getCellCentered(self):
        omega = self._omega
        m = self._m
        dim = int(len(omega) / 2)
        h = Grid.getSpacing(self)

        x = np.zeros(np.append(dim, m))
        xi = lambda i: np.linspace(omega[2*i-2]+h[i-1]/2, omega[2*i-1]-h[i-1]/2, m[i-1])
        if dim == 1:
            x = xi(1)
        elif dim == 2:
            x = np.meshgrid(xi(1), xi(2), indexing='ij')
        elif dim == 3:
            x = np.meshgrid(xi(1), xi(2), xi(3), indexing='ij')
        elif dim == 4:
            x = np.meshgrid(xi(1), xi(2), xi(3), xi(4), indexing='ij')

        x = torch.tensor(x, dtype=self.dtype)
        # grid = x.unsqueeze(0).repeat(self.bs, 1, 1, 1)
        grid = x.unsqueeze(0).repeat([self.bs, *(dim+1)*[1]])
        grid_out = grid.flatten(-dim, -1)

        return grid_out

    def torchGrid(self, input, field):
        omega = self._omega
        m = self._m

        B = input.size()[0]
        mtx = torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1)

        return F.affine_grid(mtx, field.size())

    # adapt to torch and use own interpolation
    def __prolong_restrict_image(self, data, batchsize, factor, flag, method):
        omega = self._omega
        m = self._m
        if flag == 'restrict':
            q = m / factor
            scale_factor = 1 / factor
        elif flag == 'prolong':
            q = factor * m
            scale_factor = factor
        q = q.type(torch.float64)
        m_new = torch.round(q).type(torch.int64)

        # something is odd with the downsampling options, focus not in center try torch build in alternative
        # add batch and channel dimensions if only tensor of shape heigth x width is given
        # if len shape = 2, only image dims given and batch and channel dummy has to be added
        if len(data.shape) == 2:
            # add batch and channel dummy dim
            form_data = data.unsqueeze(0).unsqueeze(1)
        else:
            # batch given, only add channel dummy dim
            form_data = data.unsqueeze(1)

        sampler = F.interpolate(form_data, scale_factor=scale_factor, mode='bicubic', align_corners=True)

        if len(data.shape) == 2:
            data_new = sampler.squeeze(1).squeeze(0)
        else:
            data_new = sampler.squeeze(1)

        return data_new, m_new

    def restrictImage(self, data, batchsize, factor=2, method='linearFAIR'):
        data_res, m_res = Grid.__prolong_restrict_image(self, data, batchsize, factor, 'restrict', method)
        return data_res, m_res


def reshape_grid(xc, m):
    # function to reconstruct flatten grid for plotting purposes
    # grids are flatten using torch build-in function, not fortan_flatten -> keep track for uneven x,y
    # works fine for quadratic grids, but may produce wrong results for uneven height and width
    # Could be alright though since flatten is used in x and y independently so using reshape here should be fine
    # compare to baseline_tests interpolation
    # m = m.type(torch.int)
    m = m.int()
    if len(xc.size()) < 2:
        # if single evaluation pretend to be batch to deal with batch later in same function
        xc = xc.unsqueeze(0)

    # reshape grid like in interpolation -> better in dfm
    xc = xc.permute((0, 2, 1))

    # use reshape in matrix indexing=ij style, m defines image dimensions (dynamic for 2d and 3d)
    dim = m.numel()
    new_shape = [xc.shape[0], *[m[i].item() for i in range(m.numel())], dim]
    xc_reshaped = xc.reshape(new_shape)

    # old version works for deform(x0, self.gridRef, self.omega).flatten(1, 2) without permute
    # N = int(len(xc[0]) / dim)
    # dim_range = torch.arange(0, len(xc[0]), N)
    # xc_resh = torch.zeros((xc.shape[0], m[0], m[1], dim), dtype=xc.dtype)
    #
    # for ii in range(dim):
    #     indx = dim_range[ii]
    #     xc_resh[:, :, :, ii] = xc[:, indx:indx + N].reshape((xc.shape[0], m[0], m[1]))

    return xc_reshaped


def prolong_grid(xc_diff, xc_Ref, m_diff, m_Ref):
    # output of network is flattend
    dims = m_diff.numel()
    B, CHW = xc_diff.shape
    xc_diff = xc_diff.reshape((B, dims, CHW // dims))

    # update transformation to last iterate wc of previous level
    # TODO check if flatten is dim free
    xc_diff = scale_grid(xc_diff, m_diff, m_Ref)
    return (xc_Ref + xc_diff).flatten(1, 2)


def scale_grid(xc, m, new_m):
    dims = m.numel()
    mode = 'bilinear' if dims == 2 else 'trilinear'
    unfold_xc = reshape_grid(xc, m).permute((0, dims + 1, *[d for d in np.arange(1, dims + 1)]))
    down_xc = F.interpolate(unfold_xc, tuple(new_m.tolist()), mode=mode, align_corners=True)
    return down_xc.flatten(-dims, -1)


def scale_to_gridres(xc, xcRef, m_xc, m_xcRef):
    dims = m_xc.numel()
    B, CHW = xc.shape
    xc = xc.reshape((B, dims, CHW // dims))

    xc_scaled = scale_grid(xc, m_xc, m_xcRef)
    # short sanity check if new grid dims are as aspected
    if not xc_scaled.shape == xcRef.shape:
        raise RuntimeError(f'Interpolated grid and reference grid do not have same size')

    return xc_scaled.flatten(1, 2)


# helper functions Alessa, Sven
def displacementDenormalization(u):
    # pytorch grid to voxel grid
    B, C, D, H, W = u.size()
    scale = torch.ones(u.size())
    # only height and width given, so D can stay scale 1 / 3 Channel for movement in x, y, z (volume)
    # scale[:, 0, :, :, :] = scale[:, 0, :, :, :] * (D - 1) / 2
    scale[:, 0, :, :, :] = scale[:, 0, :, :, :] * (H - 1) / 2
    scale[:, 1, :, :, :] = scale[:, 1, :, :, :] * (W - 1) / 2

    if u.is_cuda:
        return u*scale.cuda()
    else:
        return u*scale


# pass grid directly in u, since I have no displacement yet (testing purposes) added to Alessa / Sven implementation
def plotGrid(u, h=None, image=None, omega=None, stride=2, alpha=0.8, cmap='gray', lineColor='darkmagenta', lineWidth=0.5):
    '''
    Plot the grid of a 3D displacement field u (normalized on [-1, 1]) in 2D. Overlay with an image (e.g. the moving
    image or the determinant of the jacobian) possible. Image axis are swapped/flipped for proper visualization.

    input:
        u:          normalized displacement field, Bx3xDxHxW (required, only first displacement field of batch is used)
        h:          voxel size, Bx3 (default 1)
        image:      optional image on which the grid will be plotted on, Bx1xDxHxW
        axis:       axis from which the plotted slice is taken, 'axial', 'sagittal' or 'coronal', default 'axial'
        stride:     stride for plotting the grid lines, default 2
        slice_idx:  index of the slice on the chosen axis that is plotted, default center slice
        alpha:      opacity of the overlay image, dafault 0.8
        cmap:       colarmap of the overlay image, default 'gray' (e.g. use 'hot' for detJ)
        lineColor   color of the grid lines, default dark magenta
        lineWidth   width of the grid lines, default 0.5

    return:
        ---

    use (example):
        u = net(moving, fixed)
        detJ = jacobian(u)
        plotGrid(u, h, image=detJ, cmap='hot', lineColor='k')
        plt.colorbar()
        plt.axis('off')
        plt.show()
    '''
    # same structure used by pytroch, so that my C should be 2 (x, y displacement) and B should be batchSize,
    # D = 1 no volumes
    # or try set dummy dimension for depth, watch out for correct axis (should not be in depth orientation), sagittal?
    B, C, D, H, W = u.size()

    # h is given in my functions, pass it (also no volumes, so (B, 2))
    if h is None:
        h = torch.ones(B, 3)

    if len(h.size()) != 2:
        raise IndexError('h must have size Bx3')

    # construct grid with deformation in u (should be non existing for my example) Y = torchGrid
    # (from image_registration package)
    Y = u
    Y = displacementDenormalization(Y).permute(0, 2, 3, 4, 1)
    Y = Y.cpu()

    # add own slicing so that grid is shown for height and width
    h1 = h[0, 0]
    h2 = h[0, 1]
    Y1 = Y[0, 0, :, :, 0] * h1
    Y2 = Y[0, 0, :, :, 1] * h2

    s1 = Y1.size(1)
    s2 = Y1.size(0)
    if omega is None:
        s1_half = (s1 * h1 - 1) / 2
        s2_half = (s2*h2 - 1) / 2
        extend = [-s1_half, s1_half, -s2_half, s2_half]
    else:
        extend = omega

    I1 = torch.tensor(range(0, s1, stride)).numpy()
    I2 = torch.tensor(range(0, s2, stride)).numpy()

    Y1 = Y1.numpy()
    Y2 = Y2.numpy()

    plt.plot(Y1[:, I1], Y2[:, I1], Y1[I2, :].transpose(), Y2[I2, :].transpose(), color=lineColor,
             linewidth=lineWidth)
    plt.axis('equal')
    if not (image is None):
        plt.imshow(image.detach().numpy(), extent=extend, alpha=alpha, cmap=cmap)


def plot_grid_fair(xc, image, p, omega, stride=1, alpha=0.8, cmap='gray', lineColor='magenta', lineWidth=0.5):
    # 'darkmagenta'
    Y = xc.cpu()

    # add own slicing so that grid is shown for height and width
    # Do not scale our grids, since we do not get displacement field u but grid + displacement
    Y1 = Y[0, 0, :, :]
    Y2 = Y[0, 1, :, :]

    I1 = torch.tensor(range(0, Y1.size(1), stride))
    I2 = torch.tensor(range(0, Y1.size(0), stride))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(Y1[:, I1], Y2[:, I1], Y1[I2, :].transpose(0, 1), Y2[I2, :].transpose(0, 1), color=lineColor,
             linewidth=lineWidth)
    ax.axis('equal')
    ax.axis('off')
    if not (image is None):
        ax.imshow(p.detach().numpy(), extent=omega, alpha=alpha, cmap=cmap)
        # plt.imshow(image.detach().numpy(), extent=omega, alpha=alpha, cmap=cmap)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image.detach().numpy(), extent=omega, alpha=alpha, cmap=cmap)
    ax2.axis('equal')
    ax2.axis('off')

    return fig
