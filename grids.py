import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
        grid = x.unsqueeze(0).repeat([self.bs, *(dim+1)*[1]])
        grid_out = grid.flatten(-dim, -1)

        return grid_out

    # adapt to torch and use own interpolation
    def __prolong_restrict_image(self, data, batchsize, factor, flag, method):
        m = self._m
        if flag == 'restrict':
            q = m / factor
            scale_factor = 1 / factor
        elif flag == 'prolong':
            q = factor * m
            scale_factor = factor
        q = q.type(torch.float64)
        m_new = torch.round(q).type(torch.int64)

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
    m = m.int()
    if len(xc.size()) < 2:
        # if single evaluation pretend to be batch to deal with batch later in same function
        xc = xc.unsqueeze(0)

    xc = xc.permute((0, 2, 1))

    # use reshape in matrix indexing=ij style, m defines image dimensions
    dim = m.numel()
    new_shape = [xc.shape[0], *[m[i].item() for i in range(m.numel())], dim]
    xc_reshaped = xc.reshape(new_shape)

    return xc_reshaped


def prolong_grid(xc_diff, xc_Ref, m_diff, m_Ref):
    # output of network is flattend
    dims = m_diff.numel()
    B, CHW = xc_diff.shape
    xc_diff = xc_diff.reshape((B, dims, CHW // dims))
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


def plot_grid_fair(xc, image, p, omega, stride=1, alpha=0.8, cmap='gray', lineColor='magenta', lineWidth=0.5):
    Y = xc.cpu()

    # add own slicing so that grid is shown for height and width
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
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image.detach().numpy(), extent=omega, alpha=alpha, cmap=cmap)
    ax2.axis('equal')
    ax2.axis('off')

    return fig
