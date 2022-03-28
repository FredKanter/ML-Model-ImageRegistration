import torch
import warnings
import torch.nn.functional as F


class Interpolation:
    def __init__(self, omega=torch.tensor([]), m=torch.tensor([]), h=None, dtype=torch.double):
        super(Interpolation, self).__init__()
        self._omega = omega
        self.dim = int(len(omega) / 2)
        self.m = m
        self.h = h
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

    def check_inter(self, xi):
        if torch.sum(xi == 0) != 0:
            indx = (xi == 0).nonzero()
            warnings.warn('Interpolation on cell borders or centers')

    def get_coefficients(self, image):
        # gets image and returns coeff for interpolate function
        return image

    def interpolate(self, coeff, xc):
        # gets coeff and grid and returns interpolated image
        return xc


class LinearInterpolation(Interpolation):
    def __init__(self, omega=torch.tensor([]), m=torch.tensor([]), h=None):
        super(LinearInterpolation, self).__init__(omega, m, h)
        self.name = 'linear'

    def get_coefficients(self, image):
        return image

    @staticmethod
    def get_valid_ids(p, m):
        valid = torch.ones_like(p[..., -1], dtype=bool)
        for ii in range(len(m)):
            valid = valid & (0 <= p[..., ii]) & (p[..., ii] < m[ii] + 1)
        return torch.nonzero(valid)

    @staticmethod
    def inter2(tc, tp, p, ids, xi):
        p = p[ids[:, 0], ids[:, 1], ids[:, 2], :]
        xi = xi[ids[:, 0], ids[:, 1], ids[:, 2], :]
        # principle flow of 2d interpolation
        xd1 = tp[ids[:, 0], p[:, 0], p[:, 1]] * (1 - xi[:, 0]) + tp[ids[:, 0], (p[:, 0] + 1), p[:, 1]] * (xi[:, 0])
        xd2 = tp[ids[:, 0], (p[:, 0]), (p[:, 1] + 1)] * (1 - xi[:, 0]) + tp[ids[:, 0], (p[:, 0] + 1), (p[:, 1] + 1)] * (
        xi[:, 0])
        tc[ids[:, 0], ids[:, 1], ids[:, 2]] = xd1 * (1 - xi[:, 1]) + xd2 * (xi[:, 1])
        return tc

    @staticmethod
    def inter3(tc, tp, p, ids, xi):
        p = p[ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3], :]
        xi = xi[ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3], :]
        # principle flow of 3d interpolation
        xd1 = tp[ids[:, 0], p[:, 0], p[:, 1], p[:, 2]] * (1 - xi[:, 0]) + tp[
            ids[:, 0], (p[:, 0] + 1), p[:, 1], p[:, 2]] * (xi[:, 0])
        xd2 = tp[ids[:, 0], p[:, 0], (p[:, 1] + 1), p[:, 2]] * (1 - xi[:, 0]) + tp[
            ids[:, 0], (p[:, 0] + 1), (p[:, 1] + 1), p[:, 2]] * (xi[:, 0])
        xd3 = tp[ids[:, 0], p[:, 0], p[:, 1], (p[:, 2] + 1)] * (1 - xi[:, 0]) + tp[
            ids[:, 0], (p[:, 0] + 1), p[:, 1], (p[:, 2] + 1)] * (xi[:, 0])
        xd4 = tp[ids[:, 0], p[:, 0], (p[:, 1] + 1), (p[:, 2] + 1)] * (1 - xi[:, 0]) + tp[
            ids[:, 0], (p[:, 0] + 1), (p[:, 1] + 1), (p[:, 2] + 1)] * (xi[:, 0])

        yd1 = xd1 * (1 - xi[:, 1]) + xd2 * (xi[:, 1])
        yd2 = xd3 * (1 - xi[:, 1]) + xd4 * (xi[:, 1])

        tc[ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3]] = yd1 * (1 - xi[:, 2]) + yd2 * (xi[:, 2])
        return tc

    def interpolate(self, t, xc):
        t = self.get_coefficients(t)
        omega = self._omega.to(t.device)
        dim = self.dim
        m = self._m.type(torch.int).to(t.device)
        h = self.h.to(t.device)
        dtype = self.dtype

        if self.dim not in [2, 3]:
            raise RuntimeError(f'Linear Interpolation for dim {dim} not implemented.')
        inter = self.inter2 if dim == 2 else self.inter3

        if len(t.size()) < dim + 1:
            # if size of image of image lacks batch dim, create dummy dim
            t = t.unsqueeze(0)

        batch_size = t.size(0)
        my_x = xc.view(batch_size, dim, -1).permute((0, 2, 1)).clone()
        my_x = (my_x - omega.view(dim, 2)[:, 0]) / h + 0.5
        my_x = my_x.view([batch_size, *[i.item() for i in m], dim])
        my_p = torch.floor(my_x).type(torch.long)
        my_xi = my_x - my_p

        # padding
        my_tp = F.pad(t.clone(), [*len(m) * [1, 1]], value=torch.min(t))

        valid_ids = self.get_valid_ids(my_p, m)

        my_tc = torch.zeros_like(t, device=my_x.device, dtype=dtype) + torch.min(t)
        my_tc = inter(my_tc, my_tp, my_p, valid_ids, my_xi)

        return my_tc


def set_interpolater(method, omega, m, h):
    if method == 'linearFAIR':
        interpolator = LinearInterpolation(omega, m, h)
    else:
        raise NameError(f'Method {method} not implemented')

    return interpolator

