import torch
import numpy as np
import itertools
import math

import grids as gutils
import batch_calculation as bc
import tools


class Deformation:
    def __init__(self, img_dim):
        super(Deformation, self).__init__()
        self.name = 'identity'
        self.img_dim = img_dim

    @staticmethod
    def evaluate(w, xc, omega):
        return xc


class AffineDeform(Deformation):
    def __init__(self, img_dim):
        super(AffineDeform, self).__init__(img_dim)
        self.name = 'affine'

    @staticmethod
    def evaluate(w, xc, omega):
        if len(omega.size()) < 2:
            omega = omega.unsqueeze(0)

        if w.shape[-1] < 6:
            raise ValueError(f'Not enough parameters w')

        if len(xc.size()) < 2:
            # if single evaluation pretend to be batch to deal with batch later in same function
            xc = xc.unsqueeze(0)
        if len(w.size()) < 2:
            w = w.unsqueeze(0)

        # only one omega
        omega = omega[0]
        dims = xc.shape
        c = torch.tensor([(omega[jj + 1] + omega[jj]) / 2.0
                          for jj in np.linspace(0, len(omega) - 2, num=int(len(omega) / 2), dtype=np.int)])

        # Shift grid center to origin
        ctmp = c.unsqueeze(1).repeat(dims[0], 1, dims[-1]).clone().to(xc.device)
        xc -= ctmp

        # define affine deformation using A and b , w should consists of [A;b]
        A = torch.stack((w[:, 0:2], w[:, 2:4]), dim=-1)
        b = w[:, 4::].unsqueeze(2).repeat(1, 1, dims[-1])

        mat_phi = torch.matmul(A, xc)

        mat_phi += ctmp + b

        return mat_phi


class NonPara(Deformation):
    def __init__(self, img_dim):
        super(NonPara, self).__init__(img_dim)
        self.name = 'nonpara'

    def evaluate(self, w, xc, omega):
        dims = w.shape
        if len(dims) == 2:
            # output from network may be flattened
            w = w.reshape((dims[0], self.img_dim, dims[-1] // self.img_dim))
        # implemented for consistency reasons (deformation passed in net routine)
        # upsample grid if smaller than image grid
        B, C, HW = w.shape
        _, _, HW_ref = xc.shape
        m_old = torch.tensor([int(np.ceil(HW_ref**(1 / C))) for _ in range(C)], dtype=torch.int)
        m_new = torch.tensor([int(np.ceil(HW**(1 / C))) for _ in range(C)], dtype=torch.int)

        phi = gutils.scale_grid(w, m_old, m_new) if not torch.all(torch.eq(m_old, m_new)) else w

        phi = xc + phi
        return phi


def set_deformation(name, img_dim):
    if not isinstance(name, str):
        raise TypeError(f'{name} should be of type str')

    if not (name in ['affine', 'nonpara']):
        raise NameError(f'Method  {name} not implemented')

    elif name == 'affine':
        return AffineDeform(img_dim)
    else:
        return NonPara(img_dim)

