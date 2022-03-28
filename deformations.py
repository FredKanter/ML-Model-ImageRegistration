import torch
import numpy as np

import image_registration.reg_utils.grids as gutils
import utils.batch_calculation as bc
import itertools
import math

from image_registration.reg_utils import tools


class Deformation:
    def __init__(self, img_dim):
        super(Deformation, self).__init__()
        self.name = 'identity'
        self.img_dim = img_dim

    @staticmethod
    def evaluate(w, xc, omega):
        return xc


class AffineDeformFAIR(Deformation):
    def __init__(self, img_dim):
        super(AffineDeformFAIR, self).__init__(img_dim)
        self.name = 'affineFAIR'

    @staticmethod
    def evaluate(w, xc, omega):
        if len(xc.size()) < 2:
            # if single evaluation pretend to be batch to deal with batch later in same function
            xc = xc.unsqueeze(0)
        if len(w.size()) < 2:
            w = w.unsqueeze(0)

        phis = []
        for ii in range(xc.shape[0]):
            tmp_xc = xc[ii]
            tmp_w = w[ii]
            if len(tmp_w) < 6:
                raise ValueError(f'Not enough parameters w')

            # in FAIR xc has two dims? calculate N and get x,y components /
            # for more than 2D, change 2 to passable parameter
            N = int(len(tmp_xc) / 2)
            tmp_xc = torch.stack((tmp_xc[0:N], tmp_xc[N::]), dim=1)

            base_Q = torch.eye(2)
            shifts = torch.ones((N, 1), dtype=tmp_xc.dtype)
            qx = torch.cat((tmp_xc, shifts), dim=1)
            Q = kronecker_product(base_Q, qx)

            tmp_w = tmp_w.type(Q.dtype)
            phis.append(torch.matmul(Q, tmp_w))

        return torch.stack(phis, dim=0)


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


class RigidDeform(Deformation):
    def __init__(self, img_dim):
        super(RigidDeform, self).__init__(img_dim)
        self.name = 'rigid'

    @staticmethod
    def evaluate(w, xc, omega):
        if len(omega.size()) < 2:
            omega = omega.unsqueeze(0)

        if w.shape[-1] < 3:
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
        A = torch.stack((torch.stack((torch.cos(w[:, 0]), -torch.sin(w[:, 0])), dim=-1),
                         torch.stack((torch.sin(w[:, 0]), torch.cos(w[:, 0])), dim=-1)), dim=-1)
        b = w[:, 1::].unsqueeze(2).repeat(1, 1, dims[-1])

        mat_phi = torch.matmul(A, xc)

        mat_phi += ctmp + b

        return mat_phi


class NonPara(Deformation):
    def __init__(self, img_dim):
        super(NonPara, self).__init__(img_dim)
        self.name = 'nonpara'

    def evaluate(self, w, xc, omega):
        # should be similar to prolong grid or FAIR mfPu
        # if no deformation chosen, set default to id mapping
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

        # phi is set to output of network (so not u but phi?)
        phi = gutils.scale_grid(w, m_old, m_new) if not torch.all(torch.eq(m_old, m_new)) else w

        # xOld = xc + mfPu(yc - xOld,omega,m/2); -> prolong grid in multilevel ! / how to start with xc=0 and how to
        # implement nonpara trafo (splineInter in Matlab)
        phi = xc + phi
        return phi


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    https://discuss.pytorch.org/t/kronecker-product/3919/4
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, t2_height, t2_width, 1)
            .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def RotationCenterOfDomain(alpha, xc, omega):
    # xc auf das folgende Format bringen: 2 x N_1*N_2, d.h. erste Zeile enthält
    # x-Komponenten, zweite Zeile enthält y-Komponenten
    # flatten of xc since it should have batch-python form
    omega = np.array(omega)
    N = int(len(xc) / 2)
    xc = torch.stack((xc[0:N], xc[N::]))
    c = torch.tensor([(omega[ii + 1] + omega[ii]) / 2 for ii in np.linspace(0, len(omega)-2, num=int(len(omega)/2),
                                                                          dtype=np.int)])

    # Zentrum des Gitters in den Nullpunkt verschieben
    xc[0, :] = xc[0, :] - c[0]
    xc[1, :] = xc[1, :] - c[1]

    # Rotationsmatrix aufbauen
    # A = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)];
    A = torch.stack((torch.stack((torch.cos(alpha), -torch.sin(alpha))),
                    torch.stack((torch.sin(alpha), torch.cos(alpha)))))

    # Drehung ausfuehren
    phi = torch.matmul(A, xc.type(A.dtype))

    # Punkte zurueckverschieben
    phi[0, :] = phi[0, :] + c[0]
    phi[1, :] = phi[1, :] + c[1]

    # phi als Vektor zurueckgeben
    phi_out = torch.cat((phi[0, :], phi[1, :]), dim=0)

    return phi_out


def set_deformation(name, img_dim):
    if not isinstance(name, str):
        raise TypeError(f'{name} should be of type str')

    if not (name in ['rigid', 'affine', 'affineFAIR', 'nonpara']):
        raise NameError(f'Method  {name} not implemented')

    if name == 'rigid':
        return RigidDeform(img_dim)
    elif name == 'affine':
        return AffineDeform(img_dim)
    elif name == 'affineFAIR':
        return AffineDeformFAIR(img_dim)
    else:
        return NonPara(img_dim)

