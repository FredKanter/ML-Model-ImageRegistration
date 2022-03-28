import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import itertools
from image_registration.reg_utils.regularizer import Regularizer
from image_registration.reg_utils.interpolation import set_interpolater
import image_registration.reg_utils.tools as tools
import utils.batch_calculation as bc
from image_registration.reg_utils.derivatives import first_order_derivative_conv


class DistanceFunction:
    def __init__(self, deformation, distance, omega, m, h, gridRef):
        super(DistanceFunction, self).__init__()
        self.trafo = deformation.evaluate
        self.dist = distance
        self.omega = omega
        self.gridRef = gridRef
        self.h = h
        self.m = m
        self.inter = set_interpolater('linearFAIR', omega, m, h)

    def evaluate(self, w, **kwargs):
        return None

    @staticmethod
    def _repeat(ex, bs):
        return list(itertools.chain.from_iterable(itertools.repeat(x, bs) for x in [ex]))

    def __copy__(self):
        return DistanceFunction(self.trafo, self.dist, self.omega, self.m, self.h, self.gridRef)


# difference in PIR and NPIR in reshape could be renamed and merged
# difference in NPIR and PIR is defined in deformation
class PIR(DistanceFunction):
    def __init__(self, deformation, distance, omega, m, h, gridRef):
        super(PIR, self).__init__(deformation, distance, omega, m, h, gridRef)
        self.name = 'PIR'

    def evaluate(self, w, **kwargs):
        T = kwargs['T']
        R = kwargs['R']
        dims = self.m.numel()
        # always start from regular grid, should stay the same, so first initialization should work fine
        # same should apply to spacing h
        xcs = self.gridRef.clone()
        h = self.h
        deformation = self.trafo

        # implement batch variant (loops in reg utils functions)
        if len(T.size()) == dims + 1:
            # m = torch.tensor(T[0].shape)
            omega = self.omega.repeat(T.shape[0], 1)
            xcs = xcs.squeeze().repeat(T.shape[0], 1, 1)
            deformation = self._repeat(deformation, T.shape[0])
        else:
            # bc.batch_apply works batch-wise and needs batch dummy dim for non batched samples
            # m = torch.tensor(T.shape)
            deformation = [deformation]
            omega = self.omega.unsqueeze(0)
            w = w.unsqueeze(0)
            T = T.unsqueeze(0)
            R = R.unsqueeze(0)

        xcs = xcs.to(R.device)
        omega = omega.to(R.device)
        h = h.to(R.device)

        rot_xc = bc.batch_apply(deformation, w, xcs.shape, xcs, omega)
        # inter = set_interpolater(method='linearFAIR', omega=omega[0], m=m, h=h)
        Ty = self.inter.interpolate(T, rot_xc)

        # cuda support shift Ty and h to device
        Ty = Ty.to(R.device)

        return self.dist(Ty, R, h)

    def __copy__(self):
        return PIR(self.trafo, self.dist, self.omega, self.m, self.h, self.gridRef)


class NPIR(DistanceFunction):
    def __init__(self, deformation, distance, omega, m, h, gridRef):
        super(NPIR, self).__init__(deformation, distance, omega, m, h, gridRef)
        self.name = 'NPIR'

    def evaluate(self, w, **kwargs):
        # xc has to be passed flattend (fit LSTM), reshape consistent with flatten operation
        dims = self.m.numel()
        if len(w.shape) == 1:
            w = w.unsqueeze(0)
        B, CHW = w.shape
        w = w.reshape((B, dims, CHW // dims))

        T = kwargs['T']
        R = kwargs['R']
        h = self.h
        xcs = self.gridRef.clone()
        deformation = self.trafo

        if len(T.size()) == dims + 1:
            # m = torch.tensor(T[0].shape)
            xcs = xcs.squeeze().repeat(B, 1, 1)
            omega = self.omega.repeat(B, 1)
            deformation = self._repeat(deformation, T.shape[0])
        else:
            # m = torch.tensor(T.shape)
            deformation = [deformation]
            omega = self.omega.unsqueeze(0)
            T = T.unsqueeze(0)
            R = R.unsqueeze(0)

        xcs = xcs.to(R.device)
        omega = omega.to(R.device)
        h = h.to(R.device)
        rot_xc = bc.batch_apply(deformation, w, xcs.shape, xcs, omega)

        # inter = set_interpolater('linearFAIR', omega[0], m, h)
        Ty = self.inter.interpolate(T, rot_xc)

        # cuda support shift Ty and h to device
        Ty = Ty.to(R.device)

        return self.dist(Ty, R, h)

    def __copy__(self):
        return NPIR(self.trafo, self.dist, self.omega, self.m, self.h, self.gridRef)


def set_distance(name, deformation, distance, omega, m, h, gridRef):
    if name == 'PIR':
        return PIR(deformation, distance, omega, m, h, gridRef)
    elif name == 'NPIR':
        return NPIR(deformation, distance, omega, m, h, gridRef)
    else:
        raise RuntimeError(f'Objective of type {name} not implemented')


def SSD(T, R, h):
    dims = h.numel()
    T_flat = T.flatten(-dims, -1)
    R_flat = R.flatten(-dims, -1)

    # T and R are batched and result in batched distance
    D = 0.5 * torch.prod(h, dim=-1) * torch.sum((T_flat - R_flat)**2, dim=-1)
    return D


def NGF(T, R, h, eps=1e1, eps_numerator=False):
    # m = torch.tensor(T.shape[0])
    if len(T.shape) <= 3:
        T = T.unsqueeze(1)
        R = R.unsqueeze(1)
    # calculate image dims, -2 for image and channel dims
    dims = len(T.shape) - 2
    B, C, H, W = T.size()

    # conv_h = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(2, 1), padding=(0, 0), bias=False, groups=C)
    # conv_w = torch.nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 2), padding=(0, 0), bias=False, groups=C)
    # conv_h.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 2, 1).repeat(C, 1, 1, 1).type(T.dtype).to(T.device)
    # conv_w.weight.data = torch.Tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(C, 1, 1, 1).type(T.dtype).to(T.device)
    #
    dT_h, dT_w = first_order_derivative_conv(T, h)
    dR_h, dR_w = first_order_derivative_conv(R, h)
    #
    # dot_h = conv_h(F.pad((dT_h * dR_h), (0, 0, 0, 1), mode='replicate'))
    # dot_w = conv_w(F.pad((dT_w * dR_w), (0, 1, 0, 0), mode='replicate'))
    #
    # dot = dot_h + dot_w
    #
    # if eps_numerator:
    #     dot += eps**2

    hd = torch.prod(h, dim=-1)

    # norm_dT_h = conv_h(F.pad(dT_h ** 2, (0, 0, 0, 1), mode='replicate'))
    # norm_dT_w = conv_w(F.pad(dT_w ** 2, (0, 1, 0, 0), mode='replicate'))
    #
    # norm_dR_h = conv_h(F.pad(dR_h ** 2, (0, 0, 0, 1), mode='replicate'))
    # norm_dR_w = conv_w(F.pad(dR_w ** 2, (0, 1, 0, 0), mode='replicate'))
    #
    # # sqrt on norm?
    # norm_dT = norm_dT_h + norm_dT_w + eps ** 2
    # norm_dR = norm_dR_h + norm_dR_w + eps ** 2
    # ngf = (1 - dot ** 2 / (norm_dT * norm_dR))
    #
    # D2 = hd * torch.sum(ngf.flatten(-3, -1), dim=-1)

    # try to implement FAIR version using matrix multiplication - does not scale correctly, there is a bug (FRED)
    # gradT = torch.cat((dT_h, dT_w), axis=1).flatten(-3, -1)
    # gradR = torch.cat((dR_h, dR_w), axis=1).flatten(-3, -1)
    gradT = torch.transpose(torch.cat((dT_h, dT_w), axis=1).view(B, C, H*W*dims), -2, -1)
    gradR = torch.transpose(torch.cat((dR_h, dR_w), axis=1).view(B, C, H*W*dims), -2, -1)

    rc = torch.matmul(torch.transpose(gradT, -2, -1), gradR).squeeze(1) /\
         (torch.sqrt(torch.sum(gradT**2 + eps**2, dim=-2)) * torch.sqrt(torch.sum(gradR**2 + eps**2, dim=-2)))
    D = hd * (1-rc**2).squeeze(-1)

    # try to implement in GReAT fashion (KAI)
    # lengthGT = torch.sqrt(torch.sum(gradT * gradT, 1) + eps ** 2)
    # lengthGR = torch.sqrt(torch.sum(gradR * gradR, 1) + eps ** 2)
    # r1 = torch.sum(gradR * gradT, 1)
    # r2 = 1 / (lengthGT * lengthGR)
    # rc = r1 * r2
    # D3 = hd*torch.prod(m) - hd * (rc.T @ rc)

    return D


def l2(T, R, h):
    dims = h.numel()
    T_flat = T.flatten(-dims, -1)
    R_flat = R.flatten(-dims, -1)

    D = 0.5 * torch.prod(h, dim=-1) * torch.norm((T_flat - R_flat), dim=-1)
    return D
