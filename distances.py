import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import itertools

from interpolation import set_interpolater
import tools
import batch_calculation as bc


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


class PIR(DistanceFunction):
    def __init__(self, deformation, distance, omega, m, h, gridRef):
        super(PIR, self).__init__(deformation, distance, omega, m, h, gridRef)
        self.name = 'PIR'

    def evaluate(self, w, **kwargs):
        T = kwargs['T']
        R = kwargs['R']
        dims = self.m.numel()
        xcs = self.gridRef.clone()
        h = self.h
        deformation = self.trafo

        if len(T.size()) == dims + 1:
            omega = self.omega.repeat(T.shape[0], 1)
            xcs = xcs.squeeze().repeat(T.shape[0], 1, 1)
            deformation = self._repeat(deformation, T.shape[0])
        else:
            # bc.batch_apply works batch-wise and needs batch dummy dim for non batched samples
            deformation = [deformation]
            omega = self.omega.unsqueeze(0)
            w = w.unsqueeze(0)
            T = T.unsqueeze(0)
            R = R.unsqueeze(0)

        xcs = xcs.to(R.device)
        omega = omega.to(R.device)
        h = h.to(R.device)

        rot_xc = bc.batch_apply(deformation, w, xcs.shape, xcs, omega)
        Ty = self.inter.interpolate(T, rot_xc)
        Ty = Ty.to(R.device)

        return self.dist(Ty, R, h)

    def __copy__(self):
        return PIR(self.trafo, self.dist, self.omega, self.m, self.h, self.gridRef)


class NPIR(DistanceFunction):
    def __init__(self, deformation, distance, omega, m, h, gridRef):
        super(NPIR, self).__init__(deformation, distance, omega, m, h, gridRef)
        self.name = 'NPIR'

    def evaluate(self, w, **kwargs):
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
            xcs = xcs.squeeze().repeat(B, 1, 1)
            omega = self.omega.repeat(B, 1)
            deformation = self._repeat(deformation, T.shape[0])
        else:
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
    return 0.5 * torch.prod(h, dim=-1) * torch.sum((T_flat - R_flat)**2, dim=-1)
