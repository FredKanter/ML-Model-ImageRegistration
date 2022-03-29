import torch

from derivatives import second_order_derivative_conv, second_order_derivative_conv3d
from grids import reshape_grid
import deformations as deform


class Regularizer:
    def __init__(self, deformation, omega, m, h, gridRef):
        super(Regularizer, self).__init__()
        if not isinstance(deformation, deform.Deformation):
            raise RuntimeError('Deformation has to be child of class Deformation')
        devs = {'2': second_order_derivative_conv, '3': second_order_derivative_conv3d}
        self.omega = omega
        self.h = h
        self.m = m
        self.gridRef = gridRef
        self.dims = m.numel()
        self.nonpara = isinstance(deformation, deform.NonPara)
        self.dfm_fcn = deformation.evaluate
        self.calc_dev = devs[str(self.dims)]

    def evaluate(self, xc, **kwargs):
        xc = self.convert_to_grid(xc)
        xc = self.resize_grid(xc)
        gridsize = xc.size()
        B = gridsize[0]
        h = self.h.unsqueeze(0).repeat(B, 1)
        return torch.norm(xc - self.gridRef.reshape([1, *[s for s in gridsize[1::]]]).repeat([B, *(self.dims + 1) * [1]]), 2) * torch.prod(h, dim=-1)

    def resize_grid(self, xc):
        # xc has to be passed flattend (fit LSTM), reshape consistent with flatten operation
        return reshape_grid(xc, self.m).permute((0, self.dims + 1, *[d.item() for d in torch.arange(1, self.dims + 1)]))

    def to_device(self, device):
        return self.h.to(device), self.gridRef.to(device), self.m.to(device), self.omega.to(device)

    def convert_to_grid(self, x0):
        # regularizer needs grid, not parametrization
        if not self.nonpara:
            return self.dfm_fcn(x0, self.gridRef, self.omega)
        else:
            # if deformation is non parametric, iterate x0 is already a grid, but may be flatten
            if len(x0.shape) == 1:
                x0 = x0.unsqueeze(0)
            # check if grid is flatten or not
            B = x0.shape[0]
            x0 = x0.view(B, -1)
            B, CHW = x0.shape
            x0 = x0.reshape((B, self.dims, CHW // self.dims))
            return x0


class Curvature(Regularizer):
    # defined on cell centered grid
    def __init__(self, deformation, omega, m, h, gridRef, mode='conv'):
        super(Curvature, self).__init__(deformation, omega, m, h, gridRef)
        self.name = 'curvature'
        self.mode = mode

    def evaluate(self, xc, **kwargs):
        xc = self.convert_to_grid(xc)
        xc = self.resize_grid(xc)
        gridsize = xc.size()
        B = gridsize[0]

        h, gridRef, m, omega = self.to_device(xc.device)

        uc = xc - gridRef.reshape([1, *[s for s in gridsize[1::]]]).repeat([B, *(self.dims+1)*[1]])
        hp = h.unsqueeze(0).repeat(B, 1)

        if self.mode == 'conv':
            # this step denotes the integral of curvature := sum_{i}(lap(yi)^2), and dim = 1 defines indx i
            # lap(yi)^2
            devs = self.calc_dev(uc, h)
            L = torch.sum(torch.sum(torch.stack(devs, dim=0), dim=0)**2, dim=1)
        else:
            raise RuntimeError('Mode has to be conv')

        L = torch.sum(L, dim=[d.item() for d in torch.arange(-self.dims, 0)])

        return 1/2 * L * hp.prod(dim=-1)


def set_regularizer(name, deformation, omega, m, h, gridRef):
        if name == 'curvature':
            S = Curvature(deformation, omega, m, h, gridRef)
        else:
            S = None
            print(f'Method {name} not implemented. Use no regularizer')
        return S
