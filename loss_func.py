import torch
import itertools
import matplotlib.pyplot as plt

from grids import Grid, reshape_grid
from derivatives import first_order_derivative_conv

import batch_calculation as bc


class Loss:
    def __init__(self, objective, verbose):
        super(Loss, self).__init__()
        self.name = 'deform'
        self.objective = objective
        self.verbose = verbose

    def evaluate(self, wk, wmin, objfct, **kwargs):
        return self.base(wk, wmin, objfct, **kwargs)

    def base(self, wk, wmin, objfct, **kwargs):
        loss = self._set_objective()
        if isinstance(wk, list):
            wk = wk[-1]

        deformation, xcs, omega, h = self._prep_deform(wk, objfct)

        wk_xc = bc.batch_apply(deformation, wk, xcs.shape, xcs, omega)
        wmin_xc = bc.batch_apply(deformation, wmin, xcs.shape, xcs, omega)

        # keep discretization constant with h and scale over batch size
        return loss(wk_xc, wmin_xc, h)

    def _set_objective(self):
        # construct loss and non reducted loss for verbose mode
        # if we want to mask loss, reduction has to be set to none
        if self.objective == 'MSE':
            self.obj = torch.nn.MSELoss(reduction='none')
            loss = self.torch_loss
        elif self.objective == 'L1':
            self.obj = torch.nn.L1Loss(reduction='none')
            loss = self.torch_loss
        else:
            raise RuntimeError(f'{self.objective} Loss not implemented')
        return loss

    def torch_loss(self, gt, pred, h):
        return torch.sum((self.obj(gt, pred) * torch.prod(h, dim=-1)), dim=[-2, -1]).mean()

    def plot_loss(self, wk, wmin, objfct):
        if self.verbose:
            obj = self._set_objective()

            deformation, xcs, omega, h = self._prep_deform(wk, objfct)

            wk_xc = bc.batch_apply(deformation, wk, xcs.shape, xcs, omega)
            wmin_xc = bc.batch_apply(deformation, wmin, xcs.shape, xcs, omega)

            # plot loss per sample in batch
            fig = plt.figure()
            plt.plot(torch.sum((obj(wk_xc, wmin_xc) * torch.prod(h, dim=-1)), dim=[-2, -1]).detach().cpu().numpy(), 'x')
            plt.title('Loss values')
            plt.xlabel('Sample n')
            plt.ylabel(self.objective)
            return fig
        else:
            return plt.figure()

    @staticmethod
    def _prep_deform(x_k, objfct):
        # not dynamic, since loss is not defined per sample and can not be passed
        rep = lambda ex, bs: list(itertools.chain.from_iterable(itertools.repeat(x, bs) for x in [ex]))
        xcs = objfct.gridRef.clone().squeeze().repeat(x_k.shape[0], 1, 1)
        omega = objfct.omega.repeat(x_k.shape[0], 1)
        deformation = rep(objfct.dist.trafo, x_k.shape[0])

        return deformation, xcs.to(x_k.device), omega.to(x_k.device), objfct.h.to(x_k.device)


def set_loss(name, objective, verbose=False):
    if name == 'deform':
        return Loss(objective, verbose)
    else:
        raise RuntimeError(f'{name} loss not implemented')


class Scheduler:
    def __init__(self, optimizer, **params):
        super(Scheduler, self).__init__()
        self.name = params['name']
        self.params = params
        self.opt = optimizer

    def _set_scheduler(self):
        if not isinstance(self.params, dict):
            raise RuntimeError('Parameter for learning rate scheduler should be passed in dict format')

        param_keys = self.params.keys()
        if self.name == 'CosineAnnealing':
            if 'param' not in param_keys:
                raise RuntimeError(f'No parameter Tmax in {param_keys}')
            else:
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.params['param'])
        elif self.name == 'ExponentialDecay':
            if 'param' not in param_keys:
                raise RuntimeError(f'No parameter gamma in {param_keys}')
            else:
                return torch.optim.lr_scheduler.ExponentialLR(self.opt, self.params['param'])
        elif self.name == 'ReduceOnPlateau':
            if 'param' not in param_keys:
                raise RuntimeError(f'No parameter patient in {param_keys}')
            else:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=self.params['param'])
        elif self.name == 'StepLR':
            if 'param' not in param_keys:
                raise RuntimeError(f'No parameters step, gamma and end in {param_keys}')
            else:
                return torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.params['param'][0],
                                                       gamma=self.params['param'][1],
                                                       last_epoch=self.params['param'][2])
        else:
            raise RuntimeError(f'Scheduler {self.name} not implemented')
