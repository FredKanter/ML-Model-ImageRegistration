import torch
import numpy as np
import time
import copy

from armijo_BFGS import limited_bfgs
import batch_calculation as bc
import tools as tools
from grids import Grid, prolong_grid, scale_to_gridres


class Solver:
    def __init__(self, iterations, sparse, **kwargs):
        super(Solver, self).__init__()
        # ml flag enables multi level solver
        self.iter = iterations
        self.sparse = sparse
        if 'ml_max_lvl' in kwargs.keys():
            if torch.is_tensor(kwargs['ml_max_lvl']):
                self.ml_max_lvl = kwargs['ml_max_lvl'].detach().clone()
            else:
                self.ml_max_lvl = torch.tensor(kwargs['ml_max_lvl'], dtype=torch.int32)
        else:
            self.ml_max_lvl = torch.tensor(7, dtype=torch.int32)

    def set_iter(self, n):
        self.iter = n

    def solve(self, fng, x0, learn_param, timed=False, **kwargs):
        fng = fng.evaluate
        fng_grad = bc.auto_gradient_batch(fng)
        fks, gks = fng_grad(x0, **kwargs['params_fct'])
        return x0, fks, gks

    def convert(self, x):
        if self.sparse:
            return x.to_sparse()
        else:
            return x.to_dense()

    def __copy__(self):
        return Solver(self.iter, self.sparse)


class GD(Solver):
    def __init__(self, iterations, sparse, **kwargs):
        super(GD, self).__init__(iterations, sparse, **kwargs)
        self.name = 'GD'

    def solve(self, fng, x0, learn_param, timed=False, **kwargs):
        xk = x0
        if isinstance(learn_param, torch.nn.ParameterList):
            step = learn_param[0]
        else:
            step = learn_param
        fng_grad = bc.auto_gradient_batch(fng)
        fks = []
        gks = []

        for k in range(self.iter):
            fk, gk = fng_grad(xk, **kwargs['params_fct'])
            xk = xk - step * gk
            if timed:
                fks = [*fks, (fk, time.time())]
            else:
                fks.append(fk)
            gks.append(gk)

        return xk, fks, gks

    def __copy__(self):
        return GD(self.iter, self.sparse)


class L_BFGS(Solver):
    def __init__(self, iterations, sparse, **kwargs):
        super(L_BFGS, self).__init__(iterations, sparse, **kwargs)
        self.name = 'BFGS'
        if 'bfgs_params' not in kwargs.keys():
            raise RuntimeError(f'Solver of type {self.name} needs dict with params maxIter, lr, LS, and max level')
        self.maxBFGS = kwargs['bfgs_params'][0]
        self.lr = kwargs['bfgs_params'][1]
        self.LS = kwargs['bfgs_params'][2]

    def solve(self, fng, x0, learn_param, timed=False, **kwargs):
        # pass function object and set to evaluation routine
        Q = self._prepare_hessian(x0, learn_param)

        xk, fks, gks = limited_bfgs(fng, x0, Q, timed=timed,
                                    **{'lr': self.lr, 'maxIter': self.iter, 'maxBFGS': self.maxBFGS, **kwargs})
        return xk, fks, gks

    def _prepare_hessian(self, x0, learn_param):
        if isinstance(learn_param, torch.nn.ParameterList):
            insert_param = learn_param[0]
        else:
            insert_param = learn_param

        sz = insert_param.shape
        if len(sz) == 1:
            # data is in form of ids of diag matrix (cholesky component) / variant to ensure positive definite-ness,
            n = x0.size(1)
            H0 = torch.zeros((n, n), dtype=x0.dtype, device=insert_param.device)
            ids = torch.triu_indices(n, n)
            H0[ids[0], ids[1]] = insert_param

            # force H0 to be positive definit, using only symmetric part
            if not self.sparse:
                return torch.matmul(H0, H0.transpose(0, 1))
            else:
                return self.convert(torch.sparse.mm(self.convert(H0), H0.transpose(0, 1)))
        else:
            # data is dense inverse hessian (scaled identity)
            return insert_param

    def __copy__(self):
        return L_BFGS(self.iter, self.sparse, **{'bfgs_params': [self.maxBFGS, self.lr, self.LS],
                                                 'ml_max_lvl': self.ml_max_lvl.detach().clone()})

    @staticmethod
    def check_pos_definit(H):
        Hs = H.detach().clone().cpu().numpy()
        if not np.all(np.linalg.eigvals(Hs) > 0):
            # values all close to zero, error thrown for -2*e⁻27
            raise ValueError('H0 is not positive definit !')


class ML_wrapper(Solver):
    def __init__(self, iterations, sparse, **kwargs):
        super(ML_wrapper, self).__init__(iterations, sparse, **kwargs)
        self.iter = iterations
        if 'solver' not in kwargs.keys():
            raise RuntimeError(f'ML_wrapper needs object of type Solver')
        solver = kwargs['solver']
        self.name = f'ML-{solver.name}'
        self.sol = copy.copy(solver)
        if 'bfgs_params' in kwargs.keys():
            self.maxBFGS = kwargs['bfgs_params'][0]
            self.lr = kwargs['bfgs_params'][1]
            self.LS = kwargs['bfgs_params'][2]

    def set_iter(self, n):
        self.iter = n
        self.sol.iter = n

    def solve(self, fng, x0, learn_param, timed=False, **kwargs):
        # creates ML every Epoch, better to construct once for one run and save in additional dict entries
        w = x0.clone()
        in_params = kwargs['params_fct']
        m = torch.tensor(in_params['T'][0].shape)
        batch = in_params['T'].shape[0]
        data_ML, m_ML, omega, nP_ML = tools.getMultilevel(in_params['T'], fng.omega,
                                                          m, batch, maxLevel=self.sol.ml_max_lvl)
        R_ML = tools.getMultilevel(in_params['R'], fng.omega, m, batch, maxLevel=self.sol.ml_max_lvl)[0]

        fwk_t, wk_all = [], []

        # start with lowest resolution grid/param
        if fng.dist.name == 'NPIR':
            s_grid = Grid(batch, omega, m_ML[-1])
            w = scale_to_gridres(w, s_grid.getCellCentered(), m, m_ML[-1])
            fng.reset(omega, s_grid.getSpacing(), m_ML[-1], s_grid.getCellCentered().clone())
            level_param = tools.create_h0(w, bc.auto_gradient_batch(fng.evaluate),
                                          **{'T': data_ML[-1], 'R': R_ML[-1]})

        # multi-level approach / run through for different levels, one ML run should replace one vanilla solver run
        # incorporate multi-level approach
        all_level = torch.sort(torch.arange(0, len(data_ML)), descending=True).values
        for level in all_level:
            in_params['T'], in_params['R'] = data_ML[level].to(w.device), R_ML[level].to(w.device)

            # create batched grid, than expand in loss is not necessary
            level_grid = Grid(1, omega, m_ML[level])
            # do reset for new objective
            fng.reset(omega, level_grid.getSpacing(), m_ML[level], level_grid.getCellCentered().clone())

            # update of iterates (nonpara needs scaling to new grid size)
            if level < all_level[0] and fng.dist.name == 'NPIR':
                w, level_param = self.update_ml(w, wk, fng, level_grid, m_ML, level, **in_params)
            elif fng.dist.name == 'PIR' and learn_param is not None:
                level_param = learn_param[level]
                if level > all_level[0]:
                    # solvers return list of iterates, we are just interested in last element for next level
                    w = wk[-1]
            else:
                if level > all_level[0]:
                    # solvers return list of iterates, we are just interested in last element for next level
                    w = wk[-1]
                # for plain we want to use image pair specific Hessian. No longer learnable as ParameterList
                level_param = tools.create_h0(w, bc.auto_gradient_batch(fng.evaluate), **{'T': in_params['T'],
                                                                                          'R': in_params['R']})

            wk, fwk, gwk = self.sol.solve(fng, w, level_param, mode=kwargs['mode'], timed=timed,
                                          params_fct=in_params, protocol=kwargs['protocol'])

            # return time for whole ML routine
            fwk_t = add_times(fwk_t, fwk)
            # pop first element if list is not empty to avoid redundant values (solver adds starting iterate)
            if not wk:
                wk.pop(0)
            wk_all = add_times(wk_all, wk)

        if timed:
            fwk = fwk_t

        # return wk for single iterate approaches
        return wk_all, fwk, gwk

    def update_ml(self, w, wk, fcn, level_grid, m_ML, level, **params_fct):
        # update H0 and resize grid for NPIR registration
        fcn_grad = bc.auto_gradient_batch(fcn.evaluate)
        w = prolong_grid(w - wk[-1], level_grid.getCellCentered(), m_ML[level + 1], m_ML[level])
        level_param = tools.create_h0(w, fcn_grad, **params_fct)
        return w, level_param

    def __copy__(self):
        return ML_wrapper(self.iter, self.sparse, **{'solver': self.sol,
                                                     'ml_max_lvl': self.ml_max_lvl.detach().clone()})


def set_solver(name, iterations, sparse=False, ml_flag=False, **kwargs):
    if name == 'GD':
        if ml_flag:
            kwargs['solver'] = GD(iterations, sparse, **kwargs)
            return ML_wrapper(iterations, sparse, **kwargs)
        else:
            return GD(iterations, sparse, **kwargs)
    elif name in ['BFGS', 'ML-BFGS']:
        if ml_flag:
            kwargs['solver'] = L_BFGS(iterations, sparse, **kwargs)
            return ML_wrapper(iterations, sparse, **kwargs)
        else:
            return L_BFGS(iterations, sparse, **kwargs)
    else:
        raise RuntimeError(f'Solver method {name} not implemented')


def add_times(l, t):
    for ii in range(len(t)):
        l.append(t[ii])
    return l

