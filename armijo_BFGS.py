import os
import torch
import time
import warnings
import numpy as np

import batch_calculation as bc
from save_results import write_csv_file


def limited_bfgs(fctn, yc, H0, mode='train', timed=False, **kwargs):
    # ------------- INIT -----------------------------------
    ym, sm = [], []

    fctn_ev = fctn.evaluate
    fctn_grad = bc.auto_gradient_batch(fctn_ev)
    [Jc, dJ] = fctn_grad(yc, **kwargs['params_fct'])
    fxs = [(Jc, time.time())] if timed else [Jc]

    # set up list of iterates to use in loss
    gxs, yks = [dJ], [yc]

    yOld, dJold = 0 * yc, copy_tensor(dJ)
    kwargs['iter'] = 0

    # ------------- BODY ----------------------------------
    for ii in range(kwargs['maxIter']):
        dy, ym, sm, kwargs = update_direction(ii + 1, dJ, dJold, yc, yOld, H0, ym, sm,  **kwargs)

        alpha, LS_ep, kwargs = calc_step(mode, fctn_ev, yc, dy, Jc, dJ, kwargs)

        yt = yc + alpha.to(dy.device).detach().clone() * dy.clone()

        # update variables
        yOld = copy_tensor(yc)
        dJold = copy_tensor(dJ)
        yc = yt.clone()
        [Jc, dJ] = fctn_grad(yc, **kwargs['params_fct'])

        fxs = [*fxs, (Jc, time.time())] if timed else [*fxs, Jc]
        gxs, yks = [*gxs, dJ], [*yks, yc]

    return yks, fxs, gxs


def update_direction(current_iter, dJ, dJold, yc, yOld, H, ym_batch, sm_batch, **kwargs):
    dy_all, cc_values = [], []
    for ii in range(yc.shape[0]):
        if len(ym_batch) <= ii:
            ym_batch, sm_batch = [*ym_batch, []], [*sm_batch, []]

        ym = ym_batch[ii]
        sm = sm_batch[ii]
        m_count = len(ym)

        if current_iter > 1:
            yk = dJ[ii] - dJold[ii]
            sk = yc[ii] - yOld[ii]
            kwargs['iter'] = current_iter - 1

            if bc.batch_matmul(yk, sk) <= 0:
                warnings.warn(10*'-' + ' BFGS corrupted, curvature condition violated, stepsize not suited ' + 10*'-')
            else:
                ym = [*ym, yk]
                ym = ym[-kwargs['maxBFGS']:]
                sm = [*sm, sk]
                sm = sm[-kwargs['maxBFGS']:]
                m_count = min(kwargs['maxBFGS'], len(ym))

        dy = bfgsrec(m_count, sm, ym, H, -dJ[ii])
        dy_all = [*dy_all, copy_tensor(dy)]
        ym_batch[ii] = ym
        sm_batch[ii] = sm

    dy = torch.stack(dy_all)

    return dy, ym_batch, sm_batch, kwargs


def bfgsrec(n, S, Z, H, d):
    if n == 0:
        d = bc.batch_matmul(H, d)
    else:
        alpha = bc.batch_matmul(S[n - 1], d) / (bc.batch_matmul(Z[n - 1], S[n - 1]))
        d = d - alpha * Z[n - 1]
        d = bfgsrec(n - 1, S, Z, H, d)
        d = d + (alpha - (bc.batch_matmul(Z[n - 1], d)) / bc.batch_matmul(Z[n - 1], S[n - 1])) * S[n - 1]
    return d


def bc_LS(fcn, Yc, dY, Jc, dJ, **kwargs):
    LSmaxIter = 10  # max number of trials / default set to 10
    LSreduction = 1 * 10 ** -4  # slope of line

    # adapt LS for batch-wise computation, maybe with bool indexing
    t_batch, L = [], []

    # do not break but use fix iterations in LS and later on use network to choose step size
    for jj in range(Yc.shape[0]):
        # check for single batch Jc, may adapt output of fcn
        if len(Jc.shape) == 0:
            tmp_jc = copy_tensor(Jc)
        else:
            tmp_jc = copy_tensor(Jc[jj])

        descent = bc.batch_matmul(copy_tensor(dJ[jj]), copy_tensor(dY[jj]))
        tmp_yc = copy_tensor(Yc[jj])
        tmp_dy = copy_tensor(dY[jj])
        tmp_args = select_batch_params(kwargs['params_fct'], jj)
        L_batch = []

        t = 1
        for ii in range(LSmaxIter):
            Yt = tmp_yc + t * tmp_dy
            Jt = fcn(Yt, **tmp_args)
            LS = torch.lt(Jt, tmp_jc + t * LSreduction * descent)
            L_batch.append(t)
            if LS:
                break
            else:
                t = t / 2

        if not LS:
            t = 0
            warnings.warn('Line Search failed - break')

        t_batch, L = [*t_batch, torch.tensor(t, dtype=Yc.dtype, requires_grad=False)], [*L, L_batch]

    return torch.stack(t_batch, dim=0).unsqueeze(1), L, kwargs


def select_batch_params(params_fct, indx):
    batch_params = {}
    for ii in params_fct.keys():
        batch_params[ii] = copy_tensor(params_fct[ii][indx])
    return batch_params


def calc_step(mode, *args):
    # args = fc, xk, pk, fxk, dfxk, **kwargs
    if mode == 'test':
        # use line search for test
        alpha, LS_ep, kwargs = bc_LS(args[0], args[1], args[2], args[3], args[4], **args[-1])
    else:
        alpha = torch.tensor(args[-1]['lr'])
        LS_ep = None
        kwargs = args[-1]

    return alpha, LS_ep, kwargs


def copy_tensor(a):
    return a.clone().detach()

