import torch
from torch.utils import data
import os
import copy
import time
from collections import OrderedDict

from setup_data import HdfDataset
from models import MyDataParallel
import tools as tools
from loss import set_loss
import grids as gutils
import deformations as df

import batch_calculation as bc


def predict(model, solver_obj, params, dataset='test.h5', combined=False, noise=False):
    print(f'Prediction for dataset {dataset}')
    params['device'] = torch.device('cpu')

    # check for npir mode (adjustments for metrics)
    npir_mode = True if isinstance(params['deformation'], df.NonPara) else False

    # load model from checkpoint
    model = tools.load_checkpoint(params['dir_checkpoints'], 'test', model)
    # remove data_parallel if used
    if isinstance(model, MyDataParallel):
        model = model.module.to(params['device'])
    model.eval()

    # set loss fct / classical solver uses deform loss
    lobj = set_loss('deform', params['loss'][0], verbose=False)
    fct_loss = lobj.evaluate

    # get params for distance form objective function
    objFct = copy.copy(params['objFct'])

    # create copy of solver for last steps and set original solver for evaluation
    cs_obj = copy.copy(solver_obj)
    cs_obj.set_iter(params['test_iter']//2)
    add_solver = cs_obj.solve
    solver = solver_obj.solve

    # get path to save surf and heat plots for loss on grid
    grid_plots_path = os.path.join(params['dir_checkpoints'].split('checkpoints')[0], dataset.split('.')[0], 'grid_plots')
    if not os.path.exists(grid_plots_path) and npir_mode:
        os.makedirs(grid_plots_path)

    model.to(params['device'])

    # define as dict here
    file = params['data']

    # set up data generator
    gen_params = {'batch_size': 1,
                  'shuffle': False}

    test_set = HdfDataset(file, dataset, add_noise=noise)
    test_gen = data.DataLoader(test_set, **gen_params)

    loss_learn, loss_combined, loss_plain = [], [], []

    # return iterates from learned and plain
    iterates_learn, iterates_combined, iterates_plain = [], [], []
    fx_learn, fx_all_learn, fx_combined, fx_all_combined, fx_plain, fx_all_plain = [], [], [], [], [], []

    # also return reference images for visualization
    lst_full = time.time()
    lst_instance, let_instance, cet_instance = [], [], []
    for w, adds in test_gen:
        with torch.no_grad():
            w = w.to(params['device'])
            adds = tools.dict_to_device(adds, params['device'])

            lst_instance = [*lst_instance, time.time()]
            wk, fks, gk = model(w, mode='test', timed=True,
                                params_fct=adds)

            if combined:
                H0 = set_up_hessian(wk[-1], bc.auto_gradient_batch(objFct.evaluate), adds, ML_flag=params['ML'],
                                    device=params['device'])
                wkk, fkks, gkk = add_solver(copy.copy(objFct), wk[-1], H0, mode='test', timed=True,
                                            **{'params_fct': adds})
                # Calculate Distances for combined approach
                iterates_combined, loss_combined, fx_combined, fx_all_combined, cet_instance = prepare_metrics(wkk, fkks, iterates_combined, loss_combined, fx_combined, fx_all_combined, cet_instance, fct_loss, objFct, adds, npir_mode)

            # learn solver. Calculate Distances / check loss for all iterates
            iterates_learn, loss_learn, fx_learn, fx_all_learn, let_instance = prepare_metrics(wk, fks, iterates_learn, loss_learn, fx_learn, fx_all_learn, let_instance, fct_loss, objFct, adds, npir_mode)

    learn_time = tools.calc_time(lst_full, time.time())
    learn_its = time_per_instance(lst_instance, let_instance)

    pst_full = time.time()
    pst_instance, pet_instance = [], []
    for wp, asp in test_gen:
        with torch.no_grad():
            # test using plain param
            wp = wp.to(params['device'])
            asp = tools.dict_to_device(asp, params['device'])
            H0 = set_up_hessian(wp[-1], bc.auto_gradient_batch(objFct.evaluate), asp, ML_flag=params['ML'],
                                device=params['device'])

            pst_instance = [*pst_instance, time.time()]
            wkp, fksp, gkp = solver(copy.copy(objFct), wp, H0, mode='test', timed=True,
                                    **{'params_fct': asp})

            iterates_plain, loss_plain, fx_plain, fx_all_plain, pet_instance = prepare_metrics(wkp, fksp, iterates_plain, loss_plain, fx_plain, fx_all_plain, pet_instance, fct_loss, objFct, asp, npir_mode)

    plain_time = tools.calc_time(pst_full, time.time())
    plain_its = time_per_instance(pst_instance, pet_instance)

    # Ratio of function values f(xkp) / f(xkl)
    fx_ratio = [(fx_plain[sample] / fx_learn[sample]).squeeze() for sample in range(len(fx_plain))]

    # difference in final loss loss(xkp) - loss(xkl)
    loss_ratios = [loss_plain[sample][-1] - loss_learn[sample][-1] for sample in range(len(loss_plain))]

    # construct dict for params to return
    out = {'Diff': (loss_plain, loss_learn),
           'Iterates': (iterates_plain, iterates_learn),
           'Ratios': (fx_ratio, loss_ratios),
           'Set_times': (plain_time, learn_time),
           'Instance_times': (torch.tensor(plain_its), torch.tensor(learn_its)),
           'Instance_energy': (torch.tensor(fx_all_plain), torch.tensor(fx_all_learn))}

    # Ratio and differences for combined approach
    if combined:
        cfx_ratio = [(fx_plain[sample] / fx_combined[sample]).squeeze() for sample in range(len(fx_plain))]
        closs_ratios = [loss_plain[sample][-1] - loss_combined[sample][-1] for sample in range(len(loss_plain))]
        c_its = time_per_instance(lst_instance, cet_instance)
        out['cIterates'] = iterates_combined
        out['cDiff'] = loss_combined
        out['cRatios'] = (cfx_ratio, closs_ratios)
        out['cInstance_time'] = torch.tensor(c_its)
        out['cInstance_energy'] = torch.tensor(fx_all_combined)

    return out


def time_per_instance(start_times, end_times):
    # calculate time for each optimization step for each iterate
    run_time = []
    for idx, st in enumerate(start_times):
        run_time = [*run_time, [tools.calc_time(st, et) for et in end_times[idx]]]
    return run_time


def prepare_metrics(wk, fk, iterates, loss, fx, fx_all, time, fct_loss, objFct, adds, npir=False):
    if not all(map(lambda x: isinstance(x, list), [iterates, loss, fx, fx_all, time])):
        raise RuntimeError(f'Iterates, Loss, Fx, Fx_all, and time have to be of type list')

    # if NPIR, iterates can have different sizes, prolong to max resolution for evaluation purposes
    if npir:
        # image dims should later be passed as parameter
        C = len(objFct.m)
        wk = [wi.reshape(wi.shape[0], C, wi.shape[-1] // C) for wi in wk]
        B, _, HW_m = wk[-1].shape
        _, _, HW_f = wk[0].shape
        max_m = torch.tensor([int(HW_m ** (1 / C)) for _ in range(C)], dtype=torch.int)
        first_m = torch.tensor([int(HW_f ** (1 / C)) for _ in range(C)], dtype=torch.int)
        if not torch.all(torch.eq(max_m, first_m)):
            wk = [gutils.scale_grid(wi, torch.tensor([int(wi.shape[-1] ** (1 / C)) for _ in range(C)], dtype=torch.int), max_m) for wi in wk]
        wk = [wi.reshape(wi.shape[0], -1) for wi in wk]

    iterates = [*iterates, torch.stack([ii.detach().clone() for ii in wk], dim=-1).squeeze()]
    loss = [*loss, torch.stack([fct_loss(ii, adds['x*'], copy.copy(objFct), params_fct=adds).detach() for ii in wk], dim=-1)]
    fx = [*fx, fk[-1][0].detach()]
    fx_all = [*fx_all, [f[0].detach() for f in fk]]
    time = [*time, [f[1] for f in fk]]
    return iterates, loss, fx, fx_all, time


def set_up_hessian(w, objFct, adds, ML_flag, device='cpu'):
    # Hessian is created per level in ML wrapper / Hessian for image pair
    H0 = None if ML_flag else tools.create_h0(w, bc.auto_gradient_batch(objFct.evaluate), **adds).to(device)
    return H0


def remove_DataParallel(state):
    # if model create using DataParallel and evaluated without it. 'module' prefix has to be removed
    old_state_dict = state['state_dict']
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # remove `module.`
        name = k.replace('module.', '')
        new_state_dict[name] = v
    state['state_dict'] = new_state_dict
    return state
