import os
import numpy as np
from PIL import Image
# import nibabel as nib
import torch
import collections

from interpolation import set_interpolater
from grids import Grid
import batch_calculation as bc
from save_results import save_metrics


def save_run(cfile, res_metrics, losses, data_set='test'):
    diffs = res_metrics['Diff']
    iterates = res_metrics['Iterates']
    ratios = res_metrics['Ratios']
    run_times = res_metrics['Set_times']

    fx_ratio = ratios[0]
    loss_ratio = ratios[1]

    # ---------------- Save Results ---------------------------------------
    out_file = os.path.join(cfile, data_set)
    if not os.path.isdir(out_file):
        os.makedirs(out_file)

    tensor2disk(out_file, [iterates, losses, diffs, fx_ratio, loss_ratio,
                           res_metrics['Instance_times'], res_metrics['Instance_energy']],
                ['iterates', 'losses', 'diffs', 'fx_ratio', 'loss_ratio', 'times', 'energies'])

    # metrics should provide overview over results. Only pass last element (loss from last iterate) for every sample in
    # test set
    results = {'Diff_plain': [diffs[0][ii][-1] for ii in range(len(diffs[0]))],
               'Diff_learn': [diffs[1][ii][-1] for ii in range(len(diffs[1]))],
               'Time_plain': f'{int(run_times[0][0]):0>2}:{int(run_times[0][1]):0>2}:{run_times[0][2]:05.2f}',
               'Time_learn': f'{int(run_times[1][0]):0>2}:{int(run_times[1][1]):0>2}:{run_times[1][2]:05.2f}'}

    # save results if combined model is used / current statement is placeholder
    if 'cDiff' in res_metrics:
        results['Diff_combined'] = [res_metrics['cDiff'][ii][-1] for ii in range(len(res_metrics['cDiff']))]
        tensor2disk(out_file, [res_metrics['cIterates'], res_metrics['cDiff'],
                               res_metrics['cRatios'][0], res_metrics['cRatios'][1],
                               res_metrics['cInstance_time'], res_metrics['cInstance_energy']],
                    ['iterates_combined', 'diffs_combined', 'fx_ratio_combined', 'loss_ratio_combined',
                     'times_combined', 'energies_combined'])

    save_metrics(out_file, results)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar', only_best=True):
    if not only_best:
        torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_checkpoint(filepath, mode='train', *args):
    # can not use load_checkpoints, since it is not yet possible to pickle ml_solver wrapper
    checkfiles = [c for c in os.listdir(filepath)]
    if 'model_best.pth.tar' in checkfiles:
        best_file = 'model_best.pth.tar'
    else:
        best_file = checkfiles[-1]
    print(f"loading checkpoint {best_file}")
    checkpoint = torch.load(os.path.join(filepath, best_file), map_location=lambda storage, loc: storage)
    if 'model' in checkpoint.keys():
        # pass model, since wraped model is not pickleable
        model = checkpoint['model']
    elif len(args) > 0:
        model = args[0]
    else:
        raise RuntimeError(f'Model is not saved in checkpoint and no model is passed. Pass model to load checkpoint')

    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    if mode == 'test':
        for parameter in model.parameters():
            parameter.requires_grad = False

    return model


def calc_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds


def tensor2disk(path, tensors, names):
    if len(tensors) != len(names):
        raise RuntimeError('Input list of tensors and name list unequal')

    for t, n in zip(tensors, names):
        # ii should be in range 2, name it, 0 plain, 1 learn
        suffix = ['_plain', '_learn']
        loss_suffix = ['_train', '_val']
        for ii in range(len(t)):
            if n == 'losses':
                np.save(os.path.join(path, n + loss_suffix[ii] + '.npy'), np.stack(t[ii]))
            elif n in ['diffs', 'iterates']:
                torch.save(torch.stack(t[ii]), os.path.join(path, n + suffix[ii] + '.pt'))
            elif n in ['times', 'energies']:
                torch.save(t[ii], os.path.join(path, n + suffix[ii] + '.pt'))

        if n not in ['losses', 'diffs', 'iterates', 'times', 'energies', 'times_combined', 'energies_combined']:
            torch.save(torch.stack(t), os.path.join(path, n + '.pt'))
        else:
            torch.save(t, os.path.join(path, n + '.pt'))


def create_h0(w, fcn_grad, mode='id', **params_fct):
    n = w.shape[1]
    fwk, dfwk = fcn_grad(w, **params_fct)

    # define inverse Hessian, scaled id or triu as cholesky component
    if mode == 'id':
        learn_param = torch.eye(n, dtype=torch.double) / torch.norm(torch.mean(dfwk, dim=0), 2)
    else:
        H0 = torch.sqrt(torch.diag(torch.ones(n, dtype=torch.double) / torch.norm(torch.mean(dfwk, dim=0), 2)))
        ids = torch.triu_indices(n, n)
        learn_param = H0[ids[0], ids[1]]
    return learn_param


def normalize(image, new_max=255):
    return ((image - torch.min(image)) / (torch.max(image) - torch.min(image))) * new_max


def load_images(path, dtype=torch.double):
    if not os.path.exists(path):
        raise NameError(f'{path} does not exists')

    # do not pass suffix but get it from files, raise error if multiple data types exist
    func_dict = {'jpg': lambda file: torch.tensor(np.array(Image.open(file), dtype=np.float), dtype=dtype)}
                 # 'gz': lambda file: torch.tensor(nib.load(file).get_fdata(), dtype=dtype)}
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    suffixes = set([f.split('.')[-1] for f in files])
    if not len(suffixes) == 1:
        raise RuntimeError(f'Multiple data types in directory {suffixes}.')
    suffix = list(suffixes)[0]
    if suffix not in func_dict.keys():
        raise KeyError(f'No function implemented to load data of type {suffix}.')
    load = func_dict[suffix]

    img_names = [im for im in sorted(os.listdir(path)) if im.endswith(suffix)]
    out_imgs = [load(img) for img in files]

    return out_imgs, img_names


def dict_to_device(dict_tensors, device):
    for key in dict_tensors.keys():
        if torch.is_tensor(dict_tensors[key]):
            dict_tensors[key] = dict_tensors[key].to(device)
    return dict_tensors


def txt_to_dict(path, file):
    with open(os.path.join(path, file), 'r') as f:
        param_dict = {}
        if file == 'parameters.txt':
            for line in f:
                if 'Time' not in line:
                    (key, val) = line.split(':')
                    val = val.strip()
                    param_dict[key] = (val == 'True') if val in ['True', 'False'] else val
        else:
            for line in f:
                ll = line.split()
                key = ll[0].strip(':')
                ll.pop(0)
                val = ll
                if key == 'omega':
                    param_dict[key] = torch.tensor([int(om.strip('[').strip(']')) for om in val], dtype=torch.int32)
                elif key == 'type':
                    param_dict[key] = val[0].strip('[').strip(']')
                elif key not in ['Sizes', 'seed']:
                    if isinstance(val, list) and len(val) == 1:
                        param_dict[key] = float(val[0].strip('[').strip(']'))
                    else:
                        param_dict[key] = [float(n.strip('[').strip(']')) for n in val]
    return param_dict


def freeze_layer(model):
    names_blocks = [name for name in model.sb._modules]
    modules = [model.sb._modules[name] for name in names_blocks]
    blocks = collections.OrderedDict(zip(names_blocks, modules))
    for layer in blocks:
        for pn, param in blocks[layer].named_parameters():
            param.requires_grad = False


def unfreeze_layer(model, indx, reset=True):
    if isinstance(model.sb, torch.nn.ModuleList):
        names_blocks = [name for name in model.sb._modules]
        modules = [model.sb._modules[name] for name in names_blocks]
        blocks = collections.OrderedDict(zip(names_blocks, modules))
    else:
        lvl_blocks = model.sb._modules
        blocks = collections.OrderedDict()
        for count, lvl in enumerate(lvl_blocks):
            cdict = lvl_blocks[lvl]._modules
            new_keys = [f'{int(element) + model.num_layers*count}'for element in list(cdict.keys())]
            blocks.update(collections.OrderedDict(zip(new_keys, list(cdict.values()))))

    if indx is not None and indx > len(blocks):
        indx = None
        raise RuntimeError(f'Layer index {indx} exceeds number of layers {len(blocks)}. Enable end-to-end')
    for nn, layer in enumerate(blocks):
        if nn == indx or indx is None:
            print(f'Unfreeze layer: {layer}')
            # set all params to require grad to enable end-to-end training for last epoch, if indx is None
            for pn, param in blocks[layer].named_parameters():
                param.requires_grad = True
        elif indx > 0 and nn == indx - 1 and reset:
            print(f'Freeze layer: {layer}')
            for pn, param in blocks[layer].named_parameters():
                param.requires_grad = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def prep_model_func(param):
    obj_fct = param['objFct']
    dist = obj_fct.dist
    reg = obj_fct.reg
    if reg is None:
        func_list = [bc.auto_gradient_batch(dist.evaluate)]
    else:
        func_list = [bc.auto_gradient_batch(dist.evaluate), bc.auto_gradient_batch(reg.evaluate)]
    return func_list


def getMultilevel(image, omega, m, batchsize, method='linearFAIR', maxLevel=torch.tensor(7), minLevel=torch.tensor(4)):
    Ic = image
    intermethod = method
    maxLevel = maxLevel.type(torch.float32)
    minLevel = minLevel.type(torch.float32)

    # check whether maxLevel is to high for given data and if so, reset value
    maxLevel_data = torch.ceil(torch.log2(torch.min(m).type(maxLevel.dtype)))
    if maxLevel > maxLevel_data:
        print(f'Warning: chosen maxLevel {maxLevel.detach().numpy().astype(np.int)} is too high,'
              f'chose {maxLevel_data.detach().numpy().astype(np.int)} instead')
    level_range = torch.stack((maxLevel, maxLevel_data))
    maxLevel = torch.min(level_range).type(torch.int32)

    m_level = m

    data_ML = []
    m_ML = []
    nP_ML = []
    Ic_res = Ic
    # check if omega is batch-wise or not, to set up interpolation single sample is needed
    if isinstance(omega, list):
        base_omega = omega[0]
    else:
        base_omega = omega.squeeze()

    for level in torch.arange(maxLevel_data, minLevel-1, -1):
        grid = Grid(batchsize, base_omega, m_level)
        if level == maxLevel_data:
            interp = set_interpolater(intermethod, base_omega, m, grid.getSpacing())
            Ic_res = interp.interpolate(Ic, grid.getCellCentered())
            m_res = m_level.type(torch.int32)
        else:
            (Ic_res, m_res) = grid.restrictImage(data=Ic_res, batchsize=batchsize, method=intermethod)

        if level <= maxLevel:
            # if image is given without batch dimension, create dummy batch dim
            if len(Ic_res.shape) < 3:
                Ic_res = Ic_res.unsqueeze(0)
            data_ML.append(Ic_res)
            m_ML.append(m_res)

            nP_ML.append(torch.prod(m_res))
        m_level = m_res
    return data_ML, m_ML, omega, nP_ML
