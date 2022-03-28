import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import torch
from matplotlib import pyplot
from torch.utils import data
import math
import collections

from image_registration.reg_utils import deformations as deform
from image_registration.reg_utils.interpolation import set_interpolater
from image_registration.reg_utils.grids import Grid
from image_registration.reg_utils.visualization import construct_large, overlay_imgs, plot_grad_flow, plot_his, \
    plot_histo_grads
# from preprocessing_cg.setup_data import HdfDataset
import utils.batch_calculation as bc
import learn_optimization.evaluate as op_eval
from utils import batch_calculation as bc
from utils.model_summary import summary, countParameters
from utils.save_results import save_to_disk


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

    # op_eval.save_metrics(cfile, results)
    op_eval.save_metrics(out_file, results)


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar', only_best=True):
    # https://github.com/pytorch/examples/blob/1de2ff9338bacaaffa123d03ce53d7522d5dcc2e/imagenet/main.py#L249
    if not only_best:
        torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_checkpoint(filepath, mode='train', *args):
    # can not use load_checkpoints, since it is not yet possible to pickle ml_solver wraper
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
    # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
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


def prepare_net_params(file, fcn, solver_name, dtype=torch.double):
    if 'BFGS' in solver_name:
        # train_set = HdfDataset(file, 'train.h5')
        # # set up data generator
        # train_gen = data.DataLoader(train_set, **{'batch_size': train_set.__len__(), 'shuffle': False})
        # w, adds = next(iter(train_gen))
        # params_fct = {'T': adds['T'],
        #               'R': adds['R']}

        # Index of images depends on date of creation, so be careful if moving images to another dir
        # determine H0 before first solver iteration in training and predict routine, set to None to make visible
        # TODO decide for one option and fix path to avoid nasty workaround
        pathImg = '/' + '/'.join(file.split('/')[1:4])
        images, _ = load_images(pathImg, '.jpg')
        indicies = np.random.choice(len(images), size=2, replace=False)
        params_fct = {'T': images[indicies[0]].unsqueeze(0),
                      'R': images[indicies[1]].unsqueeze(0)}

        if fcn.dist.name == 'PIR':
            w = torch.tensor([[1.005, 0], [0, 1.005], [math.sqrt(2) * 0.01, math.sqrt(2) * 0.01]], dtype=dtype).flatten().unsqueeze(0)
        else:
            grid = Grid(1, fcn.omega, torch.tensor(images[indicies[0]].shape))
            w = grid.getCellCentered().flatten(1, 2)

        fcn_grad = bc.auto_gradient_batch(fcn.evaluate)
        net_param = create_h0(w, fcn_grad, **params_fct)
    else:
        alpha = 0.001
        net_param = torch.tensor(alpha, dtype=dtype)

    return net_param


def create_h0(w, fcn_grad, mode='id', **params_fct):
    n = w.shape[1]
    fwk, dfwk = fcn_grad(w, **params_fct)

    # critical point define inverse Hessian, scaled id or triu as cholesky component
    if mode == 'id':
        # scalar for comparison with matlab
        # learn_param = torch.eye(n, dtype=torch.double) * 0.001
        learn_param = torch.eye(n, dtype=torch.double) / torch.norm(torch.mean(dfwk, dim=0), 2)
    else:
        H0 = torch.sqrt(torch.diag(torch.ones(n, dtype=torch.double) / torch.norm(torch.mean(dfwk, dim=0), 2)))
        ids = torch.triu_indices(n, n)
        learn_param = H0[ids[0], ids[1]]
    return learn_param


def normalize(image, new_max=255):
    return ((image - torch.min(image)) / (torch.max(image) - torch.min(image))) * new_max


def load_images(path, suffix, normalize=False, mode_fair=False, dtype=torch.double):
    if not os.path.exists(path):
        raise NameError(f'{path} does not exists')
    # if not isinstance(suffix, str):
    #     raise TypeError(f'{suffix} is not of valid string type')
    # do not pass suffix but get it from files, raise error if multiple data types exist
    func_dict = {'jpg': lambda file: torch.tensor(np.array(Image.open(file), dtype=np.float), dtype=dtype),
                 'gz': lambda file: torch.tensor(nib.load(file).get_fdata(), dtype=dtype)}
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    suffixes = set([f.split('.')[-1] for f in files])
    if not len(suffixes) == 1:
        raise RuntimeError(f'Multiple data types in directory {suffixes}.')
    suffix = list(suffixes)[0]
    if suffix not in func_dict.keys():
        raise KeyError(f'No function implemented to load data of type {suffix}.')
    load = func_dict[suffix]

    # could use os.walk to get root, dir, files in path
    img_names = [im for im in sorted(os.listdir(path)) if im.endswith(suffix)]
    # image normalization to [0, 1], if 255 is biggest value
    # if normalize:
    #     images = [np.array(Image.open(os.path.join(path, im)), dtype=np.float)/255 for im in img_names]
    # else:
    #     images = [np.array(Image.open(os.path.join(path, im)), dtype=np.double) for im in img_names]
    #
    # # flip images like FAIR does to compare reg results
    # if mode_fair:
    #     out_imgs = [torch.tensor(np.flip(np.transpose(np.flipud(imgs)), axis=0).copy(), dtype=dtype) for imgs in images]
    # else:
    #     out_imgs = [torch.tensor(img, dtype=dtype) for img in images]
    out_imgs = [load(img) for img in files]

    return out_imgs, img_names


def get_grad_update(model):
    return [param.grad for param in model.parameters()]


def grad_norm(model):
    # parameters are given as type ParameterList
    M = torch.stack([param.grad for param in model.parameters()])
    mu = torch.mean(M, dim=-1, keepdim=True)
    om2 = torch.var(M, dim=-1, keepdim=True)
    new_grads = (M - mu.repeat(1, M.shape[1])) / torch.sqrt(om2.repeat(1, M.shape[1]) + torch.tensor(np.finfo(float).eps))

    for p, indx in zip(model.parameters(), torch.arange(len(M))):
        p.grad = new_grads[indx].clone()


def grad_norm_magnitude(model):
    new_grads = [param.grad / torch.norm(param.grad, 2) for param in model.parameters()]

    for p, indx in zip(model.parameters(), torch.arange(len(new_grads))):
        p.grad = new_grads[indx].clone()


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
                # (key, val) = line.split()
                # param_dict[key.strip(':')] = (val == 'True') if val in ['True', 'False'] else val
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


def vis_modules(model, tb_writer, epoch):
    for name, module in model.named_modules():
        if 'lstm' in name:
            for pn, weight in module.named_parameters():
                # want to plot weight and bias
                # if 'weight' in pn:
                tb_writer.add_histogram(f'{name}_{pn}', weight, epoch)


def freeze_layer(model):
    # blocks = model.sb._modules
    # add scale layer to enable training and unfreeze (train scale last)/ create knew dict to avoid adding scale to sb
    # blocks!
    names_blocks = [name for name in model.sb._modules]
    modules = [model.sb._modules[name] for name in names_blocks]
    blocks = collections.OrderedDict(zip(names_blocks, modules))
    # blocks['scale'] = model.scale_layer
    for layer in blocks:
        for pn, param in blocks[layer].named_parameters():
            param.requires_grad = False


def unfreeze_layer(model, indx, reset=True):
    # if model is ML, layer for each scale are stored in ModuleDict (else ModuleList). We need to extract all layers
    # and combine them in a OrderedDict to enable/disable single layer gradient updates
    if isinstance(model.sb, torch.nn.ModuleList):
        # blocks = model.sb._modules
        names_blocks = [name for name in model.sb._modules]
        modules = [model.sb._modules[name] for name in names_blocks]
        blocks = collections.OrderedDict(zip(names_blocks, modules))
    else:
        lvl_blocks = model.sb._modules
        blocks = collections.OrderedDict()
        for count, lvl in enumerate(lvl_blocks):
            # layer keys in all scales the same (created in ModuleList routine), create new keys to sum up layers in one
            # OrderedDict
            cdict = lvl_blocks[lvl]._modules
            new_keys = [f'{int(element) + model.num_layers*count}'for element in list(cdict.keys())]
            blocks.update(collections.OrderedDict(zip(new_keys, list(cdict.values()))))

    # add scale layer to enable training and unfreeze (train scale last) - here as last layer
    # blocks['scale'] = model.scale_layer

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


def plot_hist(wk, adds, parameter, tb_writer, epoch):
    m = torch.tensor(adds['T'][0].shape)
    dfm = deform.set_deformation('rigid', m.numel()).evaluate
    omega = parameter['objFct'].omega
    grid = Grid(1, omega, m)
    omega = omega.to(parameter['device'])
    inter = set_interpolater('linearFAIR', omega, m, grid.getSpacing().to(parameter['device']))

    num_iter = len(wk)

    all_T = []
    for ii in range(1, num_iter):
        xc = dfm.evaluate(wk[ii][0].unsqueeze(0), grid.getCellCentered().to(wk[-1].device), omega.unsqueeze(0))
        all_T = [*all_T, inter.interpolate(adds['T'][0], xc.to(parameter['device'])).squeeze()]

    caT = construct_large(all_T[:len(all_T) // 2], all_T[len(all_T) // 2:], hor_max=(num_iter - 1) // 2)
    # tb_writer.add_image('Deform_History', caT[0].unsqueeze(0).permute(0, 2, 1), epoch)
    fig_caT = plt.figure()
    plt.imshow(caT[0].detach().cpu(), cmap='gray')
    plt.axis('off')
    # plt.title(f'Deformation History - Epochs: {epoch}')
    # tb_writer.add_figure('Deform_History', fig_caT, epoch)

    # plot overlay of last image
    xc = dfm.evaluate(wk[-1][0].unsqueeze(0), grid.getCellCentered().to(wk[-1].device), omega.unsqueeze(0))
    imgT = inter.interpolate(adds['T'][0], xc.to(parameter['device'])).squeeze()

    ov_fig = plt.figure()
    plt.imshow(torch.tensor(overlay_imgs(imgT.detach().cpu(), adds['R'][0].squeeze().detach().cpu())))
    plt.axis('off')
    plt.show()


def plot_model(model, parameter, train_set):
    # need two s_adds, summary constructs two samples
    s_adds = next(iter(data.DataLoader(train_set, **{'batch_size': 2, 'shuffle': True})))[1]
    sum_model, info_model = summary(model, input_size=parameter['paraDim'].shape, device=parameter['device'],
                                    dtypes=[torch.double], params_fct=s_adds)
    count_model = countParameters(model)
    print(f'Param # {count_model}')
    f = open(os.path.join(parameter['dir_checkpoints'].split('checkpoints')[0], 'model.txt'), 'w')
    f.write(sum_model)
    f.close()


def verbose_tracker(model, wk, adds, fks, gk, his_grad_flow, his_bc_fks, his_bc_gk, iterates, minimizers,
                    full_mode=False):
    his_grad_flow = [*his_grad_flow, plot_grad_flow(model.named_parameters())]
    # save history of solver iterations
    # https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory
    if full_mode:
        # full mode is very memory expensive and should be only used for small samples and visualization purposes
        iterates = [*iterates, wk[-1].detach().cpu()]
        minimizers = [*minimizers, adds['x*'].detach().cpu()]
        # what should be mean over, batch(1), iteration(0) or entry(-1)?
        his_bc_fks.append(torch.mean(torch.stack(fks), dim=1).detach().cpu())
        his_bc_gk.append(torch.mean(torch.stack(gk), dim=1).detach().cpu())
    return his_grad_flow, iterates, minimizers, his_bc_fks, his_bc_gk


def verbose_plotter(parameter, lobj, epoch, gk, his_grad_flow, his_bc_fks, his_bc_gk, iterates, minimizers,
                    full_mode=False):
    stack2list = lambda stack: [stack[x] for x in range(stack.shape[0])]
    # plot gradient flow per epoch
    # flow_ep = os.path.join(parameter['dir_grad_flow'], f'ep_{epoch + 1}')
    # os.makedirs(flow_ep)
    for ii, flow_fig in enumerate(his_grad_flow):
        save_to_disk(flow_fig, f'grad_flow_e{epoch}_{ii}', parameter['dir_grad_flow'])

    if full_mode:
        # save history of solver
        his = plot_his(stack2list(torch.mean(torch.stack(his_bc_fks), dim=0)),
                       stack2list(torch.mean(torch.stack(his_bc_gk), dim=0)),
                       f'Training history epoch {epoch}')
        save_to_disk(his, f'his_plot_ep{epoch}', parameter['dir_his'])

        # save gradients of iterates / how to display? how deal with loop in data loader ? show just last batch
        for ii, grad_histo in enumerate(plot_histo_grads(gk)):
            # save_to_disk(grad_histo, f'grad_histo_{ii}', flow_ep)
            save_to_disk(grad_histo, f'grad_histo_{ii}', parameter['dir_grad_flow'])

        # plot loss values
        loss_fig_dir = os.path.join(parameter['dir_his'], 'loss')
        if not os.path.exists(loss_fig_dir):
            os.makedirs(loss_fig_dir)
        save_to_disk(lobj.plot_loss(torch.cat(iterates, dim=0), torch.cat(minimizers, dim=0), parameter['objFct']),
                     f'sample_losses_{epoch + 1}', loss_fig_dir)


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


def getOmega(m, h=1, invert=False, normalize=False):
    tmp = np.zeros((len(m), 2))
    msize = len(m)
    tmp[:, 1] = h * m

    if invert:
        omega = tmp.reshape(2 * msize)
        omg = omega.copy()
        omega[0] = omg[2]
        omega[1] = omg[3]
        omega[2] = omg[0]
        omega[3] = omg[1]
    else:
        omega = tmp.reshape(2 * msize)

    if normalize:
        # set omega to torch standard interval (hard coded)
        tmp[:, 0] = -1
        tmp[:, 1] = 1
        omega = tmp.reshape(2 * msize)

    return torch.tensor(omega, dtype=torch.int32)


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


def test_ml(data_path):
    # Test preprocessing functionalities
    images, _ = load_images(os.path.join('/data/image_registration', data_path), '.jpg')
    batch = 1
    # T, R = images[0][:, 0:-59], images[1][:, 0:-59]
    T, R = images[0], images[1]

    m = torch.tensor(T.shape)
    omega = getOmega(m, normalize=False)

    t_ml, _, omega, _ = getMultilevel(T, omega, m, batch)
    r_ml = getMultilevel(R, omega, m, batch)[0]

    for idx, (t, r) in enumerate(zip(t_ml, r_ml)):
        plt.figure()
        plt.subplot(121)
        plt.imshow(t.squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(r.squeeze(), cmap='gray')
        plt.axis('off')
        plt.suptitle(f'Level {idx}')
    plt.show()


if __name__ == '__main__':
    test_ml('Hands')
