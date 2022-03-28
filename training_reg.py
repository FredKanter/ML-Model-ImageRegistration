import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from collections import OrderedDict

from models_reg import init_model, MyDataParallel
from loss_func import set_loss, Scheduler
from reg_utils.tools import freeze_layer, unfreeze_layer, get_lr, prep_model_func
from setup_data import HdfDataset, set_generator
import tools as tools


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


def freeze_conditions(epoch, parameter):
    return parameter['freeze'][0] and epoch % parameter['freeze'][1] == 0 and epoch > 0


def train(parameter, restart=False, verbose=False):
    file = parameter['data']
    best_loss = np.inf

    # set up GPU use
    device_id = parameter['device']
    if parameter['device'] is None:
        parameter['device'] = torch.device('cpu')
    elif len(device_id) > 1:
        # a bit strange were to set device ids for DataParallel, set device default to largest GPU
        parameter['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        parameter['device'] = torch.device('cuda', parameter['device'][-1])

    # set up data generator
    gen_params = {'batch_size': parameter['batch'],
                  'shuffle': True}

    val_set = HdfDataset(file, 'val.h5')
    gen_params['shuffle'] = False
    val_gen = data.DataLoader(val_set, **gen_params)

    # Use on fly generator to use unlimited data resources
    dfm_params = tools.txt_to_dict(file, 'Configs.txt')
    name_gen = 'onfly' if parameter['data_mode']['onfly'] else 'hdf'
    train_set = set_generator(name_gen, file, parameter['data_mode']['nb'], parameter['data_mode']['set'],
                              parameter['objFct'].omega, parameter['deformation'], dfm_params,
                              noise=parameter['data_mode']['augment'][0], invert=parameter['data_mode']['augment'][1])
    train_gen = data.DataLoader(train_set, **gen_params)

    # clone net params so that they will not be updated and keep starting value for testing
    fcn_grad = prep_model_func(parameter)
    parameter['fcn'] = fcn_grad
    parameter['paraDim'] = next(iter(data.DataLoader(train_set,
                                                     **{'batch_size': 1, 'shuffle': True})))[0].to(parameter['device'])
    model = init_model(parameter['model'], **parameter)
    # train from best checkpoint from other model
    if parameter['snapshot'][0]:
        state = torch.load(os.path.join(parameter['snapshot'][1], 'model_best.pth.tar'))
        # check if module is prefix in state dict
        if all(['module' in key for key in list(state['state_dict'].keys())]):
            state = remove_DataParallel(state)
        # set state loding function to newly create checkpoint state dir
        torch.save(state, os.path.join(parameter['dir_checkpoints'], 'model_best.pth.tar'))
        print(f"---- Use snapshot from {parameter['snapshot'][1]} ----")
        model = tools.load_checkpoint(parameter['dir_checkpoints'], 'train', model)

    checkpoint = {}

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameter['lr'])
    shdobj = Scheduler(optimizer, **parameter['scheduler'])
    scheduler = shdobj._set_scheduler()
    scheduler_on = parameter['scheduler']['mode']

    refreeze = parameter['freeze'][2]
    if parameter['freeze'][0]:
        freeze_layer(model)
        # unfreeze first layer to enable learning
        unfreeze_layer(model, 0)
        limit, indx, stop_reset = 1, 1, False
    else:
        limit, indx, stop_reset = None, None, True

    # warp model in DataParallel if multiple GPUs should be used
    if len(device_id) > 1:
        print(f'Use {len(device_id)} GPUs')
        # add list of custom attributes to keep them accessable
        model = MyDataParallel(['sb', 'num_layers', 'scale_layer'], model, device_ids=device_id)

    model.to(parameter['device'])
    model.train()

    # set loss
    lobj = set_loss(parameter['loss'][1], parameter['loss'][0], verbose=verbose)
    fct_loss = lobj.evaluate

    # ---------------- Training ---------------------------------------
    epochs = parameter['epochs']
    lr = parameter['lr']
    print(f'\nStart Training: Epochs={epochs}, LR={lr}')

    path_pieces = parameter['dir_checkpoints'].split('/')
    writer = SummaryWriter(os.path.join('/results', 'tensorboard', path_pieces[2], path_pieces[3]))

    nb_layer_blocks = parameter['num_layer'] if isinstance(model.sb, torch.nn.ModuleList) else len(model.sb._modules)*parameter['num_layer']
    train_loss, val_loss, running_train, running_val = [], [], [], []
    train_objective, val_objective = [], []
    his_bc_fks, his_bc_gk, his_grad_flow = [], [], []
    iterates, minimizers = [], []

    for epoch in range(parameter['epochs']):
        if restart and epoch % parameter['freeze'][1] == 0 and epoch > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print(f'T={epoch}, reset optimizer')

        if freeze_conditions(epoch, parameter) and not stop_reset:
            if indx is None:
                print('Unfreeze all layers. Enable end-to-end ')
                stop_reset = True
            unfreeze_layer(model, indx, refreeze)
            limit = indx if indx is None else indx + 1
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            scheduler = Scheduler(optimizer, **parameter['scheduler'])._set_scheduler()
            indx = indx + 1 if indx is not None and indx + 1 <= nb_layer_blocks - 1 else None

        for w, adds in train_gen:
            w = w.to(parameter['device'])
            adds = tools.dict_to_device(adds, parameter['device'])

            optimizer.zero_grad()

            wk, fks, gk = model(w, params_fct=adds)

            loss = fct_loss(wk, adds['x*'], parameter['objFct'], m_ml=None, limit=limit, refreeze=refreeze,
                            params_fct=adds)
            loss.backward()

            optimizer.step()

            running_train.append(loss.item())
            writer.add_scalar("train/step_loss", loss.item(), epoch * len(train_gen) + len(running_train))

            # add objective energy to tensorboard
            writer.add_scalar('train/step_energy', fks[-1].mean().item(), epoch * len(train_gen) + len(running_train))
            train_objective = [*train_objective, fks[-1].mean().item()]


        for v, vadds in val_gen:
            with torch.no_grad():
                v = v.to(parameter['device'])
                vadds = tools.dict_to_device(vadds, parameter['device'])

                vk, vfks, vgk = model(v, params_fct=vadds)

                v_loss = fct_loss(vk, vadds['x*'], parameter['objFct'], m_ml=None, limit=limit, refreeze=refreeze,
                                  params_fct=vadds)

                running_val.append(v_loss.item())
                writer.add_scalar('val/step_loss', v_loss.item(), epoch * len(val_gen) + len(running_val))

                # add objective for val set
                writer.add_scalar('val/step_energy', vfks[-1].mean().item(), epoch * len(val_gen) + len(running_val))
                val_objective = [*val_objective, vfks[-1].mean().item()]

        old_lr = get_lr(optimizer)
        if scheduler_on:
            if shdobj.name == 'ReduceOnPlateau':
                scheduler.step(torch.tensor(val_loss).mean())
            else:
                scheduler.step()
        lr = get_lr(optimizer)
        if old_lr > lr:
            print(f'Learning Rate reduced to {lr}')
        writer.add_scalar("lr", lr, epoch)

        train_loss.append(np.mean(running_train))
        val_loss.append(np.mean(running_val))
        writer.add_scalar("train/mean_loss", train_loss[epoch], epoch)
        writer.add_scalar('val/mean_loss', val_loss[epoch], epoch)
        running_train.clear()
        running_val.clear()

        # add summed energy over all samples
        writer.add_scalar('train/mean_energy', sum(train_objective)/len(train_objective), epoch)
        writer.add_scalar('val/mean_energy', sum(val_objective)/len(val_objective), epoch)
        train_objective.clear()
        val_objective.clear()

        print(f'Epoch: {epoch + 1}   Train: {train_loss[epoch]:.5f}   Val: {val_loss[epoch]:.5f}')

        # save checkpoint, monitor val loss to get best model
        is_best = val_loss[epoch] < best_loss
        best_loss = min(val_loss[epoch], best_loss)

        checkpoint['state_dict'] = model.state_dict()
        checkpoint['best_loss'] = best_loss
        checkpoint['optimizer'] = optimizer.state_dict()

        tools.save_checkpoint(checkpoint, is_best, checkpoint_dir=parameter['dir_checkpoints'],
                              filename=f'checkpoint_e{epoch + 1}.pth.tar')

    writer.flush()
    writer.close()

    return model, [train_loss, val_loss]
