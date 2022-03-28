import torch
import numpy as np
import h5py
import os
import time
import datetime
import dill

from optimization import set_solver
import deformations as deform
import tools as tools
from grids import Grid
from distances import set_distance, SSD, NGF
from regularizer import set_regularizer
from objectives import ObjectiveFunction
from save_results import write_file
from writer_csv import to_csv
from prediction_reg import predict
from training_reg import train


def do_run(settings):
    torch.manual_seed(settings['seed'])
    res_path = settings['results']
    soltmp = settings['solver']
    obj_fctn = settings['objFct']
    train_params = {**settings}

    # create dir for results and checkpoints in train
    time_stamp = os.path.join(datetime.date.today().isoformat(), datetime.datetime.now().strftime('%H_%M_%S'))
    res_path = os.path.join(res_path, time_stamp)

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    print(f"---- results path: {res_path} ----")

    # save objective function later needed in visualizing results
    with open(os.path.join(res_path, 'objective'), 'wb') as dill_file:
        dill.dump(obj_fctn, dill_file)

    parameters = {'Solver': soltmp.name,
                  'Model': settings['model'],
                  'num_layer': settings['num_layer'],
                  'normalize': settings['normalize'],
                  'regularize': settings['regularize'],
                  'freeze': str(settings['freeze'][0]) + '_' + str(settings['freeze'][1]) + '_' + str(settings['freeze'][2]),
                  'grad_norm': str(settings['grad_norm'][0]) + '_' + str(settings['grad_norm'][1]),
                  'ML_model': settings['ML_training'],
                  'Loss': settings['loss'][1] + '_' + settings['loss'][0],
                  'Iterations': soltmp.iter.copy(),
                  'Epochs': settings['epochs'],
                  'Batch-Size': settings['batch'],
                  'LR': settings['lr'],
                  'LR_restart': settings['restart'],
                  'Scheduler': str(settings['scheduler']['mode']) + '_' + settings['scheduler']['name'] + '_' + str(
                      settings['scheduler']['param']),
                  'Deformation': settings['deformation'].name,
                  'Data': settings['data'].split('/')[-1],
                  'Data_Onfly': str(settings['data_mode']['onfly']) + '_' + str(settings['data_mode']['nb']) + '_' +
                                settings['data_mode']['set'],
                  'Seed': settings['seed'],
                  'Augment': str(settings['data_mode']['augment'][0]) + '_' + str(settings['data_mode']['augment'][1]),
                  'solver_reg': settings['solver_reg'][0] + '_' + str(settings['solver_reg'][1]),
                  'ML': settings['ML'],
                  'nb_level': settings['nb_level'],
                  'Test_Iterations': settings['test_iter']}
    if 'BFGS' in soltmp.name:
        parameters['numBFGSVec'] = soltmp.maxBFGS
        parameters['LR-BFGS'] = soltmp.lr
        parameters['LineSearch'] = soltmp.LS
    if settings['snapshot'][0]:
        parameters['Snapshot'] = settings['snapshot'][-1]
    write_file(parameters, 'parameters', res_path)

    # make dir for checkpoints to use best model for testing later on
    dir_checkpoints = os.path.join(res_path, 'checkpoints')
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints)
    train_params['dir_checkpoints'] = dir_checkpoints

    # make dir for solver history plots
    dir_his = os.path.join(res_path, 'solver_history')
    if not os.path.exists(dir_his) and settings['verbose']:
        os.makedirs(dir_his)
    train_params['dir_his'] = dir_his

    # make dir for gradient flow
    dir_grad_flow = os.path.join(res_path, 'grad_flow')
    if not os.path.exists(dir_grad_flow) and settings['verbose']:
        os.makedirs(dir_grad_flow)
    train_params['dir_grad_flow'] = dir_grad_flow

    # ---------------- Training ---------------------------------------
    start_init = time.time()
    model, losses = train(train_params, settings['restart'], settings['verbose'])
    end_init = time.time()

    hours, minutes, seconds = tools.calc_time(start_init, end_init)
    print(f'Train loss: {losses[0][-1]:.4f}, Test loss: {losses[1][-1]:.4f}  '
          f'Time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}')

    # add runtime of training to parameters
    file = open(os.path.join(res_path, 'parameters.txt'), 'a')
    file.write(f'Time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}\n')
    file.close()

    # ---------------- Evaluation ---------------------------------------
    # reset solver with new iteration in test
    if settings['test_iter'] != soltmp.iter:
        soltmp.set_iter(settings['test_iter'])

    res_metrics = predict(model, soltmp, train_params, combined=True)

    # write run to csv-file / for energy and loss we are only interested in the last entry, not in trajectory
    params = {'Run': time_stamp,
              'Model': settings['model'],
              'Objective': settings['deformation'].name,
              'Data': settings['data'].split('/')[-1] if not settings['data_mode']['onfly'] else 'OnFly_' + str(settings['data_mode']['nb']),
              'BS': settings['batch'],
              'LR': settings['lr'],
              'Epochs': settings['epochs'],
              'Time': f'{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}',
              'Mean Loss (plain)': torch.mean(torch.stack(res_metrics['Diff'][0])[:, -1]).item(),
              'Mean_Loss (learn)': torch.mean(torch.stack(res_metrics['Diff'][1])[:, -1]).item(),
              'Mean Energy (plain)': torch.mean(res_metrics['Instance_energy'][0][:, -1]).item(),
              'Mean Energy (learn)': torch.mean(res_metrics['Instance_energy'][1][:, -1]).item()}
    to_csv('/results', [params], settings['name_csv'])

    # write metrics to disk
    tools.save_run(res_path, res_metrics, losses)
    torch.cuda.empty_cache()


# Give do_expermients parameters to pass from yaml to use bash script for conducting experiments
def do_experiments(input_params):
    num_experiments = input_params['num_experiments']
    pool_epoch = input_params['epochs']
    pool_lr = input_params['learning_rates']
    pool_bs = input_params['batch_sizes']
    pool_inner_iter = input_params['num_solver_steps']
    def_type = input_params['objective'][1]

    # additional parameter for solver settings
    enable_ml = input_params['ML_flag']
    solver_name = input_params['solver_name']
    solver_settings = input_params['solver_set']

    for k in range(num_experiments):
        # ---------------- Setup ---------------------------------------
        with h5py.File(os.path.join(input_params['data'], 'train.h5'), 'r') as fp:
            m = torch.tensor(fp['T'][0].shape)

        # setup for network hyperparameters
        batch = int(np.random.choice(pool_bs))
        input_params['epochs'] = np.random.choice(pool_epoch)
        lr = np.random.choice(pool_lr)
        iterations = np.random.choice(pool_inner_iter)

        # set up objective and static parameter like grid, h, omega
        omega = tools.txt_to_dict(input_params['data'], 'Configs.txt')['omega']

        grid = Grid(1, omega, m)
        dfm = deform.set_deformation(def_type, m.numel())
        regularizer = set_regularizer(input_params['solver_reg'][0], dfm, omega, m, grid.getSpacing(),
                                      grid.getCellCentered())
        distance = set_distance(input_params['objective'][0], dfm, SSD, omega, m, grid.getSpacing(),
                                grid.getCellCentered())

        # newly created objective (combination of distance and reg/ should be separated for networks)
        fcn = ObjectiveFunction(distance, regularizer, input_params['solver_reg'][1])

        # params for inner objective (registration)
        soltmp = set_solver(solver_name, iterations, ml_flag=enable_ml, **solver_settings)

        run_settings = {'results': '/results',
                        'solver': soltmp,
                        'objFct': fcn,
                        'deformation': deform.set_deformation(def_type, m.numel()),
                        'lr': lr,
                        'restart': input_params['lr_restart'],
                        'batch': batch,
                        'device': input_params['device'],
                        'ML': enable_ml,
                        'nb_level': (soltmp.sol.ml_max_lvl - 4 + 1).item()}

        run_settings = {**run_settings, **input_params}

        do_run(run_settings)
