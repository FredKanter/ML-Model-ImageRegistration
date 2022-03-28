import torch
import torch.nn as nn
import numpy as np
import time
import math

import tools as tools
from grids import Grid
import distances as dist


class MyDataParallel(nn.DataParallel):
    # Want to access some variables in model, which are blocked in DataParallel
    # overwriting the getattr method of nn.Module
    def __init__(self, my_methods, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mymethods = my_methods

    def __getattr__(self, name):
        if name in self._mymethods:
            return getattr(self.module, name)

        else:
            return super().__getattr__(name)


class FuseLayer(nn.Module):
    def __init__(self, lenFcn, imgDims, ksz, stride):
        super(FuseLayer, self).__init__()
        self.ksz = ksz
        self.stride = stride
        self.pad = 0
        self.imgDims = imgDims
        layers = {'2': nn.Conv2d, '3': nn.Conv3d}
        self.fuse = layers[str(imgDims)]((lenFcn * 2 + 1)*imgDims, imgDims, kernel_size=ksz,
                                         stride=stride, padding_mode='reflect')
        self.init_weigths()
        # disable learning for fuse layer
        self.fuse.weight.requires_grad = False
        self.fuse.bias.requires_grad = False

    def init_weigths(self):
        with torch.no_grad():
            # use fixed mean over filter kernel
            self.fuse.weight.data = torch.ones(self.fuse.weight.data.shape)/self.ksz**self.imgDims
            if self.fuse.bias is not None:
                nn.init.constant_(self.fuse.bias, 0.0)

    def calc_conv(self, isz):
        return int((isz + 2 * self.pad - 1 * (self.ksz - 1) - 1) / self.stride + 1)

    def forward(self, x):
        return self.fuse(x).flatten(-(self.imgDims+1), -1).squeeze(1)


class FuseLayerAffine(nn.Module):
    def __init__(self, indim, hidden_sz):
        super(FuseLayerAffine, self).__init__()
        self.reshape = lambda x: x.view(x.shape[0], -1)
        self.indim = indim
        self.hidden_sz = hidden_sz
        if indim != hidden_sz:
            self.reduce = True
            self.fuse = nn.Linear(indim, hidden_sz)
            self.init_weights()
            # disable learning for fuse layer
            self.fuse.weight.requires_grad = False
            self.fuse.bias.requires_grad = False
        else:
            self.reduce = False

    def init_weights(self):
        with torch.no_grad():
            if not self.hidden_sz % self.indim == 0:
                raise RuntimeError('Not mod!')
            # implemented linear layer id mapping
            self.fuse.weight.data = torch.tensor(np.kron(np.ones((self.hidden_sz // self.indim, 1)), np.eye(self.indim))
                                                 + np.random.normal(0.0, 0.01, size=(self.hidden_sz, self.indim)))
        nn.init.constant_(self.fuse.bias, 0.0)

    def forward(self, x):
        x = self.reshape(x)
        if self.reduce:
            x = self.fuse(x)
        return x


class ScaleLayer(nn.Module):
    def __init__(self, hidden_sz, out_dim, imgDims, nb_fcn, ksz, stride, h, pad=0, skip_grad=False):
        super(ScaleLayer, self).__init__()
        self.hidden_sz = hidden_sz
        self.h = h.clone().detach().requires_grad_(True).unsqueeze(-1).unsqueeze(-1)
        self.ksz = ksz
        self.stride = stride
        self.pad = pad
        self.out = out_dim
        self.imgDims = imgDims
        self.nb_fcn = nb_fcn
        self.skip_grad = skip_grad
        layers = {'2': nn.Conv2d, '3': nn.Conv3d}

        if skip_grad:
            self.conv1 = layers[str(imgDims)](nb_fcn * imgDims, imgDims, 1)
            self.conv2 = layers[str(imgDims)](2 * imgDims, imgDims, 1)
            self.tanh = nn.Tanh()
            self.init_skip_weights()

    def init_skip_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        nn.init.constant_(self.conv2.bias, 0.0)

    def skip_connection(self, x, g):
        B, HW, img_size = self.calc_sizes(x)
        x = x.view((B, self.imgDims, *self.imgDims*[self.imgSz]))
        g = self.conv1(g.reshape((B, self.nb_fcn*self.imgDims, *self.imgDims*[self.imgSz])))
        x = self.conv2(torch.cat((x, g), 1))
        x = self.tanh(x)
        return x.flatten(-(self.imgDims+1), -1)

    def calc_sizes(self, x):
        B, HW = x.shape
        img_size = round(math.pow(HW // self.imgDims, 1./self.imgDims))
        return B, HW, img_size

    def forward(self, x, *args):
        if len(args) > 0 and self.skip_grad:
            x = self.skip_connection(x, args[0])
        return x


class ScaleLayerAffine(ScaleLayer):
    def __init__(self, hidden_sz, out_dim, imgDims, nb_fcn, h, skip_grad=False):
        super(ScaleLayerAffine, self).__init__(hidden_sz, out_dim, imgDims, nb_fcn, None, None, h, skip_grad=skip_grad)
        self.scale = nn.Linear(hidden_sz, out_dim)
        self.out = out_dim
        self.tanh = nn.Tanh()
        self.init_scale_weigths()

        # disable learning for scale layer
        self.scale.weight.requires_grad = False
        self.scale.bias.requires_grad = False

    def init_scale_weigths(self):
        with torch.no_grad():
            if not self.hidden_sz % self.out == 0:
                raise RuntimeError('Not mod!')
            # implemented linear layer id mapping
            self.scale.weight.data = torch.tensor(np.kron(np.eye(self.out), np.ones((1, self.hidden_sz//self.out))/(self.hidden_sz//self.out)))
        nn.init.constant_(self.scale.bias, 0.0)
        if self.skip_grad:
            nn.init.normal_(self.skip.weight.data, 0, 0.01)
            nn.init.constant_(self.skip.bias, 0.0)

    def skip_connection(self, x, g):
        return self.skip(torch.cat((x, g.squeeze(1)), 1))

    def forward(self, x, *args):
        x = self.scale(x)

        if len(args) > 0 and self.skip_grad:
            x = self.skip_connection(x, args[0].clone().detach())

        return x


class LSTMNetBlock(nn.Module):
    def __init__(self, hidden_sz):
        super(LSTMNetBlock, self).__init__()
        self.lstm_in = nn.LSTMCell(hidden_sz, hidden_sz)
        self.init_weigths()

    def init_weigths(self):
        # Initialize weights and bias so that first update is identity mapping
        for name, param in self.lstm_in.named_parameters():
            if 'bias' in name:
                # set hidden bias to small values to force first step to result in minor update
                with torch.no_grad():
                    dim = param.shape[0]//4
                    b_hi = torch.ones(dim) * -4
                    b_hf = torch.ones(dim) * -4

                    b_hg = torch.zeros(dim)
                    b_ho = torch.zeros(dim)
                    param.copy_(torch.cat((b_hi, b_hf, b_hg, b_ho), 0))

            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, hc1):
        h_0, c_0 = self.lstm_in(x, (hc1[0], hc1[1]))
        return c_0, (h_0, c_0)


class BaseNet(nn.Module):
    def __init__(self, x, fcn_grad, k_iter, num_layers, block_name, fcn_object, normalize=False, dtype=torch.double):
        super(BaseNet, self).__init__()
        # create one class to ensemble all solver blocks / pass list of functions to combine in model (e.g distance/reg)
        if not isinstance(fcn_grad, list):
            # fcn should be list of functions
            fcn_grad = [fcn_grad]
        self.name = f'Base'
        self.block_name = block_name
        self.k = k_iter
        self.num_layers = num_layers
        self.normalize = normalize
        self.dtype = dtype
        self.dimOut = x.size(1)
        self.imgDims = fcn_object.m.numel()
        self.imgSz = round(math.pow(self.dimOut // self.imgDims, 1./self.imgDims))
        self.fcn = fcn_grad
        self.fac_hidden = 4
        # distinct between npir and pir to use fuse layer
        self.non_para = isinstance(fcn_object.dist, dist.NPIR)

        # define common layers
        in_affine = self.dimOut * (1 + 2*len(fcn_grad))
        if self.non_para:
            self.fuse_layer = FuseLayer(len(fcn_grad), self.imgDims, 1, 1)
            self.hidden_sz = (self.fuse_layer.calc_conv(self.imgSz) ** self.imgDims) * self.imgDims
            self.scale_layer = ScaleLayer(self.hidden_sz, self.dimOut, self.imgDims, len(fcn_grad), 1, 1, fcn_object.h, skip_grad=False)
        else:
            self.fuse_layer = FuseLayerAffine(in_affine, in_affine * self.fac_hidden)
            self.hidden_sz = in_affine*self.fac_hidden
            self.scale_layer = ScaleLayerAffine(self.hidden_sz, self.dimOut, self.imgDims, len(fcn_grad), fcn_object.h, skip_grad=False)

    @staticmethod
    def _expand_list(x_k, gx, fx, xkl, gkl, fkl, timed=False):
        # for DataParallel all outputs (xk, gk, fk) has to be on same device, this can get quite large if his
        # (gk, fk) are on gpu
        xkl = [*xkl, x_k]
        if len(fx.shape) > 1:
            fx = torch.sum(fx, dim=1)
        if timed:
            fkl = [*fkl, (fx.detach().clone(), time.time())]
        else:
            fkl = [*fkl, fx.detach().clone()]
        gkl = [*gkl, gx.detach().clone()]
        return xkl, gkl, fkl

    def _prepare_input(self, xk, gk, fk):
        # check if fk consists of multiple parts, no dummy dim needed
        if not len(gk.shape) == len(fk.shape):
            fk = fk.unsqueeze(-1).repeat(1, 1, gk.shape[-1])
        if not len(gk.shape) == len(xk.shape):
            xk = xk.unsqueeze(1)

        if self.normalize:
            (gk, gk_norm), (fk, fk_norm) = preprocess_input(xk, gk, fk)

        x = torch.cat((xk, gk, fk), 1)
        # x = gk

        # if fused layer in network (npir approaches), reshape data to fit 2D conv on grid, grads and fcts
        if self.non_para:
            B, C, HW = x.shape
            x = x.reshape((B, C*self.imgDims, *self.imgDims*[self.imgSz]))
        return x

    def _eval_func(self, x_k, **params):
        fx, gx = [], []
        for func in self.fcn:
            fxc, gxc = func(x_k, **params)
            fx, gx = [*fx, fxc], [*gx, gxc]
        return torch.stack(fx, dim=1), torch.stack(gx, dim=1)

    def forward(self, x, timed=False, **kwargs):
        fk_list, gk_list, xk_list = [], [], []
        x_k = x

        fx, gx = self._eval_func(x_k, **kwargs['params_fct'])
        x = self._prepare_input(x_k, gx, fx)
        xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)

        # pass through layer
        for k in range(self.k):
            x = self.fuse_layer(x)
            delta = self.scale_layer(x)

            x_k = x_k + delta

            fx, gx = self._eval_func(x_k, **kwargs['params_fct'])
            xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)
            x = self._prepare_input(x_k, gx, fx)

        xk_out = xk_list.copy()
        xk_list.clear()
        fk_out = fk_list.copy()
        fk_list.clear()
        gk_out = gk_list.copy()
        gk_list.clear()
        return xk_out, fk_out, gk_out


class MetaNet(BaseNet):
    def __init__(self, x, fcn_grad, k_iter, num_layers, block_name, fcn_object, normalize=False, dtype=torch.double):
        super(MetaNet, self).__init__(x, fcn_grad, k_iter, num_layers, block_name, fcn_object, normalize, dtype)
        # create one class to ensemble all solver blocks / pass list of functions to combine in model (e.g distance/reg)
        self.name = f'Meta-{block_name}'
        self.block_moduls = {'LSTMNet': LSTMNetBlock, 'GRUNet': GRUNetBlock, 'DenseNet': DenseBlock}
        if block_name not in self.block_moduls.keys():
            raise RuntimeError(f'{block_name} is not implemented base block')
        else:
            block = self.block_moduls[block_name]
        self.sb = nn.ModuleList([block(self.hidden_sz) for _ in range(self.num_layers)])

    def init_hidden_state(self, x):
        rng = np.random.RandomState(45)
        hc1 = (torch.tensor(rng.uniform(0, 1, (x.size(0), self.hidden_sz)), dtype=self.dtype).to(x.device),
               torch.tensor(rng.uniform(0, 1, (x.size(0), self.hidden_sz)), dtype=self.dtype).to(x.device))
        return hc1

    def forward(self, x, timed=False, **kwargs):
        fk_list, gk_list, xk_list = [], [], []
        x_k = x

        fx, gx = self._eval_func(x_k, **kwargs['params_fct'])
        x = self._prepare_input(x_k, gx, fx)
        xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)
        if self.normalize:
            gx, gx_norm = preprocess_grad_loss(gx)

        hc1 = self.init_hidden_state(x)

        for i, l in enumerate(self.sb):
                x = self.fuse_layer(x)
                x, hc1 = self.sb[i](x, hc1)
                delta = self.scale_layer(x, gx) if self.scale_layer.skip_grad else self.scale_layer(x)

                x_k = x_k + delta

                fx, gx = self._eval_func(x_k, **kwargs['params_fct'])
                xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)
                x = self._prepare_input(x_k, gx, fx)

        xk_out = xk_list.copy()
        xk_list.clear()
        fk_out = fk_list.copy()
        fk_list.clear()
        gk_out = gk_list.copy()
        gk_list.clear()

        return xk_out, fk_out, gk_out


class ML_MetaNet(BaseNet):
    def __init__(self, x, fcn_grad, k_iter, num_layers, block_name, fcn_object, normalize=False, dtype=torch.double):
        super(ML_MetaNet, self).__init__(x, fcn_grad, k_iter, num_layers, block_name, fcn_object, normalize, dtype)
        # create one class to ensemble all solver blocks / pass list of functions to combine in model (e.g distance/reg)
        self.name = f'ML-Meta-{block_name}'
        self.block_moduls = {'LSTMNet': LSTMNetBlock, 'GRUNet': GRUNetBlock, 'DenseNet': DenseBlock}
        if block_name not in self.block_moduls.keys():
            raise RuntimeError(f'{block_name} is not implemented base block')
        else:
            block = self.block_moduls[block_name]
        self.maxM = fcn_object.m
        self.omega = fcn_object.omega
        self.fcn_object = fcn_object
        self.maxLevel = torch.ceil(torch.log2(torch.min(self.maxM.type(torch.float32))))
        # min level in getMultiLevel default to 4
        self.minLevel = 4
        # Nested module list to enable multiple layer per level
        self.sb = nn.ModuleDict()
        self.lvl_scales = torch.arange(self.maxLevel - self.minLevel, -1, -1).type(torch.int)
        for lvl in self.lvl_scales:
            lvl_block = nn.ModuleList([block(self.hidden_sz) for _ in range(self.num_layers)])
            self.sb[f'lvl_{lvl}'] = lvl_block

    def init_hidden_state(self, x):
        rng = np.random.RandomState(45)
        hc1 = (torch.tensor(rng.uniform(0, 1, (x.size(0), self.hidden_sz)), dtype=self.dtype).to(x.device),
               torch.tensor(rng.uniform(0, 1, (x.size(0), self.hidden_sz)), dtype=self.dtype).to(x.device))
        return hc1

    def init_scales(self, **kwargs):
        in_params = kwargs['params_fct']
        batch = in_params['T'].shape[0]
        T_scales, m_scales, _, _ = tools.getMultilevel(in_params['T'], self.omega, self.maxM, batch, maxLevel=self.maxLevel)
        R_scales = tools.getMultilevel(in_params['R'], self.omega, self.maxM, batch, maxLevel=self.maxLevel)[0]
        return T_scales, R_scales, m_scales

    def update_to_scale(self, m_ML):
        level_grid = Grid(1, self.omega, m_ML)
        # do reset for new objective
        self.fcn_object.reset(self.omega, level_grid.getSpacing(), m_ML, level_grid.getCellCentered().clone())
        self.fcn = tools.prep_model_func({'objFct': self.fcn_object})

    def forward(self, x, timed=False, **kwargs):
        # adapted form of MetaNet, but instead of multiple steps k use multiple scales
        ml_T, ml_R, m_scales = self.init_scales(**kwargs)
        fk_list, gk_list, xk_list = [], [], []
        x_k = x

        hc1 = self.init_hidden_state(x)

        for lvl in self.lvl_scales:
            lvl_params = {'T': ml_T[lvl].to(x.device), 'R': ml_R[lvl].to(x.device)}
            lvl_block_layer = self.sb[f'lvl_{lvl}']
            self.update_to_scale(m_scales[lvl])

            fx, gx = self._eval_func(x_k, **lvl_params)
            x = self._prepare_input(x_k, gx, fx)
            xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)
            if self.normalize:
                gx, gx_norm = preprocess_grad_loss(gx)

            for block_layer in lvl_block_layer:
                x = self.fuse_layer(x)
                x, hc1 = block_layer(x, hc1)
                delta = self.scale_layer(x, gx) if self.scale_layer.skip_grad else self.scale_layer(x)

                x_k = x_k + delta

                fx, gx = self._eval_func(x_k, **lvl_params)
                xk_list, gk_list, fk_list = self._expand_list(x_k, gx, fx, xk_list, gk_list, fk_list, timed=timed)
                x = self._prepare_input(x_k, gx, fx)


        xk_out = xk_list.copy()
        xk_list.clear()
        fk_out = fk_list.copy()
        fk_list.clear()
        gk_out = gk_list.copy()
        gk_list.clear()

        return xk_out, fk_out, gk_out


def preprocess_grad_loss(x, epsilon=1e-8):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).repeat(1, 1, x.shape[-1])
    return x / (x_norm + epsilon), x_norm


def preprocess_input(xk, grad, fct):
    grad_prep, grad_norm = preprocess_grad_loss(grad)
    fct_prep, fct_norm = preprocess_grad_loss(fct)
    return (grad_prep, grad_norm), (fct_prep, fct_norm)


def repackage_hidden(h):
    # Back Propagation through time (BBT) - clear history of hidden states
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def init_model(name, **kwargs):
    if name in ['LSTMNet'] and not kwargs['ML_training']:
        return MetaNet(kwargs['paraDim'], kwargs['fcn'], kwargs['solver'].iter, kwargs['num_layer'],
                       name, kwargs['objFct'], normalize=kwargs['normalize']).double()
    elif name in ['LSTMNet'] and kwargs['ML_training']:
        return ML_MetaNet(kwargs['paraDim'], kwargs['fcn'], kwargs['solver'].iter, kwargs['num_layer'],
                          name, kwargs['objFct'], normalize=kwargs['normalize']).double()
    else:
        print(f'Model {name} not implemented. Use BaseModel instead.')
        return BaseNet(kwargs['paraDim'], kwargs['fcn'], kwargs['solver'].iter, kwargs['num_layer'],
                       'base', kwargs['objFct'].h, normalize=kwargs['normalize']).double()
