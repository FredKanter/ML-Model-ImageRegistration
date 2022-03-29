import torch
from torch import nn as nn
from torch.nn import functional as F


def second_order_derivative_conv(uc, h):
    B, C, H, W = uc.size()
    h_conv = h.clone().view(1, C, 1, 1).repeat(B, 1, H, W)

    # C often one for alessa (if only one u has to be determined), H W would define grid in two dim (vgl. torch)
    uc_h = F.pad(uc, (0, 0, 1, 1), mode='replicate')
    uc_w = F.pad(uc, (1, 1, 0, 0), mode='replicate')

    conv_d11 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(3, 1), padding=(0, 0), bias=False, groups=C)
    conv_d22 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 3), padding=(0, 0), bias=False, groups=C)

    conv_d11.weight.requires_grad = False
    conv_d22.weight.requires_grad = False

    conv_d11.weight.data = torch.Tensor([1, -2, 1]).type(uc.dtype).view(1, 1, 3, 1).repeat(C, 1, 1, 1)
    conv_d22.weight.data = torch.Tensor([1, -2, 1]).type(uc.dtype).view(1, 1, 1, 3).repeat(C, 1, 1, 1)

    conv_d11 = conv_d11.to(uc.device)
    conv_d22 = conv_d22.to(uc.device)
    h_conv = h_conv.to(uc.device)

    d11u = conv_d11(uc_h)
    d22u = conv_d22(uc_w)

    return d11u/(h_conv**2), d22u/(h_conv**2)


def second_order_derivative_conv3d(uc, h):
    B, C, H, W, D = uc.size()
    h_conv = h.clone().view(1, C, 1, 1, 1).repeat(B, 1, D, H, W)

    uc_h = F.pad(uc, (0, 0, 0, 0, 1, 1), mode='replicate')
    uc_w = F.pad(uc, (0, 0, 1, 1, 0, 0), mode='replicate')
    uc_d = F.pad(uc, (1, 1, 0, 0, 0, 0), mode='replicate')

    conv_h = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(3, 1, 1), padding=(0, 0, 0), bias=False, groups=C)
    conv_w = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(1, 3, 1), padding=(0, 0, 0), bias=False, groups=C)
    conv_d = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(1, 1, 3), padding=(0, 0, 0), bias=False, groups=C)

    conv_h.weight.requires_grad = False
    conv_w.weight.requires_grad = False
    conv_d.weight.requires_grad = False

    conv_h.weight.data = torch.Tensor([1, -2, 1]).type(uc.dtype).view(1, 1, 3, 1, 1).repeat(C, 1, 1, 1, 1)
    conv_w.weight.data = torch.Tensor([1, -2, 1]).type(uc.dtype).view(1, 1, 1, 3, 1).repeat(C, 1, 1, 1, 1)
    conv_d.weight.data = torch.Tensor([1, -2, 1]).type(uc.dtype).view(1, 1, 1, 1, 3).repeat(C, 1, 1, 1, 1)

    conv_h = conv_h.to(uc.device)
    conv_w = conv_w.to(uc.device)
    conv_d = conv_d.to(uc.device)
    h_conv = h_conv.to(uc.device)

    d11u = conv_h(uc_h)
    d22u = conv_w(uc_w)
    d33u = conv_d(uc_d)

    return d11u/(h_conv**2), d22u/(h_conv**2), d33u/(h_conv**2)


def second_order_derivative(uc, h):
    uc_h = F.pad(uc.clone(), (0, 0, 1, 1), mode='replicate')
    uc_w = F.pad(uc.clone(), (1, 1, 0, 0), mode='replicate')

    # d11u1
    d11u1 = (uc_h[:, 0, 2:, :] - 2*uc[:, 0, :, :] + uc_h[:, 0, :-2, :]) / h[:1]**2
    d12u1 = (uc_w[:, 0, :, 2:] - 2*uc[:, 0, :, :] + uc_w[:, 0, :, :-2]) / h[:1]**2

    # d22u2
    d21u2 = (uc_h[:, 1, 2:, :] - 2*uc[:, 1, :, :] + uc_h[:, 1, :-2, :]) / h[1:2]**2
    d22u2 = (uc_w[:, 1, :, 2:] - 2*uc[:, 1, :, :] + uc_w[:, 1, :, :-2]) / h[1:2]**2

    return d11u1, d12u1, d21u2, d22u2


def finite_differences(uc, h):
    # uc[1:] - uc[:-1] for each dim (x, y, z) of grid x (two dims for each image dimension)
    uc_h = F.pad(uc.clone(), (0, 0, 0, 1), mode='replicate')
    uc_w = F.pad(uc.clone(), (0, 1, 0, 0), mode='replicate')

    # du1
    d1u1 = (uc_h[:, 0, 1:, :] - uc_h[:, 0, :-1, :]) / h[:1]
    d2u1 = (uc_w[:, 0, :, 1:] - uc_w[:, 0, :, :-1]) / h[:1]

    # du2
    d1u2 = (uc_h[:, 1, 1:, :] - uc_h[:, 1, :-1, :]) / h[1:2]
    d2u2 = (uc_w[:, 1, :, 1:] - uc_w[:, 1, :, :-1]) / h[1:2]

    # compose u1 and u2
    d_uc = torch.stack((d1u1, d1u2, d2u1, d2u2), dim=1)
    return d_uc


def finite_differences_conv(uc, h):
    # version with conv kernel (favourable for element in network)/ should work similar to narrow
    B, C, H, W = uc.shape
    h_m = h.clone().view(1, C, 1, 1).repeat(B, 1, H, W)

    uc_h = F.pad(uc.clone(), (0, 0, 0, 1), mode='replicate')
    uc_w = F.pad(uc.clone(), (0, 1, 0, 0), mode='replicate')

    conv_d1 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(2, 1), padding=(0, 0), bias=False, groups=C)
    conv_d2 = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=(1, 2), padding=(0, 0), bias=False, groups=C)

    conv_d1.weight.requires_grad = False
    conv_d2.weight.requires_grad = False

    conv_d1.weight.data = torch.Tensor([-1, 1]).type(uc.dtype).view(1, 1, 2, 1).repeat(C, 1, 1, 1)
    conv_d2.weight.data = torch.Tensor([-1, 1]).type(uc.dtype).view(1, 1, 1, 2).repeat(C, 1, 1, 1)

    d1u = conv_d1(uc_h) / h_m
    d2u = conv_d2(uc_w) / h_m

    # combine to one tensor
    d_uc = torch.cat((d1u, d2u), dim=1)
    return d_uc

