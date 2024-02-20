import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpBatchLinNet(nn.Module):
    def __init__(self, exp_bs, in_dim, out_dim, device, tch_dtype, init=False):
        super(ExpBatchLinNet, self).__init__()
        self.exp_bs = exp_bs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Parameter(torch.zeros((exp_bs, in_dim, out_dim), device=device, dtype=tch_dtype))
        self.fc1_bias = torch.nn.Parameter(torch.zeros((exp_bs, 1, out_dim), device=device, dtype=tch_dtype))

        if init:
            torch.nn.init.normal_(self.fc1)
            torch.nn.init.normal_(self.fc1_bias)

    def forward(self, x):
        assert x.ndim == 3, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[0] == self.exp_bs, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[2] == self.in_dim, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        x = x.matmul(self.fc1) + self.fc1_bias
        return x
