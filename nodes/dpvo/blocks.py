import logging

import torch
import torch.nn as nn

import torch_scatter

import comfy.ops
ops = comfy.ops.manual_cast

log = logging.getLogger("motioncapture")

class LayerNorm1D(nn.Module):
    def __init__(self, dim, operations=ops):
        super(LayerNorm1D, self).__init__()
        self.norm = operations.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim, operations=ops):
        super().__init__()

        self.gate = nn.Sequential(
            operations.Linear(dim, dim),
            nn.Sigmoid())

        self.res = nn.Sequential(
            operations.Linear(dim, dim),
            nn.ReLU(inplace=True),
            operations.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):
    def __init__(self, dim=512, expand=True, operations=ops):
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = operations.Linear(self.dim, self.dim)
        self.g = operations.Linear(self.dim, self.dim)
        self.h = operations.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)

class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True, operations=ops):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = operations.Linear(self.dim, self.dim)
        self.g = operations.Linear(self.dim,        1)
        self.h = operations.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]

        return self.h(y)


### Gradient Clipping and Zeroing Operations ###

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        log.debug("%s", grad_x.abs().mean())
        return grad_x
