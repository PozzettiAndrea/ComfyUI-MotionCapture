"""
Lie group operations for DPVO.
Merged from lietorch/{broadcasting,group_ops,groups}.py into a single flat module.
"""

import lietorch_backends
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Broadcasting helpers
# ---------------------------------------------------------------------------

def check_broadcastable(x, y):
    assert len(x.shape) == len(y.shape)
    for (n, m) in zip(x.shape[:-1], y.shape[:-1]):
        assert n==m or n==1 or m==1

def broadcast_inputs(x, y):
    """ Automatic broadcasting of missing dimensions """
    if y is None:
        xs, xd = x.shape[:-1], x.shape[-1]
        return (x.view(-1, xd).contiguous(), ), x.shape[:-1]

    check_broadcastable(x, y)

    xs, xd = x.shape[:-1], x.shape[-1]
    ys, yd = y.shape[:-1], y.shape[-1]
    out_shape = [max(n,m) for (n,m) in zip(xs,ys)]

    if x.shape[:-1] == y.shape[-1]:
        x1 = x.view(-1, xd)
        y1 = y.view(-1, yd)

    else:
        x_expand = [m if n==1 else 1 for (n,m) in zip(xs, ys)]
        y_expand = [n if m==1 else 1 for (n,m) in zip(xs, ys)]
        x1 = x.repeat(x_expand + [1]).reshape(-1, xd).contiguous()
        y1 = y.repeat(y_expand + [1]).reshape(-1, yd).contiguous()

    return (x1, y1), tuple(out_shape)


# ---------------------------------------------------------------------------
# Group operations (autograd wrappers around lietorch_backends)
# ---------------------------------------------------------------------------

class GroupOp(torch.autograd.Function):
    """ group operation base class """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        out = cls.forward_op(ctx.group_id, *inputs)
        return out

    @classmethod
    def backward(cls, ctx, grad):
        error_str = "Backward operation not implemented for {}".format(cls)
        assert cls.backward_op is not None, error_str

        inputs = ctx.saved_tensors
        grad = grad.contiguous()
        grad_inputs = cls.backward_op(ctx.group_id, grad, *inputs)
        return (None, ) + tuple(grad_inputs)


class Exp(GroupOp):
    forward_op, backward_op = lietorch_backends.expm, lietorch_backends.expm_backward

class Log(GroupOp):
    forward_op, backward_op = lietorch_backends.logm, lietorch_backends.logm_backward

class Inv(GroupOp):
    forward_op, backward_op = lietorch_backends.inv, lietorch_backends.inv_backward

class Mul(GroupOp):
    forward_op, backward_op = lietorch_backends.mul, lietorch_backends.mul_backward

class Adj(GroupOp):
    forward_op, backward_op = lietorch_backends.adj, lietorch_backends.adj_backward

class AdjT(GroupOp):
    forward_op, backward_op = lietorch_backends.adjT, lietorch_backends.adjT_backward

class Act3(GroupOp):
    forward_op, backward_op = lietorch_backends.act, lietorch_backends.act_backward

class Act4(GroupOp):
    forward_op, backward_op = lietorch_backends.act4, lietorch_backends.act4_backward

class Jinv(GroupOp):
    forward_op, backward_op = lietorch_backends.Jinv, None

class ToMatrix(GroupOp):
    forward_op, backward_op = lietorch_backends.as_matrix, None


class FromVec(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), torch.linalg.pinv(J)).squeeze(-2)

class ToVec(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        return inputs[0]

    @classmethod
    def backward(cls, ctx, grad):
        inputs = ctx.saved_tensors
        J = lietorch_backends.projector(ctx.group_id, *inputs)
        return None, torch.matmul(grad.unsqueeze(-2), J).squeeze(-2)


# ---------------------------------------------------------------------------
# Lie group classes
# ---------------------------------------------------------------------------

class LieGroupParameter(torch.Tensor):
    """ Wrapper class for LieGroup """

    from torch._C import _disabled_torch_function_impl
    __torch_function__ = _disabled_torch_function_impl

    def __new__(cls, group, requires_grad=True):
        data = torch.zeros(group.tangent_shape,
                           device=group.data.device,
                           dtype=group.data.dtype,
                           requires_grad=True)

        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self, group):
        self.group = group

    def retr(self):
        return self.group.retr(self)

    def log(self):
        return self.retr().log()

    def inv(self):
        return self.retr().inv()

    def adj(self, a):
        return self.retr().adj(a)

    def __mul__(self, other):
        if isinstance(other, LieGroupParameter):
            return self.retr() * other.retr()
        else:
            return self.retr() * other

    def add_(self, update, alpha):
        self.group = self.group.exp(alpha*update) * self.group

    def __getitem__(self, index):
        return self.retr().__getitem__(index)


class LieGroup:
    """ Base class for Lie Group """

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "{}: size={}, device={}, dtype={}".format(
            self.group_name, self.shape, self.device, self.dtype)

    @property
    def shape(self):
        return self.data.shape[:-1]

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def vec(self):
        return self.apply_op(ToVec, self.data)

    @property
    def tangent_shape(self):
        return self.data.shape[:-1] + (self.manifold_dim,)

    @classmethod
    def Identity(cls, *batch_shape, **kwargs):
        """ Construct identity element with batch shape """

        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]

        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])

        numel = np.prod(batch_shape)
        data = cls.id_elem.reshape(1,-1)

        if 'device' in kwargs:
            data = data.to(kwargs['device'])

        if 'dtype' in kwargs:
            data = data.type(kwargs['dtype'])

        data = data.repeat(numel, 1)
        return cls(data).view(batch_shape)

    @classmethod
    def IdentityLike(cls, G):
        return cls.Identity(G.shape, device=G.data.device, dtype=G.data.dtype)

    @classmethod
    def InitFromVec(cls, data):
        return cls(cls.apply_op(FromVec, data))

    @classmethod
    def Random(cls, *batch_shape, sigma=1.0, **kwargs):
        if isinstance(batch_shape[0], tuple):
            batch_shape = batch_shape[0]

        elif isinstance(batch_shape[0], list):
            batch_shape = tuple(batch_shape[0])

        tangent_shape = batch_shape + (cls.manifold_dim,)
        xi = torch.randn(tangent_shape, **kwargs)
        return cls.exp(sigma * xi)

    @classmethod
    def apply_op(cls, op, x, y=None):
        """ Apply group operator """
        inputs, out_shape = broadcast_inputs(x, y)

        data = op.apply(cls.group_id, *inputs)
        return data.view(out_shape + (-1,))

    @classmethod
    def exp(cls, x):
        """ exponential map: x -> X """
        return cls(cls.apply_op(Exp, x))

    def log(self):
        """ logarithm map """
        return self.apply_op(Log, self.data)

    def inv(self):
        """ group inverse """
        return self.__class__(self.apply_op(Inv, self.data))

    def mul(self, other):
        """ group multiplication """
        return self.__class__(self.apply_op(Mul, self.data, other.data))

    def retr(self, a):
        """ retraction: Exp(a) * X """
        dX = self.__class__.apply_op(Exp, a)
        return self.__class__(self.apply_op(Mul, dX, self.data))

    def adj(self, a):
        """ adjoint operator: b = A(X) * a """
        return self.apply_op(Adj, self.data, a)

    def adjT(self, a):
        """ transposed adjoint operator: b = a * A(X) """
        return self.apply_op(AdjT, self.data, a)

    def Jinv(self, a):
        return self.apply_op(Jinv, self.data, a)

    def act(self, p):
        """ action on a point cloud """

        # action on point
        if p.shape[-1] == 3:
            return self.apply_op(Act3, self.data, p)

        # action on homogeneous point
        elif p.shape[-1] == 4:
            return self.apply_op(Act4, self.data, p)

    def matrix(self):
        """ convert element to 4x4 matrix """
        I = torch.eye(4, dtype=self.dtype, device=self.device)
        I = I.view([1] * (len(self.data.shape) - 1) + [4, 4])
        return self.__class__(self.data[...,None,:]).act(I).transpose(-1,-2)

    def translation(self):
        """ extract translation component """
        p = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        p = p.view([1] * (len(self.data.shape) - 1) + [4,])
        return self.apply_op(Act4, self.data, p)

    def detach(self):
        return self.__class__(self.data.detach())

    def view(self, dims):
        data_reshaped = self.data.view(dims + (self.embedded_dim,))
        return self.__class__(data_reshaped)

    def __mul__(self, other):
        # group multiplication
        if isinstance(other, LieGroup):
            return self.mul(other)

        # action on point
        elif isinstance(other, torch.Tensor):
            return self.act(other)

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    def __setitem__(self, index, item):
        self.data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self.data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self.data.cpu())

    def float(self, device=None):
        return self.__class__(self.data.float())

    def double(self, device=None):
        return self.__class__(self.data.double())

    def unbind(self, dim=0):
        return [self.__class__(x) for x in self.data.unbind(dim=dim)]


class SO3(LieGroup):
    group_name = 'SO3'
    group_id = 1
    manifold_dim = 3
    embedded_dim = 4

    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0])

    def __init__(self, data):
        if isinstance(data, SE3):
            data = data.data[..., 3:7]

        super(SO3, self).__init__(data)


class RxSO3(LieGroup):
    group_name = 'RxSO3'
    group_id = 2
    manifold_dim = 4
    embedded_dim = 5

    # unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data):
        if isinstance(data, Sim3):
            data = data.data[..., 3:8]

        super(RxSO3, self).__init__(data)


class SE3(LieGroup):
    group_name = 'SE3'
    group_id = 3
    manifold_dim = 6
    embedded_dim = 7

    # translation, unit quaternion
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def __init__(self, data):
        if isinstance(data, SO3):
            translation = torch.zeros_like(data.data[...,:3])
            data = torch.cat([translation, data.data], -1)

        super(SE3, self).__init__(data)

    def scale(self, s):
        t, q = self.data.split([3,4], -1)
        t = t * s.unsqueeze(-1)
        return SE3(torch.cat([t, q], dim=-1))


class Sim3(LieGroup):
    group_name = 'Sim3'
    group_id = 4
    manifold_dim = 7
    embedded_dim = 8

    # translation, unit quaternion, scale
    id_elem = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def __init__(self, data):

        if isinstance(data, SO3):
            scale = torch.ones_like(SO3.data[...,:1])
            translation = torch.zeros_like(SO3.data[...,:3])
            data = torch.cat([translation, SO3.data, scale], -1)

        elif isinstance(data, SE3):
            scale = torch.ones_like(data.data[...,:1])
            data = torch.cat([data.data, scale], -1)

        elif isinstance(data, Sim3):
            data = data.data

        super(Sim3, self).__init__(data)


def cat(group_list, dim):
    """ Concatenate groups along dimension """
    data = torch.cat([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)

def stack(group_list, dim):
    """ Concatenate groups along dimension """
    data = torch.stack([X.data for X in group_list], dim=dim)
    return group_list[0].__class__(data)
