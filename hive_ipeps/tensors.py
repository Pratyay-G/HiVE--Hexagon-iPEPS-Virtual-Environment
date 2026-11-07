"""Real honeycomb iPEPS tensors, with axes (virtual 1, 2, 3, spin)."""
from itertools import permutations

import opt_einsum as oe
import torch

ANSATZE = ("3legSymmetrized1tensor", "c3rotated2tensors", "unrestricted6tensors")
DTYPE = torch.float64


def contract(expression, *tensors):
    return oe.contract(expression, *tensors, backend="torch", optimize="greedy")


def normalize(tensor):
    return tensor / torch.linalg.vector_norm(tensor).clamp_min(1e-30)


def unit_cell(parameters, ansatz):
    """Six sites A--F; ansatz symmetries constrain the state, not its CTM."""
    if ansatz == "3legSymmetrized1tensor":
        a = normalize(sum(parameters[0].permute(*p, 3)
                          for p in permutations(range(3))) / 6)
        b = torch.einsum("ij,...j->...i", a.new_tensor([[0, 1], [-1, 0]]), a)
        return a, b, a, b, a, b
    if ansatz == "c3rotated2tensors":
        a, b = map(normalize, parameters)
        return (a, b, a.permute(2, 0, 1, 3), b.permute(2, 0, 1, 3),
                a.permute(1, 2, 0, 3), b.permute(1, 2, 0, 3))
    if ansatz == "unrestricted6tensors" and len(parameters) == 6:
        return tuple(map(normalize, parameters))
    raise ValueError(f"Choose an ansatz from {ANSATZE}.")


def initial_parameters(ansatz="c3rotated2tensors", D=3, seed=7, device="cpu"):
    """A reproducible noisy Neel initialization."""
    if ansatz not in ANSATZE or D < 1:
        raise ValueError("Invalid ansatz or bond dimension.")
    generator = torch.Generator().manual_seed(seed)
    count = {"3legSymmetrized1tensor": 1, "c3rotated2tensors": 2, "unrestricted6tensors": 6}[ansatz]
    result = []
    for index in range(count):
        tensor = .1 * torch.randn(D, D, D, 2, dtype=DTYPE, generator=generator)
        tensor[0, 0, 0, index % 2] += (-1.) ** index
        result.append(tensor.to(device).requires_grad_())
    return result


def double_layer(site):
    q = site.shape[0] ** 2
    return contract("uvwp,xyzp->uxvywz", site, site).reshape(q, q, q)


class _RegularizedSVD(torch.autograd.Function):
    """Full real SVD: backward retains kept--discarded vector mixing."""
    @staticmethod
    def forward(ctx, matrix):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        ctx.save_for_backward(u, s, vh.T)
        return u, s, vh.T

    @staticmethod
    def backward(ctx, gu, gs, gv):
        u, s, v = ctx.saved_tensors
        epsilon = (s[0] * 1e-12).clamp_min(1e-30)
        difference = s[None, :] - s[:, None]
        total = s[None, :] + s[:, None]
        f = difference / (difference.square() + epsilon.square())
        g = total / (total.square() + epsilon.square())
        f.fill_diagonal_(0)
        g.fill_diagonal_(0)
        left, right = u.T @ gu, v.T @ gv
        middle = torch.diag(gs) + .5 * ((f + g) * (left - left.T)
                                      + (f - g) * (right - right.T))
        return u @ middle @ v.T


def truncated_svd(matrix, chi):
    u, s, v = _RegularizedSVD.apply(matrix)
    return u[:, :chi], s[:chi], v[:, :chi]


def inverse_sqrt(s):
    return torch.diag(s.clamp_min(s[0] * torch.finfo(s.dtype).eps).rsqrt())
