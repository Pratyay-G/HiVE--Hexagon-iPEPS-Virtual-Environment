"""Opposite CTM boundaries, row correlations and cylinder Renyi-2 entropy."""
import numpy as np
import torch

from .ctm import ctmrg
from .tensors import contract, double_layer, normalize, unit_cell

# Straight boundary pairs (odd, even), parallel to alpha, beta and gamma.
_PAIRS = ((7, 4), (3, 6), (5, 8))
_AXES = ((2, 1, 0), (0, 2, 1), (1, 0, 2))  # odd, even, remaining bond
_CENTRAL = (((1, 2), (3, 4), (5, 0)),
            ((5, 4), (1, 0), (3, 2)),
            ((3, 0), (5, 2), (1, 4)))  # [translated cut][direction]


@torch.no_grad()
def paired_environments(parameters, ansatz, chi=4, **ctm_options):
    """Invert the lattice about an AB bond midpoint; virtual leg names stay fixed."""
    sites = unit_cell(parameters, ansatz)
    reverse = tuple(sites[i] for i in (1, 0, 3, 2, 5, 4))
    return (sites, ctmrg(sites, chi, **ctm_options),
            ctmrg(reverse, chi, **ctm_options))


def _cuts(ansatz):
    # Threefold symmetry relates directions, but not translated cuts.
    return ((direction, cut) for direction in range(3 if ansatz == "unrestricted6tensors" else 1)
            for cut in range(3))


@torch.no_grad()
def correlation_lengths(parameters, ansatz, chi=4, **ctm_options):
    """Ordinary row-transfer lengths for each inequivalent direction and cut.

    One cell contains two central sites and advances three NN bond lengths.
    Each straight edge pair shares both CTM boundary spaces.
    """
    sites, normal, reverse = paired_environments(parameters, ansatz, chi, **ctm_options)
    layers = tuple(map(double_layer, sites))
    rows = []
    for direction, cut in _cuts(ansatz):
        odd, even = _PAIRS[direction]
        u, v, w = _AXES[direction]
        left, right = _CENTRAL[cut][direction]
        upper, lower = normal.sectors[cut], reverse.sectors[(2-cut) % 3]
        transfer = contract("MYi,MVj,ihk,jhl,myk,mvl->YyVv",
                            upper[even], upper[odd],
                            layers[left].permute(v, w, u), layers[right].permute(u, w, v),
                            lower[odd], lower[even]).reshape(chi*chi, chi*chi)
        values = torch.linalg.eigvals(transfer)
        values = values[torch.argsort(values.abs(), descending=True)]
        ratio = float((values[1] / values[0]).abs()) if len(values) > 1 else 0.
        inverse = float(-np.log(ratio)) if ratio else None
        xi = 0. if ratio == 0 else (1 / inverse if inverse > 1e-14 else None)
        rows.append({"direction": direction + 1, "cut": cut + 1,
                     "xi": xi, "inverse_xi": inverse,
                     "lambda2_over_lambda1_abs": ratio,
                     "normalized_eigenvalues": [[float(z.real), float(z.imag)]
                                                for z in values / values[0]]})
    return {"ansatz": ansatz, "D": sites[0].shape[0], "chi": chi,
            "length_unit": "one row-transfer cell (three NN bond lengths)",
            "eigenproblem": "ordinary row transfer", "directions": rows}


def replica_edges(normal, reverse, direction, cut):
    """Unfuse ket/bra legs; the opposite half has reversed physical indices."""
    odd, even = _PAIRS[direction]
    n1, n2 = (normal.sectors[cut][i] for i in (odd, even))
    r1, r2 = (reverse.sectors[-cut % 3][i] for i in (odd, even))
    chi, _, q = n1.shape
    D = int(q ** .5)
    shape = (chi, chi, D, D)
    edges = (n1.reshape(shape).permute(2, 3, 1, 0),
             r1.reshape(shape).permute(3, 2, 0, 1),
             r2.reshape(shape).permute(3, 2, 1, 0),
             n2.reshape(shape).permute(2, 3, 0, 1))
    return tuple(map(normalize, edges))


def replica_matrices(edges):
    """Dense one- and two-replica transfers for a small CTM boundary."""
    a, b, c, d = edges
    chi = a.shape[2]
    ab = contract("abip,bajq->ijpq", a, b)
    dc = contract("abpi,baqj->pqij", d, c)
    one = contract("pqij,klpq->ijkl", dc, ab).reshape(chi**2, chi**2)

    def ring(first, second):
        return contract("abip,bcjq,cdkr,dals->pqrsijkl",
                        first, second, first, second).reshape(chi**4, chi**4)

    return one, ring(d, c) @ ring(a, b)


def log_trace_powers(values, powers):
    scale = np.max(np.abs(values))
    trace = np.sum((values[:, None] / scale) ** powers[None, :], axis=0)
    if np.any(trace.real <= 0) or np.max(np.abs(trace.imag)) > 1e-8 * np.max(np.abs(trace.real)):
        raise RuntimeError("A replica trace is nonpositive or complex.")
    return powers * np.log(scale) + np.log(trace.real)


@torch.no_grad()
def renyi_entropy(parameters, ansatz, chi=4, lengths=(2, 4, 6, 8, 10, 12),
                  **ctm_options):
    """S2 = 2 log Tr(T1**(L/2)) - log Tr(T2**(L/2)), using full spectra.

    L counts cut virtual bonds around the cylinder and is positive and even.
    Six independent tensors have three translated cuts in each of three directions.
    """
    lengths = np.asarray(lengths, dtype=int)
    if np.any(lengths < 2) or np.any(lengths % 2):
        raise ValueError("Circumferences must be positive even numbers of cut bonds.")
    sites, normal, reverse = paired_environments(parameters, ansatz, chi, **ctm_options)
    rows = []
    for direction, cut in _cuts(ansatz):
        one, two = replica_matrices(replica_edges(normal, reverse, direction, cut))
        eig1 = torch.linalg.eigvals(one).cpu().numpy()
        eig2 = torch.linalg.eigvals(two).cpu().numpy()
        entropy = 2 * log_trace_powers(eig1, lengths // 2) - log_trace_powers(eig2, lengths // 2)
        rows.append({"direction": direction + 1, "cut": cut + 1, "S2": entropy.tolist()})
    return {"ansatz": ansatz, "D": sites[0].shape[0], "chi": chi,
            "L_cut_bonds": lengths.tolist(), "logarithm": "natural", "cuts": rows}
