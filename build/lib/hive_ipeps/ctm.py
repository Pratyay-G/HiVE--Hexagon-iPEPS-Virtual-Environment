"""Full hexagonal CTMRG: three translated sectors, each with 3 corners and 6 edges."""
from dataclasses import dataclass

import torch

from .tensors import contract, double_layer, inverse_sqrt, normalize, truncated_svd


def truncate_corners(corners, chi):
    """Biorthogonal projectors from three independently enlarged corners."""
    incoming, outgoing = [None] * 3, [None] * 3
    for i in range(3):
        r1 = corners[i].T
        r2 = corners[(i+2) % 3] @ corners[(i+1) % 3]
        u, s, v = truncated_svd(r1 @ r2.T, chi)
        scale = inverse_sqrt(s)
        outgoing[i] = r2.T @ v @ scale
        incoming[(i+1) % 3] = scale @ u.T @ r1
    small = tuple(normalize(outgoing[i].T @ corners[i] @ incoming[i].T)
                  for i in range(3))
    return incoming, small, outgoing


def initialize_environment(layers, chi):
    """A finite patch, with each external virtual leg traced over ket and bra."""
    A, B, C, D, E, F = layers
    q = A.shape[0]
    bond = int(q ** .5)

    def trace(t, rotation=0):
        t = t.permute(*((0, 1, 2), (1, 2, 0), (2, 0, 1))[rotation])
        return t.reshape(bond, bond, q, q).diagonal(dim1=0, dim2=1).sum(-1)

    BC = contract("iBG,ibg->BGbg", B, C)
    DE = contract("AiG,aig->GAga", D, E)
    FA = contract("ABi,abi->ABab", F, A)
    raw = (
        contract("yj,ijYk,in,nm,kXlm,lx->YyXx", trace(E), BC, trace(A),
                 trace(F, 1), DE, trace(B, 1)).reshape(q*q, q*q),
        contract("zj,ijZk,in,nm,kYlm,ly->ZzYy", trace(A, 1), DE, trace(C, 1),
                 trace(B, 2), FA, trace(D, 2)).reshape(q*q, q*q),
        contract("xj,ijXk,in,nm,kZlm,lz->XxZz", trace(C, 2), FA, trace(E, 2),
                 trace(D), BC, trace(F)).reshape(q*q, q*q))
    pin, corners, pout = truncate_corners(raw, chi)
    pin = [p.reshape(chi, q, q) for p in pin]
    pout = [p.reshape(q, q, chi) for p in pout]
    # Contract a projector before forming each enlarged edge.
    edges = [
        contract("mi,jyi,aYjM,pYy->Mmpa", trace(C), D, FA, pin[1]),
        contract("il,xji,LjXb,Xxp->Llpb", trace(D, 1), C, FA, pout[2]),
        contract("ni,ijz,bZjN,pZz->Nnpb", trace(E, 1), F, BC, pin[2]),
        contract("im,iyj,MjYg,Yyp->Mmpg", trace(F, 2), E, BC, pout[0]),
        contract("li,xij,gXjL,pXx->Llpg", trace(A, 2), B, DE, pin[0]),
        contract("in,jiz,NjZa,Zzp->Nnpa", trace(B), A, DE, pout[1])]
    edges = [t.reshape(q*q, chi*q) for t in edges]
    for first, second in ((0, 3), (4, 1), (2, 5)):
        x, y = edges[first], edges[second]
        u, s, v = truncated_svd(x.T @ y, chi)
        scale = inverse_sqrt(s)
        edges[first] = (y @ v @ scale).T @ x
        edges[second] = scale @ u.T @ x.T @ y
    return (*corners, *(normalize(t.reshape(chi, chi, q)) for t in edges))


def grow_environment(environment, layers, chi, stage):
    """Absorb one hexagonal layer without identifying any corners or edges."""
    c21, c32, c13, t1o, t2e, t2o, t3e, t3o, t1e = environment
    # Successive translations: (E,B,A,D,C,F), (A,F,C,B,E,D), (C,D,E,F,A,B).
    order = ((4, 1, 0, 3, 2, 5), (0, 5, 2, 1, 4, 3), (2, 3, 4, 5, 0, 1))[stage]
    a, b, c, d, e, f = (layers[i] for i in order)
    q = a.shape[0]
    boundary = t1o.shape[0]
    chi = min(chi, boundary*q)
    raw = (
        contract("YX,MYa,LXb,amg,lbg->MmLl", c21, t1o, t2e, a, b),
        contract("ZY,NZb,MYg,abn,amg->NnMm", c32, t2o, t3e, c, d),
        contract("XZ,LXg,NZa,lbg,abn->LlNn", c13, t3o, t1e, e, f))
    pin, corners, pout = truncate_corners(
        tuple(t.reshape(boundary*q, boundary*q) for t in raw), chi)
    pin = [p.reshape(chi, boundary, q) for p in pin]
    pout = [p.reshape(boundary, q, chi) for p in pout]
    edges = (
        contract("OYg,abg,MOb->YMa", t3e, d, pin[1]),
        contract("OXg,abg,OaL->XLb", t3o, e, pout[2]),
        contract("OZa,abg,NOg->ZNb", t1e, f, pin[2]),
        contract("OYa,abg,ObM->YMg", t1o, a, pout[0]),
        contract("OXb,abg,LOa->XLg", t2e, b, pin[0]),
        contract("OZb,abg,OgN->ZNa", t2o, c, pout[1]))
    return (*corners, *map(normalize, edges))


@dataclass
class Environment:
    sectors: tuple
    steps: int
    residual: float
    energy_residual: float
    converged: bool


def ctmrg(sites, chi=6, max_steps=80, tolerance=1e-8, energy_tolerance=1e-9):
    """Stop when both corner singular values and energy per site stabilize.

    The finite-patch initialization is fixed during differentiation. All
    subsequent CTMRG iterations belong to the automatic-differentiation graph.
    """
    if chi < 1:
        raise ValueError("The boundary dimension must be positive.")
    if max_steps < 2:
        raise ValueError("At least two CTMRG iterations are needed for convergence.")
    from .observables import bond_densities, energy_from_densities

    layers = tuple(map(double_layer, sites))
    with torch.no_grad():
        first = initialize_environment(layers, min(chi, sites[0].shape[0] ** 4))
        if sites[0].shape[0] == 1 and chi > 1:
            # A product-state boundary has rank one at every size.
            first = tuple(torch.nn.functional.pad(t, (0, chi-1, 0, chi-1))
                          for t in first[:3]) + tuple(
                torch.nn.functional.pad(t, (0, 0, 0, chi-1, 0, chi-1))
                for t in first[3:])
        while first[0].shape[0] != chi or first[3].shape[:2] != (chi, chi):
            for stage in range(3):
                first = grow_environment(first, layers, chi, stage)
    previous, previous_energy = None, None
    residual = energy_residual = float("inf")
    for step in range(1, max_steps + 1):
        second = grow_environment(first, layers, chi, 0)
        third = grow_environment(second, layers, chi, 1)
        first = grow_environment(third, layers, chi, 2)
        sectors = (first, second, third)
        with torch.no_grad():
            spectra = []
            for sector in sectors:
                for i in range(3):
                    s = torch.linalg.svdvals(sector[(i+2) % 3] @
                                             sector[(i+1) % 3] @ sector[i])
                    spectra.append(s / s[0])
            current = torch.stack(spectra)
            current_energy = float(energy_from_densities(bond_densities(sites, sectors)))
            if previous is not None:
                residual = float(((current - previous).abs() * .5 *
                                  (current + previous)).max())
                energy_residual = abs(current_energy - previous_energy)
            previous, previous_energy = current, current_energy
        if residual < tolerance and energy_residual < energy_tolerance:
            break
    return Environment(sectors, step, residual, energy_residual,
                       residual < tolerance and energy_residual < energy_tolerance)
