"""Uniform nearest-neighbor Heisenberg energy, spins and individual bonds."""
import torch

from .tensors import DTYPE, contract, unit_cell


def spin_operators(device="cpu"):
    sx = torch.tensor([[0, .5], [.5, 0]], dtype=DTYPE, device=device)
    sz = torch.diag(torch.tensor([.5, -.5], dtype=DTYPE, device=device))
    plus = torch.tensor([[0, 1], [0, 0]], dtype=DTYPE, device=device)
    h = torch.kron(sz, sz) + .5 * (torch.kron(plus, plus.T.contiguous())
                                  + torch.kron(plus.T.contiguous(), plus))
    return sx, sz, h.reshape(2, 2, 2, 2)


def bond_densities(sites, sectors):
    """Eighteen oriented estimates: two orientations of each of nine NN bonds."""
    D = sites[0].shape[0]
    result = {}
    for sector, names in zip(sectors, ("EDAFCB", "ABCDEF", "CFEBAD")):
        c21, c32, c13 = sector[:3]
        chi = c21.shape[0]
        t1o, t2e, t2o, t3e, t3o, t1e = (
            t.reshape(chi, chi, D, D) for t in sector[3:])
        a, b, c, d, e, f = (sites["ABCDEF".index(name)] for name in names)
        opens = (
            contract("YX,MYar,abci,rstj->MbsXctij", c21, t1o, a, a),
            contract("MYct,abci,rstj->YarMbsij", t3e, b, b),
            contract("ZY,NZbs,abci,rstj->NctYarij", c32, t2o, c, c),
            contract("NZar,abci,rstj->ZbsNctij", t1e, d, d),
            contract("XZ,LXct,abci,rstj->LarZbsij", c13, t3o, e, e),
            contract("LXbs,abci,rstj->XctLarij", t2e, f, f))
        ring_order = (2, 1, 0, 5, 4, 3)
        opens = [opens[i].reshape(chi*D*D, chi*D*D, 2, 2) for i in ring_order]
        names = [names[i] for i in ring_order]
        closed = [torch.einsum("abii->ab", item) for item in opens]
        for i in range(6):
            closure = closed[(i+2) % 6] @ closed[(i+3) % 6]
            closure = closure @ closed[(i+4) % 6] @ closed[(i+5) % 6]
            rho = contract("MN,NYij,YMkl->ikjl", closure,
                           opens[i], opens[(i+1) % 6]).reshape(4, 4)
            rho = .5 * (rho + rho.T)
            # Match the positive, normalized two-spin density used for measurements.
            values, vectors = torch.linalg.eigh(rho)
            values = values.clamp_min(0)
            rho = (vectors * values) @ vectors.T / values.sum().clamp_min(1e-30)
            result[names[i] + names[(i+1) % 6]] = rho.reshape(2, 2, 2, 2)
    return result


def energy_from_densities(rhos, J1=1.):
    h = spin_operators(next(iter(rhos.values())).device)[2]
    return 1.5 * J1 * torch.stack([
        torch.einsum("ijkl,klij->", rho, h) for rho in rhos.values()]).mean()


def energy(parameters, ansatz, J1=1., chi=6, **ctm_options):
    from .ctm import ctmrg
    sites = unit_cell(parameters, ansatz)
    environment = ctmrg(sites, chi, **ctm_options)
    rhos = bond_densities(sites, environment.sectors)
    return energy_from_densities(rhos, J1), environment, rhos


@torch.no_grad()
def observables(parameters, ansatz, J1=1., chi=6, **ctm_options):
    value, environment, rhos = energy(parameters, ansatz, J1, chi, **ctm_options)
    sx, sz, h = spin_operators(parameters[0].device)
    bonds = {name: float(torch.einsum("ijkl,klij->", rho, h))
             for name, rho in rhos.items()}
    local = {name: [] for name in "ABCDEF"}
    for names, rho in rhos.items():
        reduced = (torch.einsum("ikjk->ij", rho), torch.einsum("ikil->kl", rho))
        for name, density in zip(names, reduced):
            local[name].append(torch.stack((torch.trace(density @ sx),
                                           density.new_zeros(()), torch.trace(density @ sz))))
    spins = {name: torch.stack(values).mean(0) for name, values in local.items()}
    staggered = sum((-1.) ** i * spins[name] for i, name in enumerate("ABCDEF")) / 6
    return {"ansatz": ansatz, "D": parameters[0].shape[0], "chi": chi, "J1": J1,
            "energy_per_site": float(value), "staggered_magnetization": float(staggered.norm()),
            "site_spins": {name: spin.tolist() for name, spin in spins.items()},
            "bond_correlations": bonds, "ctm_steps": environment.steps,
            "ctm_residual": environment.residual,
            "ctm_energy_residual": environment.energy_residual,
            "ctm_converged": environment.converged}
