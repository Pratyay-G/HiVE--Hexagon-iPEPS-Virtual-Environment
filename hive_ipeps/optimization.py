"""L-BFGS optimization and an energy check at three boundary dimensions."""
from itertools import combinations
from time import perf_counter

import torch

from .observables import energy, observables


def optimize(parameters, ansatz, J1=1., chi=None, steps=60, chi_tolerance=3e-5,
             energy_tolerance=1e-10, callback=None, **ctm_options):
    """Resume at chi+2D until all three energy differences pass the lookahead.

    Each lookahead uses the same optimized site tensors at chi, chi+D and
    chi+2D. If it fails, these tensors seed the next L-BFGS optimization.
    """
    D = parameters[0].shape[0]
    chi = chi or 2*D
    if steps < 1:
        raise ValueError("At least one optimization step is required.")
    for tensor in parameters:
        tensor.requires_grad_(True)
    started = perf_counter()
    history, checks = [], []
    accepted = False
    while True:
        optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=1, history_size=20,
                                     line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                                     tolerance_change=1e-12)

        def closure():
            optimizer.zero_grad()
            value, _, _ = energy(parameters, ansatz, J1, chi, **ctm_options)
            value.backward()
            if not torch.isfinite(value) or any(not torch.isfinite(p.grad).all()
                                               for p in parameters):
                raise RuntimeError("Non-finite energy or gradient.")
            return value

        energies = []
        for step in range(steps + 1):
            if step:
                optimizer.step(closure)
            measured = observables(parameters, ansatz, J1, chi, **ctm_options)
            row = {"chi": chi, "step": step,
                   "energy_per_site": measured["energy_per_site"],
                   "staggered_magnetization": measured["staggered_magnetization"],
                   "ctm_steps": measured["ctm_steps"],
                   "ctm_residual": measured["ctm_residual"],
                   "ctm_energy_residual": measured["ctm_energy_residual"]}
            history.append(row)
            energies.append(row["energy_per_site"])
            if callback:
                callback(row)
            if step >= 2 and max(abs(energies[-i] - energies[-i-1])
                                 for i in (1, 2)) < energy_tolerance:
                break
        dimensions = (chi, chi+D, chi+2*D)
        lookahead = [measured] + [observables(parameters, ansatz, J1, value, **ctm_options)
                                  for value in dimensions[1:]]
        differences = [abs(a["energy_per_site"] - b["energy_per_site"])
                       for a, b in combinations(lookahead, 2)]
        accepted = all(value < chi_tolerance for value in differences)
        checks.append({"chi": list(dimensions),
                       "energies": [item["energy_per_site"] for item in lookahead],
                       "pairwise_energy_differences": differences, "passed": accepted})
        if accepted:
            break
        chi += 2*D
    return {"observables": measured, "history": history, "chi_checks": checks,
            "chi_converged": accepted, "elapsed_seconds": perf_counter() - started}
