"""Minimize the nearest-neighbor Heisenberg energy by L-BFGS."""
import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hive_ipeps import ANSATZE, initial_parameters, load_tensor, save_tensor, write_json
from hive_ipeps.optimization import optimize
from hive_ipeps.tensors import unit_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ansatz", choices=ANSATZE, default="c3rotated2tensors")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--chi", type=int)
    parser.add_argument("--J1", type=float, default=1.)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--ctm-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or Path(__file__).resolve().parents[1] / "output" / f"{args.ansatz}_D{args.D}"
    output.mkdir(parents=True, exist_ok=True)
    parameters = initial_parameters(args.ansatz, args.D, args.seed, args.device)
    if args.resume:
        old, old_ansatz, _, _ = load_tensor(args.resume, args.device)
        if old[0].shape[0] > args.D:
            raise ValueError("The resumed bond dimension cannot exceed the target D.")
        if args.ansatz == "unrestricted6tensors":
            old = unit_cell(old, old_ansatz)
        elif old_ansatz != args.ansatz:
            raise ValueError("Resume the same ansatz, or expand it to six independent tensors.")
        with torch.no_grad():
            for new, previous in zip(parameters, old):
                new.mul_(.01)
                d = previous.shape[0]
                new[:d, :d, :d, :] = previous

    def report(row):
        print(f"chi={row['chi']:3d} step={row['step']:3d} "
              f"E/site={row['energy_per_site']:.12f} "
              f"m={row['staggered_magnetization']:.8f}", flush=True)

    result = optimize(parameters, args.ansatz, args.J1, args.chi, args.steps,
                      callback=report, max_steps=args.ctm_steps)
    save_tensor(output / "tensor.npz", parameters, args.ansatz, args.J1,
                result["observables"]["chi"])
    write_json(output / "observables.json", result["observables"])
    write_json(output / "run.json", {key: value for key, value in result.items()
                                      if key not in ("observables", "history")})
    with (output / "optimization.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result["history"][0].keys())
        writer.writeheader()
        writer.writerows(result["history"])
    print(f"Three-dimension energy check: {result['chi_converged']}; "
          f"elapsed: {result['elapsed_seconds']:.1f} s")


if __name__ == "__main__":
    main()
