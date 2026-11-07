"""Measure energy, Neel order and all nearest-neighbor bond correlations."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hive_ipeps import load_tensor, observables, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", nargs="?", type=Path,
                        default=Path(__file__).parent / "tensor.npz")
    parser.add_argument("--chi", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output/evaluate.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    parameters, ansatz, J1, chi = load_tensor(args.tensor, args.device)
    result = observables(parameters, ansatz, J1, args.chi or chi)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
