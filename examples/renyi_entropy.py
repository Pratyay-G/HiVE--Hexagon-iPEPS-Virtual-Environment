"""Measure cylinder Renyi-2 entropy across each inequivalent lattice cut."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hive_ipeps import load_tensor, renyi_entropy, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor", nargs="?", type=Path,
                        default=Path(__file__).parent / "tensor.npz")
    parser.add_argument("--chi", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lengths", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output/renyi_entropy.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    parameters, ansatz, _, _ = load_tensor(args.tensor, args.device)
    result = renyi_entropy(parameters, ansatz, args.chi, args.lengths)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
