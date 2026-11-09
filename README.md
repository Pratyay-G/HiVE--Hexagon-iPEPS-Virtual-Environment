# HiVE: Hexagon iPEPS with Virtual Environment

![HiVE](logo.png)

Real honeycomb iPEPS for the spin-1/2 nearest-neighbor Heisenberg model,
`H = J1 sum_<ij> Si.Sj`, on CPU or GPU.

From this directory, with Python 3.11 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python examples/optimize.py --ansatz c3rotated2tensors --D 2 --chi 4
python examples/evaluate.py
python examples/correlation_length.py
python examples/renyi_entropy.py
```

The virtual environment keeps HiVE's dependencies separate from other Python
installations. Installing the project with `pip install .` also installs the
dependencies listed in `requirements.txt`.

The [four example guides](examples/README.md) explain each calculation,
its physical expectations, and every saved quantity. Measurement scripts
use the supplied `D=2` [sample tensor](examples/tensor.npz) by default;
pass another tensor path to measure an optimized state. Results go to
`output/`, or to the destination supplied with `--output`.

`hive_ipeps/` contains the shared tensor, full hexagonal CTMRG, observable,
L-BFGS optimization and boundary-transfer routines. The three ansatze use
real site tensors with axes `(virtual 1, virtual 2, virtual 3, spin)` and
shape `(D, D, D, 2)`:

| Identifier | Independent tensors | Six-site unit cell `(A, B, C, D, E, F)` |
| --- | --- | --- |
| `3legSymmetrized1tensor` | One tensor, averaged over all six permutations of its three virtual legs. | `(a, U a, a, U a, a, U a)`, where `U = [[0, 1], [-1, 0]]` acts on the spin index: a pi spin rotation about y, up to an overall phase. |
| `c3rotated2tensors` | Two independent tensors `a, b`; neither is individually symmetrized. | `(a, b, P a, P b, P^2 a, P^2 b)`, where `P T = T.permute(2, 0, 1, 3)` rotates the three virtual legs. |
| `unrestricted6tensors` | Six independent tensors. | `(A, B, C, D, E, F)`, with no imposed site identification or virtual-leg symmetry within the periodic cell. |

The identifier `3legSymmetrized1tensor` denotes the single-tensor Neel parameterization
in commands and saved tensor files. All three ansatze use the same full
environment: three translated sectors with three corners and six edges each.
Tensor files are portable between CPU and GPU; add `--device cuda` to a
command with a CUDA-enabled PyTorch installation.

CTMRG stops on both corner-spectrum and energy changes. Optimization
also checks the same state at `chi`, `chi+D`, and `chi+2D`; all three
energy differences must be below `3e-5`. Correlation lengths and Renyi
entropies use opposite full CTM boundaries for all three ansatze.

## Authors and license

Chengzhi Ye and Pratyay Ghosh. 

Developed at the
[Chair of Condensed Matter Theory (CTMC), EPFL](https://www.epfl.ch/labs/ctmc/)

Contact: [pratyay.ghosh@epfl.ch](mailto:pratyay.ghosh@epfl.ch).

Contact: [chengzhi.ye@epfl.ch](mailto:chengzhi.ye@epfl.ch).

The software and its accompanying documentation in this directory are
distributed under the [MIT License](LICENSE), copyright 2026 Chengzhi Ye
and Pratyay Ghosh. Runtime dependencies are listed once in
`requirements.txt` and read by `pyproject.toml`. The installable library is
`hive_ipeps`; the four examples and their input tensor are included in the source
distribution.

## Release information

| Field | Value |
| --- | --- |
| Distribution name | `hive-ipeps`; Python import: `hive_ipeps` |
| Version | `1010.1011.0` |
| Public contact | `chengzhi.ye@epfl.ch` `pratyay.ghosh@epfl.ch` |
| Source repository | 【TO BE CREATED】 |
| Chengzhi Ye ORCID | [0009-0004-0190-1625](https://orcid.org/0009-0004-0190-1625) |
| Pratyay Ghosh ORCID | [0000-0001-8192-3963](https://orcid.org/0000-0001-8192-3963) |
| Associated publication | See [Citation](#citation) below. |

## Citation

Cite the associated publication when using HiVE or its hexagonal CTMRG
implementation in scientific work (to be updated):

> 【FINAL AUTHOR LIST】, *【FINAL PUBLICATION TITLE】*,
> 【JOURNAL, VOLUME, PAGES AND YEAR OR ARXIV IDENTIFIER】,
> 【PUBLICATION DOI OR URL】.

Replace these placeholders with the final paper reference when available.
