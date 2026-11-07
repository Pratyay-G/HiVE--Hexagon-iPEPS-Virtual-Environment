# Optimize the nearest-neighbor Heisenberg state

`optimize.py` lowers the energy per site of
`H = J1 sum_<ij> Si.Sj` by L-BFGS. For `J1=1`, the Neel product state
has energy `-3/8` per site. Entanglement should lower this energy while
retaining a staggered moment. Larger `D` admits a richer wave function;
larger `chi` improves its infinite-environment contraction.

Run from `src_code`:

```sh
python examples/optimize.py --ansatz c3rotated2tensors --D 2 --chi 4
```

This creates `output/c3rotated2tensors_D2/`. The three ansatze are:

| `--ansatz` | Independent site tensors and symmetry |
| --- | --- |
| `3legSymmetrized1tensor` | One tensor, symmetrized over its virtual legs; a spin rotation generates the other sublattice. |
| `c3rotated2tensors` | Two tensors; threefold lattice rotations generate the six-site cell. |
| `unrestricted6tensors` | Six independent tensors, allowing each site and direction to vary. |

Examples of other variational and boundary dimensions:

```sh
python examples/optimize.py --ansatz 3legSymmetrized1tensor --D 3 --chi 6
python examples/optimize.py --ansatz c3rotated2tensors --D 4 --chi 8
python examples/optimize.py --ansatz unrestricted6tensors --D 5 --chi 10
python examples/optimize.py --ansatz c3rotated2tensors --D 4 --chi 12 --output output/c3rotated2tensors_D4_chi12
```

Any listed ansatz can be combined with these `D` and `chi` values. Omitting
`--chi` starts at `2*D`. `--steps` sets the maximum L-BFGS steps at each
boundary dimension (default 60); `--ctm-steps` sets the CTMRG iteration
budget (default 80). `--seed` controls the initial noisy Neel state.
`--device cuda` selects a GPU; `--J1` changes the energy scale.

The optimized tensors are measured at `chi`, `chi+D`, and `chi+2D`. All
three energy differences must be below `3e-5`. Otherwise optimization
continues from the same tensors at `chi+2D`; the requested `chi` is a
starting value. Two consecutive optimization energy changes below `1e-10`
also stop the L-BFGS stage early.

To increase `D` or release the symmetry of a previous state:

```sh
python examples/optimize.py --ansatz c3rotated2tensors --D 3 --chi 6 --resume output/c3rotated2tensors_D2/tensor.npz
python examples/optimize.py --ansatz unrestricted6tensors --D 3 --chi 6 --resume output/c3rotated2tensors_D3/tensor.npz
```

At unchanged `D`, `--resume` starts from the saved tensor values without
adding noise. Increasing `D` preserves the old tensor block exactly and
initializes only the added components with a small seeded perturbation.
Releasing the symmetry first expands the saved state to its six site tensors.

## Saved quantities

| File | Contents |
| --- | --- |
| `tensor.npz` | `ansatz`, `J1`, the final optimization `chi`, and the independent `tensor_i` arrays. Each array has axes `(alpha,beta,gamma,spin)` and shape `(D,D,D,2)`. The spin basis is `Sz=(+1/2,-1/2)`. |
| `observables.json` | Energy per site, staggered moment, six local spins, individual NN correlations, and CTMRG stopping diagnostics; every field is defined in [evaluate.md](evaluate.md#saved-quantities). |
| `optimization.csv` | One row before optimization (`step=0`) and after each L-BFGS step. `step` restarts at zero when `chi` increases. Other columns: `chi`, `energy_per_site`, `staggered_magnetization`, `ctm_steps`, `ctm_residual`, and `ctm_energy_residual`, with the same meanings as in `observables.json`. |
| `run.json` | Boundary-dimension energy checks and total `elapsed_seconds`, described below. |

Each entry of `run.json`'s `chi_checks` records:

| Field | Meaning |
| --- | --- |
| `chi` | The three boundary dimensions `[chi, chi+D, chi+2D]`. |
| `energies` | Energy per site of the same optimized tensors at those dimensions. |
| `pairwise_energy_differences` | Absolute differences in the order `(0,1)`, `(0,2)`, `(1,2)` of that list. |
| `passed` | Whether all three differences are below `3e-5`. |

`chi_converged` records acceptance of the final energy check. It is distinct
from the CTMRG stopping flag. The saved tensor and observables use the base
`chi` of the accepted triplet, not its largest lookahead value.
