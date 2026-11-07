# Measure a given iPEPS

`evaluate.py` contracts a saved state and measures its energy, local magnetic
moments and nearest-neighbor correlations. It does not optimize the tensors.

Run from `src_code`:

```sh
python examples/evaluate.py
python examples/evaluate.py output/c3rotated2tensors_D3/tensor.npz --chi 8 --output output/c3rotated2tensors_D3/evaluate.json
```

The first command reads the supplied `examples/tensor.npz` and writes
`output/evaluate.json`. Without `--chi`, the boundary dimension is read
from the tensor file. Add `--device cuda` to contract on a GPU.
The sample should have antiferromagnetic NN correlations and nonzero
staggered magnetization.

## Saved quantities

| Field | Physical meaning |
| --- | --- |
| `ansatz` | Symmetry constraints of the saved wave function: `3legSymmetrized1tensor`, `c3rotated2tensors`, or `unrestricted6tensors`. |
| `D` | Virtual bond dimension of the iPEPS; it controls the variational entanglement. |
| `chi` | Retained CTM boundary dimension used for this measurement. |
| `J1` | Uniform nearest-neighbor exchange; positive values are antiferromagnetic. |
| `energy_per_site` | `<H>/N = (3 J1/2) mean_<ij><Si.Sj>`, in the same energy units as `J1`. |
| `staggered_magnetization` | Magnitude of `(SA-SB+SC-SD+SE-SF)/6`; each `Si` is the local spin expectation. A fully ordered product state has `m=1/2`. |
| `site_spins` | For each site `A`–`F`, `[<Sx>,<Sy>,<Sz>]`, with `hbar=1`. `Sy=0` for the real wave functions used here. |
| `bond_correlations` | `<Si.Sj>` for each labeled NN bond. Negative values indicate antiferromagnetic alignment; an isolated singlet gives `-3/4`. |
| `ctm_steps` | Number of complete CTMRG iterations, each containing the three translated growth steps. |
| `ctm_residual` | Change in normalized corner spectra between the final two CTMRG iterations, defined below. |
| `ctm_energy_residual` | Absolute change in energy per site between the final two CTMRG iterations. |
| `ctm_converged` | Whether both stopping thresholds were met: `ctm_residual < 1e-8` and `ctm_energy_residual < 1e-9`. |

The six-site cell has nine NN bonds: `AB`, `BC`, `CD`, `DE`, `EF`, `FA`,
`AD`, `BE`, and `CF`. Both `XY` and `YX` are stored: they are two CTM
embeddings of the same physical bond, not different interactions. The
energy averages all 18 estimates; each local spin averages its incident
bond estimates.

For each product of the three corners, let `s_i` denote its singular
values divided by the largest one. Then

`ctm_residual = (1/2) max |s_i(new)^2 - s_i(old)^2|`,

where the maximum includes all corner starting points and all three
translated environments. It is a dimensionless stopping diagnostic,
not a discarded weight or an uncertainty on an observable.
