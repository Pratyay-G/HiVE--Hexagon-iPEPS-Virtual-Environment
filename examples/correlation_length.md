# Measure correlation decay

`correlation_length.py` builds the ordinary row transfer matrix between
opposite CTM boundaries. Its leading eigenvalue normalizes the state;
the ratio of the two largest eigenvalue magnitudes gives the longest
decay length supported by that strip.

Run from `src_code`:

```sh
python examples/correlation_length.py
python examples/correlation_length.py output/unrestricted6tensors_D3/tensor.npz --chi 6 --output output/unrestricted6tensors_D3/xi.json
```

The first command uses `examples/tensor.npz` and writes
`output/correlation_length.json`. `--chi` overrides the tensor file's
boundary dimension. Add `--device cuda` for a GPU. Finite-`D` states
typically give finite lengths; the script measures the supplied state
without a `D` extrapolation.

## Directions and cuts

`direction` labels the propagation axis, parallel to one honeycomb bond
orientation. `cut` labels a transverse shift of the strip, not a new
direction or a successive propagation step.

| `direction` | Axis and parallel NN bonds | Central sites for `cut=1,2,3` |
| --- | --- | --- |
| 1 | `alpha`: `AD`, `BC`, `EF` | `BC`, `FE`, `DA` |
| 2 | `beta`: `AB`, `CF`, `DE` | `DE`, `BA`, `FC` |
| 3 | `gamma`: `AF`, `BE`, `CD` | `FA`, `DC`, `BE` |

One transfer cell advances three NN bond lengths. Six independent tensors
give all nine direction/cut combinations. Threefold symmetry relates the
directions for the other ansatze, so they return the three translated
cuts in direction 1. Keep the cut estimates separately.

## Saved quantities

`ansatz`, `D`, and `chi` identify the state and contraction as in
[evaluate.md](evaluate.md#saved-quantities). `eigenproblem` identifies the
ordinary row transfer; `length_unit` states the cell length. Each entry
of `directions` contains:

| Field | Meaning |
| --- | --- |
| `direction`, `cut` | The axis and transverse offset defined above. |
| `lambda2_over_lambda1_abs` | `r = abs(lambda2/lambda1)`, with eigenvalues sorted by decreasing magnitude. |
| `xi` | `-1/log(r)`, in transfer cells. Multiply by 3 for NN bond-length units. |
| `inverse_xi` | `-log(r)`, the inverse decay length per transfer cell. |
| `normalized_eigenvalues` | Every eigenvalue divided by `lambda1`, stored as `[real, imaginary]`. Phases describe oscillations along the strip. |

If `r=0`, the output is `xi=0`, `inverse_xi=null`: there is no subleading
propagating mode. If the inverse length is numerically zero (`<=1e-14`),
`xi=null` represents a nonfinite or unresolved longest length. These
measurement files contain no CTMRG residual fields; those are available
from `evaluate.py`.
