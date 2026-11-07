# Measure cylinder Renyi-2 entropy

`renyi_entropy.py` measures entanglement between the two half-cylinders of
a saved iPEPS. It joins one and two copies of the opposite CTM boundaries
and evaluates

`S2(L) = -log Tr(rho_L^2) = 2 log Tr(T1^(L/2)) - log Tr(T2^(L/2))`.

Run from `src_code`:

```sh
python examples/renyi_entropy.py
python examples/renyi_entropy.py output/unrestricted6tensors_D3/tensor.npz --chi 4 --lengths 2 4 6 8 12 --output output/unrestricted6tensors_D3/S2.json
```

The first command reads `examples/tensor.npz` and writes
`output/renyi_entropy.json`. The default `chi=4` keeps the full dense
two-replica transfer matrix small; its dimension is `chi^4`. Add
`--device cuda` for a GPU. Increasing the circumference generally
increases the entropy; the script returns the curve without fitting
an area-law coefficient or a constant term.

## Saved quantities

| Field | Physical meaning |
| --- | --- |
| `ansatz`, `D`, `chi` | Wave-function ansatz, virtual bond dimension, and CTM boundary dimension. |
| `L_cut_bonds` | Number of virtual bonds cut around the cylinder. Values must be positive and even. The geometric circumference is `3L/2` NN bond lengths. |
| `logarithm` | `natural`: entropies are in nats, not bits. |
| `cuts` | One entropy curve for each inequivalent direction and transverse cut. |
| `cuts[].direction` | Direction along the cylinder circumference: `1=alpha`, `2=beta`, `3=gamma`. |
| `cuts[].cut` | Which of the three translated cuts crosses the six-site cell. It shifts the partition transversely while retaining its direction. |
| `cuts[].S2` | Renyi-2 entropies in the same order as `L_cut_bonds`. |

The three directions and offsets use the same labels as
[correlation_length.md](correlation_length.md#directions-and-cuts).
For direction 1, the two alternating crossed bonds identify each cut:

| `cut` | Crossed NN bonds |
| --- | --- |
| 1 | `DC` (`gamma`) and `AB` (`beta`) |
| 2 | `BE` (`gamma`) and `CF` (`beta`) |
| 3 | `FA` (`gamma`) and `ED` (`beta`) |

Six independent tensors return nine curves. Neel and two-C3 states return
three translated cuts, with the rotated directions related by symmetry.
The curves are kept separately because the cut can pass through different
bond patterns. A product state has `S2=0`; stronger entanglement gives a
larger entropy. No CTMRG residuals are stored in this output.
