# Four calculations

Run each script from `src_code`. It writes its results to `output/`; use
`--output` to choose another destination. Each script has a matching guide.

| Script | Calculation and guide |
| --- | --- |
| `python examples/optimize.py --ansatz c3rotated2tensors --D 2 --chi 4` | [Optimize the Heisenberg ground state](optimize.md), with examples for all three ansatze and several `D`, `chi`. |
| `python examples/evaluate.py` | [Measure energy, local spins and NN correlations](evaluate.md) of a given tensor. |
| `python examples/correlation_length.py` | [Measure correlation decay](correlation_length.md) in each inequivalent direction and cut. |
| `python examples/renyi_entropy.py` | [Measure cylinder Renyi-2 entropy](renyi_entropy.md) as its circumference grows. |

The three measurements use `tensor.npz` by default: an existing optimized
two-C3 sample with `D=2`, `J1=1`, and stored `chi=4`. Replace this positional
argument with any tensor saved by `optimize.py`. The same scripts support
`3legSymmetrized1tensor`, `c3rotated2tensors`, and `unrestricted6tensors`; the file records its ansatz.

Plot the energy and staggered-magnetization history from an optimization run:

```sh
python examples/plot_optimization.py output/3legSymmetrized1tensor_D2/optimization.csv
```

The PNG is saved beside the CSV as `optimization_convergence.png`. Pass
`--output path/to/plot.png` to choose another destination.

`python -m examples.evaluate` is equivalent to the direct command above.
Add `--device cuda` for a GPU. All quantities use spin-1/2 operators and
natural logarithms. Output fields, units and cut labels are defined in the
matching guides; generated numerical results are not distributed here.
