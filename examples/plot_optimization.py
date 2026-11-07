"""Plot energy and staggered magnetization from an optimization CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="optimization.csv written by optimize.py")
    parser.add_argument("--output", type=Path, help="PNG path (defaults beside the CSV)")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"chi", "step", "energy_per_site", "staggered_magnetization"}
        if not required.issubset(reader.fieldnames or ()):
            parser.error(f"{args.csv} must contain columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        parser.error(f"{args.csv} contains no optimization rows")

    groups = {}
    for row in rows:
        chi = row["chi"]
        groups.setdefault(chi, []).append(row)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (energy_ax, magnetization_ax) = plt.subplots(
        2, 1, figsize=(9.6, 7.2), sharex=True, constrained_layout=True)
    figure.suptitle("Optimization convergence", fontsize=19, fontweight="bold",
                    ha="left", x=0.09)
    figure.text(0.09, 0.94, args.csv.parent.name, fontsize=10.5, color="#617184")

    for chi, points in groups.items():
        steps = [int(row["step"]) for row in points]
        energies = [float(row["energy_per_site"]) for row in points]
        magnetizations = [float(row["staggered_magnetization"]) for row in points]
        label = f"chi={chi}" if len(groups) > 1 else None
        energy_ax.plot(steps, energies, marker="o", linewidth=2.4, markersize=4.5,
                       label=label)
        magnetization_ax.plot(steps, magnetizations, marker="o", linewidth=2.4,
                              markersize=4.5, label=label)
        energy_ax.annotate(f"Final: {energies[-1]:.6f}", (steps[-1], energies[-1]),
                           xytext=(-8, 12), textcoords="offset points", ha="right")
        magnetization_ax.annotate(
            f"Final: {magnetizations[-1]:.6f}",
            (steps[-1], magnetizations[-1]), xytext=(-8, 12),
            textcoords="offset points", ha="right")

    energy_ax.set_title("Energy per site", loc="left", fontsize=13, fontweight="bold")
    energy_ax.set_ylabel("Energy per site ($J_1$)")
    magnetization_ax.set_title("Staggered magnetization", loc="left",
                               fontsize=13, fontweight="bold")
    magnetization_ax.set_ylabel("Magnetization")
    magnetization_ax.set_xlabel("Optimization step")
    if len(groups) > 1:
        energy_ax.legend()
        magnetization_ax.legend()
    for axis in (energy_ax, magnetization_ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.04)

    output = args.output or args.csv.with_name("optimization_convergence.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
