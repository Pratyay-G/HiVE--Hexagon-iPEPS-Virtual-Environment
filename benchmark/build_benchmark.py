"""Compare J2=0 energies using the bundled observations and quoted values."""
from pathlib import Path
import csv
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TEMP", ".")) / "hive-benchmark-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np

GREEN, RED = "#32936f", "#c65e58"
QMC, QMC_ERROR = -.5445525, .0000195


def observations():
    result = []
    for folder in HERE.glob("D_*"):
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["energy_comparison"] and metadata["D"] >= 3:
            obs = json.loads((folder / "observables.json").read_text(encoding="utf-8"))
            result.append((metadata["D"], obs["energy_per_site"]))
    return sorted(result, reverse=True)


def references():
    with (HERE / "quoted_energies.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["D"]) >= 3]
    naumann = [(int(r["D"]), float(r["E_per_site"])) for r in rows if r["source"] == "Naumann2024"]
    yang = [(int(r["D"]), float(r["E_per_site"])) for r in rows if r["source"] == "YangCorboz2025"]
    best = {d: min(e for dd, e in yang if dd == d) for d, _ in yang}
    others = [(d, e) for d, e in yang if e > best[d]]
    return sorted(naumann, reverse=True), sorted(best.items(), reverse=True), others


def series(ax, data, color, marker, size, zorder):
    ax.plot([1/d for d, _ in data], [e for _, e in data],
            color=color, marker=marker, ms=size, lw=1., zorder=zorder)


def draw(ax, ours, naumann, yang, others, inset=False):
    ax.axhspan(QMC-QMC_ERROR, QMC+QMC_ERROR, color="gray", alpha=.30, zorder=0)
    ax.axhline(QMC, color=".4", ls="--", lw=.8, zorder=1)
    series(ax, naumann, RED, "^", 6.8 if inset else 7.2, 3)
    series(ax, yang, GREEN, "s", 5.6 if inset else 6.2, 4)
    ax.scatter([1/d for d, _ in others], [e for _, e in others],
               marker="s", s=32 if inset else 38, facecolors="none",
               edgecolors=GREEN, linewidths=1., alpha=.6, zorder=4)
    series(ax, ours, "black", "o", 3.9 if inset else 4.5, 10)
    ax.tick_params(top=True, right=True, pad=2)


def main():
    plt.style.use(HERE / "publication.mplstyle")
    ours = observations()
    naumann, yang, others = references()
    fig, ax = plt.subplots(figsize=(6.4, 4.15))
    fig.subplots_adjust(left=.145, right=.995, bottom=.135, top=.985)
    draw(ax, ours, naumann, yang, others)
    ax.set(xlim=(.105, .348), ylim=(-.544750, -.5395),
           xticks=[.125, 1/6, .20, .25, 1/3],
           xticklabels=[r"$1/8$", r"$1/6$", r"$1/5$", r"$1/4$", r"$1/3$"],
           xlabel=r"$1/D$", ylabel=r"$E/J_1$")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    handles = [
        Patch(facecolor="gray", alpha=.30,
              label=r"QMC: $\pm 1\sigma$ [Condens.\ Matter Phys.\ \textbf{12}, 497 (2009)]"),
        Line2D([], [], color=RED, marker="^", ms=7.2, lw=1.,
               label="iPEPS: two-site square CTM "
                     r"[SciPost Phys.\ Lect.\ Notes \textbf{86} (2024)]"),
        Line2D([], [], color=GREEN, marker="s", ms=6.2, lw=1.,
               label="iPEPS: two-site honeycomb CTM with spin $U(1)$: best charge\n"
                     r"[Phys.\ Rev.\ B \textbf{113}, 085109 (2026)]"),
        Line2D([], [], color=GREEN, marker="s", mfc="none", ms=6.2,
               ls="", alpha=.6,
               label="iPEPS: two-site honeycomb CTM with spin $U(1)$: other charges\n"
                     r"[Phys.\ Rev.\ B \textbf{113}, 085109 (2026)]"),
        Line2D([], [], color="black", marker="o", ms=4.5, lw=1.,
               label="iPEPS: six-site hexagonal CTM (HiVE) "
                     r"[arXiv:XXXX.XXXXX]"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.01, .99),
              fontsize=9, handlelength=1.6, handletextpad=.6,
              labelspacing=.2, borderaxespad=.15)
    ins = ax.inset_axes([.17, .235, .48, .43], zorder=20)
    draw(ins, ours, naumann, yang, others, inset=True)
    ins.set(xlim=(.107, .173), ylim=(-.544596, -.544482),
            xticks=[1/9, 1/8, 1/7, 1/6],
            xticklabels=[r"$1/9$", r"$1/8$", r"$1/7$", r"$1/6$"],
            yticks=[-.54458, -.54454, -.54450])
    ins.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ins.tick_params(labelsize=9, pad=2)
    fig.savefig(HERE / "benchmark.png", dpi=250,
                bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
