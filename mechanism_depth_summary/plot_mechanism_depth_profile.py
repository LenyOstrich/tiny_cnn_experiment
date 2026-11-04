"""
Строит mechanism_depth_profile.png (SUMMARY_mechanism_depth_profile.md,
"Результат 1 - какой механизм гасит связи с глубиной") из
data/tapnet_mechanism_ablation.json - 4 варианта механизма (plain/bn/residual/
bn_residual), C=16, 5 обученных + 5 необученных сидов каждый.

Два панели - обученные / необученные (контроль) сети, по 4 линии (один
механизм = одна линия, тот же цвет в обеих панелях) с тенью ±1 std по сидам
вокруг каждой точки.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "images"

MECHANISMS = [
    ("plain", "plain", "#52514e"),                       # secondary ink, серый
    ("bn", "+BatchNorm", "#2a78d6"),                      # slot 1, синий
    ("residual", "+residual", "#008300"),                 # slot 6, зелёный
    ("bn_residual", "+BN+residual (BasicBlock)", "#e34948"),  # slot 8, красный
]


def load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)["results"]


def per_seed_block_values(results, net_type, mechanism):
    """Возвращает (n_blocks, n_seeds) список: для каждого блока - список значений
    nonadditive_frac_mean по сидам (нужно для среднего И std по сидам)."""
    rows_list = [r["rows"] for r in results
                 if r["net_type"] == net_type and r["mechanism"] == mechanism and not r.get("skipped")]
    n_blocks = len(rows_list[0])
    return [[rows[b]["nonadditive_frac_mean"] for rows in rows_list] for b in range(n_blocks)]


def mean_std(per_block_values):
    means = [statistics.mean(v) for v in per_block_values]
    stds = [statistics.stdev(v) if len(v) > 1 else 0.0 for v in per_block_values]
    return means, stds


def plot_panel(ax, results, net_type, title, show_legend):
    for mech, label, color in MECHANISMS:
        per_block = per_seed_block_values(results, net_type, mech)
        means, stds = mean_std(per_block)
        blocks = list(range(len(means)))
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.plot(blocks, means, "o-", color=color, label=label)
        ax.fill_between(blocks, lo, hi, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("блок (глубина)")
    ax.set_ylim(0, 1)
    if show_legend:
        ax.legend(fontsize=9, loc="upper left")


def main():
    results = load("tapnet_mechanism_ablation.json")

    fig, (ax_tr, ax_un) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    plot_panel(ax_tr, results, "обученная", "Обученные сети", show_legend=True)
    ax_tr.set_ylabel("доля неаддитивных пар (synergy+subadditive)")
    plot_panel(ax_un, results, "необученная", "Необученные сети (контроль)", show_legend=False)

    out_path = OUT_DIR / "mechanism_depth_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
