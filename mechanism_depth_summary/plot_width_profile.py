"""
Строит width_profile.png (SUMMARY_mechanism_depth_profile.md, "Результат 2 -
ширина") из data/tapnet_mechanism_ablation.json (C=16) и
data/tapnet_width_followup.json (C=64, все 4 механизма). Dropout сюда не входит -
он отдельный, изолированный фактор (см. "Результат 3" и plot_dropout_profile.py).

Малые кратные (2x2, один механизм на мини-панель): 16 линий (4 механизма x
train/untrain x C=16/C=64) в одной оси не читались бы. Цвет = обученная (красный)
/ необученная (серый); штрих/альфа = C=16 (сплошная) / C=64 (пунктир,
полупрозрачная) - dataviz-скилл, small multiples + цвет не единственный код.

Самодостаточный: оба JSON (data/) включены в папку, пути относительные.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "images"

COLOR_TRAIN = "#e34948"
COLOR_UNTRAIN = "#52514e"
MECH_ORDER = ["plain", "bn", "residual", "bn_residual"]


def load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)["results"]


def block_means(rows_list):
    if not rows_list:
        return None
    n_blocks = len(rows_list[0])
    return [statistics.mean(r[b]["nonadditive_frac_mean"] for r in rows_list) for b in range(n_blocks)]


def profile(results, net_type, mechanism=None):
    rows_list = [r["rows"] for r in results
                 if r["net_type"] == net_type and not r.get("skipped")
                 and (mechanism is None or r["mechanism"] == mechanism)]
    return block_means(rows_list)


def main():
    mech_results = load("tapnet_mechanism_ablation.json")
    width_results = load("tapnet_width_followup.json")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    for i, mech in enumerate(MECH_ORDER):
        ax = axes[i // 2, i % 2]
        c16_tr = profile(mech_results, "обученная", mechanism=mech)
        c16_un = profile(mech_results, "необученная", mechanism=mech)
        c64_tr = profile(width_results, "обученная", mechanism=f"{mech}_c64")
        c64_un = profile(width_results, "необученная", mechanism=f"{mech}_c64")
        blocks = list(range(len(c16_tr)))
        ax.plot(blocks, c16_tr, "o-", color=COLOR_TRAIN, label="обученная, C=16")
        ax.plot(blocks, c64_tr, "o--", color=COLOR_TRAIN, alpha=0.5, label="обученная, C=64")
        ax.plot(blocks, c16_un, "s-", color=COLOR_UNTRAIN, label="необученная, C=16")
        ax.plot(blocks, c64_un, "s--", color=COLOR_UNTRAIN, alpha=0.5, label="необученная, C=64")
        ax.set_title(mech, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.supxlabel("блок (глубина)")
    fig.supylabel("доля неаддитивных пар")
    fig.tight_layout()
    out_path = OUT_DIR / "width_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
