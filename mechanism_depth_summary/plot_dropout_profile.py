"""
Строит dropout_profile.png (SUMMARY_mechanism_depth_profile.md, "Результат 3 -
dropout") из data/tapnet_mechanism_ablation.json (C=16, без dropout - часть 1)
и data/tapnet_dropout_followup.json (C=16, dropout=0.1, все 4 механизма -
часть 3, изолированный прогон). Dropout сам по себе здесь не считается - только
читает оба готовых файла и сравнивает их.

Малые кратные (2x2, один механизм на мини-панель), как в plot_width_profile.py:
обученная/необученная = цвет (красный/серый), без/с dropout = штрих (сплошная/
пунктир, с dropout полупрозрачнее).
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


def profile(results, net_type, mechanism):
    rows_list = [r["rows"] for r in results
                 if r["net_type"] == net_type and r["mechanism"] == mechanism and not r.get("skipped")]
    return block_means(rows_list)


def main():
    no_dropout = load("tapnet_mechanism_ablation.json")
    dropout = load("tapnet_dropout_followup.json")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    for i, mech in enumerate(MECH_ORDER):
        ax = axes[i // 2, i % 2]
        nd_tr = profile(no_dropout, "обученная", mech)
        nd_un = profile(no_dropout, "необученная", mech)
        d_tr = profile(dropout, "обученная", f"{mech}_dropout")
        d_un = profile(dropout, "необученная", f"{mech}_dropout")
        blocks = list(range(len(nd_tr)))
        ax.plot(blocks, nd_tr, "o-", color=COLOR_TRAIN, label="обученная, без dropout")
        ax.plot(blocks, d_tr, "o--", color=COLOR_TRAIN, alpha=0.5, label="обученная, +dropout 0.1")
        ax.plot(blocks, nd_un, "s-", color=COLOR_UNTRAIN, label="необученная, без dropout")
        ax.plot(blocks, d_un, "s--", color=COLOR_UNTRAIN, alpha=0.5, label="необученная, +dropout 0.1")
        ax.set_title(mech, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.supxlabel("блок (глубина)")
    fig.supylabel("доля неаддитивных пар")
    fig.tight_layout()
    out_path = OUT_DIR / "dropout_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
