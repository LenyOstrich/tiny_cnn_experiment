"""
Строит width_dropout_profile.png (SUMMARY_mechanism_depth_profile.md, "Результат 2
- ширина и dropout") из уже посчитанных JSON:
  - tapnet_mechanism_ablation.json          - базовые 4 механизма, C=16, без dropout
  - tapnet_width_and_dropout_followup.json  - ширина (bn_residual C=64) и dropout=0.1
    для всех 4 механизмов (plain/bn/residual/bn_residual + dropout, посчитано в два
    захода: исходно plain/bn_residual, потом добавлены bn/residual через
    tapnet_full_ablation.py --extra-dropout)

Левая панель - ширина (как раньше). Правая панель - dropout: раньше это были 2
линии "без/с dropout" на plain и bn_residual в одной панели; теперь, когда
dropout проверен для всех 4 механизмов, 8 линий в одной панели не читались бы -
поэтому малые кратные (2x2, один механизм на мини-панель), один и тот же смысл
цвета (синий = без dropout, оранжевый = + dropout 0.1) и маркера (круг сплошная
/ квадрат пунктир) во всех четырёх - секции dataviz-скилла про small multiples
и про то, что цвет не должен быть единственным кодом идентичности.

Полностью самодостаточный: оба нужных JSON (data/, ~140 КБ суммарно) включены в
эту папку, путь относительный - скрипт реально запускается прямо отсюда, без
остального проекта, нужен только `pip install matplotlib`.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "images"

# первые 2 слота валидированной категориальной палитры дataviz-скилла (references/palette.md)
COLOR_NO_DROPOUT = "#2a78d6"   # slot 1, синий
COLOR_DROPOUT = "#eb6834"      # slot 2, оранжевый
COLOR_TRAIN = "#e34948"        # slot 8, красный - ширина: обученная
COLOR_UNTRAIN = "#52514e"      # secondary ink, серый - ширина: необученная

MECH_ORDER = ["plain", "bn", "residual", "bn_residual"]


def load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)["results"]


def block_means(rows_list):
    """rows_list: список 'rows' (один на сид); rows - список словарей по блокам.
    Возвращает среднюю по сидам nonadditive_frac_mean на каждый блок."""
    if not rows_list:
        return None
    n_blocks = len(rows_list[0])
    return [statistics.mean(r[b]["nonadditive_frac_mean"] for r in rows_list) for b in range(n_blocks)]


def profile(results, net_type, mechanism=None, c=None):
    rows_list = [r["rows"] for r in results
                 if r["net_type"] == net_type and not r.get("skipped")
                 and (mechanism is None or r["mechanism"] == mechanism)
                 and (c is None or r.get("c") == c)]
    return block_means(rows_list)


def main():
    mech_results = load("tapnet_mechanism_ablation.json")
    follow_results = load("tapnet_width_and_dropout_followup.json")

    fig = plt.figure(figsize=(16, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2], wspace=0.28)

    # --- левая панель: ширина, bn_residual C=16 vs C=64 ---
    ax_w = fig.add_subplot(gs[0, 0])
    c16_tr = profile(mech_results, "обученная", mechanism="bn_residual")
    c16_un = profile(mech_results, "необученная", mechanism="bn_residual")
    c64_tr = profile(follow_results, "обученная", mechanism="bn_residual_c64")
    c64_un = profile(follow_results, "необученная", mechanism="bn_residual_c64")
    blocks = list(range(len(c16_tr)))
    ax_w.plot(blocks, c16_tr, "o-", color=COLOR_TRAIN, label="C=16, обученная")
    ax_w.plot(blocks, c64_tr, "o-", color=COLOR_TRAIN, alpha=0.5, label="C=64, обученная")
    ax_w.plot(blocks, c16_un, "o--", color=COLOR_UNTRAIN, label="C=16, необученная")
    ax_w.plot(blocks, c64_un, "o--", color=COLOR_UNTRAIN, alpha=0.5, label="C=64, необученная")
    ax_w.set_title("Ширина: bn_residual, C=16 vs C=64")
    ax_w.set_xlabel("блок (глубина)")
    ax_w.set_ylabel("доля неаддитивных пар")
    ax_w.set_ylim(0, 1)
    ax_w.legend(fontsize=8, loc="upper right")

    # --- правая панель: dropout, малые кратные по 4 механизмам ---
    ax_outer = fig.add_subplot(gs[0, 1])
    ax_outer.set_title("Dropout (обученные сети), по механизмам", pad=14)
    ax_outer.axis("off")

    gs_d = gs[0, 1].subgridspec(2, 2, wspace=0.12, hspace=0.5)
    axes_d = []
    for i, mech in enumerate(MECH_ORDER):
        share_x = axes_d[0] if axes_d else None
        share_y = axes_d[0] if axes_d else None
        ax = fig.add_subplot(gs_d[i // 2, i % 2], sharex=share_x, sharey=share_y)
        axes_d.append(ax)

        no_drop = profile(mech_results, "обученная", mechanism=mech)
        with_drop = profile(follow_results, "обученная", mechanism=f"{mech}_dropout")
        mech_blocks = list(range(len(no_drop)))
        ax.plot(mech_blocks, no_drop, "o-", color=COLOR_NO_DROPOUT, label="без dropout")
        ax.plot(mech_blocks, with_drop, "s--", color=COLOR_DROPOUT, label="+ dropout 0.1")
        ax.set_title(mech, fontsize=10)
        ax.set_ylim(0, 1)
        ax.label_outer()
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    out_path = OUT_DIR / "width_dropout_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Сохранено:", out_path)


if __name__ == "__main__":
    main()
