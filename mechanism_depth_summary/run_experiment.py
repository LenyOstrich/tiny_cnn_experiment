"""
Самостоятельный скрипт - воспроизводит с нуля (обучение + причинный тест) все
данные для SUMMARY_mechanism_depth_profile.md: какой архитектурный элемент
(BatchNorm, residual-связь, ширина слоя, dropout) гасит попарные причинные
взаимодействия каналов в TapNet (управляемая MNIST-сеть, 6 блоков).

Не требует остального проекта `cnn_mi_experiments` - только стандартные пакеты
(см. requirements.txt рядом). Пути ниже всегда относительны расположения этого
файла, так что папку можно перенести куда угодно.

Семейство "TapNet": conv-стебель + DEPTH одинаковых по ширине C блоков +
avgpool + fc. Каждый блок: conv1 -> (BN?) -> ReLU [точка абляции "a"] ->
(dropout?) -> conv2 -> (BN?) -> (+identity блока?) -> ReLU. Причинный тест
S(A,B)=TE(A,B)-TE(A)-TE(B) (S>0 - синергия), TE=baseline-ablated, do(zero channel)
как интервенция, FDR-BY поправка + порог эффекта tau.
ИСПРАВЛЕНИЕ ЗНАКА (после разбора в чате): раньше было S=TE(A)+TE(B)-TE(A,B) -
перевёрнутое относительно synergy_experiments/experiment_common_task.ipynb
соглашение, из-за которого synergy/subadditive в data/*.json были помечены
наоборот (n_syn_mean/n_sub_mean там переставлены местами постфактум, без
повторного прогона).

Три части, запускаются последовательно одним main() (~7 часов на CPU
суммарно - см. тайминги в комментариях ниже; части можно гонять по
отдельности флагами):

  1. МЕХАНИЗМ (MECHANISMS) - 4 варианта (plain/bn/residual/bn_residual),
     C=16, DEPTH=6, полный перебор пар (120), 5 обученных + 5 необученных
     сидов на вариант. ~90 минут. -> data/tapnet_mechanism_ablation.json

  2. ШИРИНА (WIDTH_CONFIGS) - все 4 механизма при C=64 (случайная подвыборка
     пар, 300 на блок), 2+2 сида на вариант. ~3.5 часа. (Изначально проверялся
     только bn_residual - механизм, который и так давал наименьшую долю связей
     из четырёх; plain/bn/residual добавлены, чтобы проверить, убивает ли
     ширина связи сама по себе, без BatchNorm+residual.) -> data/tapnet_width_followup.json

  3. DROPOUT (DROPOUT_CONFIGS) - все 4 механизма при C=16 с dropout=0.1
     (channel-wise, после ReLU точки абляции - identity в .eval(), влияет
     только на обучение), 5 обученных + 5 необученных сидов на вариант, на
     3 эпохи больше, чем в части 1 (без запаса plain+dropout не сходится
     стабильно). Изолированная фаза - не пересчитывает и не трогает части 1
     и 2, сравнивается с их уже готовыми данными при построении графика.
     ~2 часа. -> data/tapnet_dropout_followup.json

Флаги командной строки (без флагов - все три части по очереди):
  --mechanisms-only   только часть 1
  --width-only        только часть 2
  --dropout-only      только часть 3 (использует уже готовый
                       data/tapnet_mechanism_ablation.json как базу "без
                       dropout" - его пересчитывать не нужно)
  --skip-mechanisms   части 2+3, без части 1

После - постройте графики: `python plot_mechanism_depth_profile.py`,
`python plot_width_profile.py` и `python plot_dropout_profile.py` (читают те
же data/*.json).
"""
import gzip
import itertools
import json
import os
import struct
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data" / "mnist_raw"
RESULTS_DIR = HERE / "data"
SEED = 42

# ---------------------------------------------------------------- причинный тест ---
# То же самое, что mean_test/fdr/ALPHA/FDR_METHOD/TAU_REL в synergy_experiments/
# resnet_layer_interactions.py основного проекта - здесь продублировано (а не
# импортировано), чтобы этот файл не тянул за собой torchvision и весь
# ResNet-специфичный код, которые для TapNet не нужны.
ALPHA = 0.05
FDR_METHOD = "fdr_by"      # Benjamini-Yekutieli: парные тесты структурно зависимы
TAU_REL = 0.02             # порог размера эффекта, в долях std логита класса


def mean_test(values, tau):
    _, p = stats.ttest_1samp(values, 0.0)
    return {"ate": float(np.mean(values)), "p": float(np.nan_to_num(p, nan=1.0))}


def fdr(rows, alpha, method):
    if not rows:
        return rows
    reject, p_adj, _, _ = multipletests([r["p"] for r in rows], alpha=alpha, method=method)
    for r, ok, p in zip(rows, reject, p_adj):
        r["p_fdr"], r["significant"] = float(p), bool(ok)
    return rows


# ----------------------------------------------------------------------- конфиг ---
C = 16                  # ширина всех блоков в части 1 (фиксированная - residual без projection)
DEPTH = 6                # блоков в стеке - профиль по глубине внутри одного варианта
N_CAL = int(os.environ.get("TAP_N_CAL", "128"))
N_CLASSES = 10
EPOCHS = int(os.environ.get("TAP_EPOCHS", "5"))
LR = 1e-3
N_TRAIN_SEEDS = int(os.environ.get("TAP_N_TRAIN_SEEDS", "5"))
N_UNTRAIN_SEEDS = int(os.environ.get("TAP_N_UNTRAIN_SEEDS", "5"))

MECHANISMS = [
    ("plain", False, False),        # ни BN, ни residual - как в маленькой сети
    ("bn", True, False),
    ("residual", False, True),
    ("bn_residual", True, True),    # структура BasicBlock ResNet-18 (фикс. ширина, без downsample)
]

MECHANISM_RESULTS_PATH = RESULTS_DIR / "tapnet_mechanism_ablation.json"
FOLLOWUP_RESULTS_PATH = RESULTS_DIR / "tapnet_width_followup.json"

N_CAL_WIDTH = 64     # урезано вдвое относительно части 1 (128) - шире сеть, дороже причинный тест
MAX_PAIRS_WIDE = 300  # для C=64: C(64,2)=2016, тестируем случайную подвыборку

WIDTH_CONFIGS = [
    # (name, c, use_bn, use_residual, n_train_seeds, n_untrain_seeds, n_cal, max_pairs)
    ("plain_c64", 64, False, False, 2, 2, N_CAL_WIDTH, MAX_PAIRS_WIDE),
    ("bn_c64", 64, True, False, 2, 2, N_CAL_WIDTH, MAX_PAIRS_WIDE),
    ("residual_c64", 64, False, True, 2, 2, N_CAL_WIDTH, MAX_PAIRS_WIDE),
    ("bn_residual_c64", 64, True, True, 2, 2, N_CAL_WIDTH, MAX_PAIRS_WIDE),
]

# --------------------------------------------------------------- часть 3: dropout ---
DROPOUT_RESULTS_PATH = RESULTS_DIR / "tapnet_dropout_followup.json"
DROPOUT_P = 0.1                    # channel-wise, после ReLU точки абляции (TapBlock.to_tap)
DROPOUT_EPOCHS = EPOCHS + 3        # без запаса plain+dropout не сходится стабильно (acc~0.12)

DROPOUT_CONFIGS = [
    # (name, use_bn, use_residual, n_train_seeds, n_untrain_seeds) - C=16, как в части 1,
    # сравнивается с уже готовым data/tapnet_mechanism_ablation.json (та же архитектура,
    # без dropout) - изолированная фаза ничего в части 1 не пересчитывает.
    ("plain_dropout", False, False, 5, 5),
    ("bn_dropout", True, False, 5, 5),
    ("residual_dropout", False, True, 5, 5),
    ("bn_residual_dropout", True, True, 5, 5),
]


# ---------------------------------------------------------------- данные (MNIST) ---
def _load_mnist():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {"tr_x": "train-images-idx3-ubyte.gz", "tr_y": "train-labels-idx1-ubyte.gz",
             "te_x": "t10k-images-idx3-ubyte.gz", "te_y": "t10k-labels-idx1-ubyte.gz"}
    for f in files.values():
        if not (DATA_ROOT / f).exists():
            urlretrieve(base + f, DATA_ROOT / f)

    def img(p):
        with gzip.open(DATA_ROOT / p, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, r, c).copy()

    def lab(p):
        with gzip.open(DATA_ROOT / p, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64).copy()

    return img(files["tr_x"]), lab(files["tr_y"]), img(files["te_x"]), lab(files["te_y"])


class Data:
    def __init__(self):
        xtr, ytr, xte, yte = _load_mnist()
        self.xtr, self.ytr, self.xte, self.yte = map(torch.from_numpy, (xtr, ytr, xte, yte))
        perm = torch.randperm(len(self.ytr), generator=torch.Generator().manual_seed(SEED))
        self.train_idx, self.sel_idx = perm[:48000], perm[48000:]

    def prep(self, xu8):
        return xu8.float().div_(255)

    def x_train(self, idx):
        return self.prep(self.xtr[idx])

    def y_train(self, idx):
        return self.ytr[idx]


# ------------------------------------------------------------------- TapNet ---
class TapBlock(nn.Module):
    def __init__(self, c, use_bn, use_residual, dropout_p=0.0):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.drop = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

    def to_tap(self, x):
        """Точка абляции 'a' - вход в conv2, аналог pre_conv2 хука в ResNet-блоке.
        Dropout - identity в .eval(), поэтому на сам причинный тест (всегда в eval) не
        влияет, только на то, что обучение "впечатывает" в веса."""
        return self.drop(F.relu(self.bn1(self.conv1(x))))

    def from_tap(self, a, x_in):
        out = self.bn2(self.conv2(a))
        if self.use_residual:
            out = out + x_in
        return F.relu(out)


class TapNet(nn.Module):
    """conv-стебель -> DEPTH блоков фиксированной ширины C -> avgpool -> fc."""

    def __init__(self, c=C, depth=DEPTH, use_bn=False, use_residual=False, dropout_p=0.0, n_out=10):
        super().__init__()
        self.depth = depth
        self.stem_conv = nn.Conv2d(1, c, 5, padding=2)
        self.blocks = nn.ModuleList([TapBlock(c, use_bn, use_residual, dropout_p) for _ in range(depth)])
        self.fc = nn.Linear(c, n_out)

    def stem(self, x):
        return F.max_pool2d(F.relu(self.stem_conv(x)), 2)   # 28x28 -> 14x14

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b.from_tap(b.to_tap(x), x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def capture(self, x, block_idx):
        x = self.stem(x)
        for i in range(block_idx):
            b = self.blocks[i]
            x = b.from_tap(b.to_tap(x), x)
        a = self.blocks[block_idx].to_tap(x)
        return a, x   # (точка абляции, вход блока - для residual identity)

    def continue_from(self, a, x_in, block_idx):
        x = self.blocks[block_idx].from_tap(a, x_in)
        for i in range(block_idx + 1, self.depth):
            b = self.blocks[i]
            x = b.from_tap(b.to_tap(x), x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def sanity_check(model, x):
    model.eval()
    with torch.no_grad():
        baseline = model(x)
        for block_idx in range(model.depth):
            a, x_in = model.capture(x, block_idx)
            recon = model.continue_from(a, x_in, block_idx)
            diff = (recon - baseline).abs().max().item()
            assert diff < 1e-4, f"block {block_idx}: recon diff {diff}"


def train_net(model, data, epochs, lr, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = data.train_idx
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(idx), generator=g)
        for i in range(0, len(idx), 128):
            b = idx[order[i:i + 128]]
            xb, yb = data.x_train(b), data.y_train(b)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def eval_accuracy(model, data, n=2000):
    xb, yb = data.prep(data.xte[:n]), data.yte[:n]
    pred = model(xb).argmax(1)
    return float((pred == yb).float().mean())


def calibrate_batchnorm(model, data, n_images=512, seed=SEED):
    """Как в synergy_experiments/resnet_layer_interactions.py: случайной сети нужны
    реальные running_mean/var BatchNorm перед причинным тестом, не заводские 0/1.
    Для вариантов без BN (nn.Identity вместо BatchNorm2d) - безвредный no-op."""
    model.train()
    rng = np.random.default_rng(seed)
    idx = torch.from_numpy(rng.choice(len(data.train_idx), size=n_images, replace=False))
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            model(data.x_train(data.train_idx[idx[i:i + 64]]))
    model.eval()


def calib_batch(data, class_idx, n, seed, model=None, require_correct=True):
    cand = data.sel_idx[data.ytr[data.sel_idx] == class_idx]
    if require_correct and model is not None:
        with torch.no_grad():
            pred = model(data.prep(data.xtr[cand])).argmax(1)
        cand = cand[pred == class_idx]
        if len(cand) < 16:
            raise RuntimeError(f"мало правильно классифицированных примеров класса {class_idx}: {len(cand)}")
    rng = np.random.default_rng(seed + class_idx)
    pick = cand[torch.from_numpy(rng.choice(len(cand), size=min(n, len(cand)), replace=False))]
    return data.prep(data.xtr[pick])


def batched_variant_logits(model, a, x_in, block_idx, variants, chunk=64):
    B = a.shape[0]
    out_rows = []
    with torch.no_grad():
        for start in range(0, len(variants), chunk):
            bv = variants[start:start + chunk]
            k = len(bv)
            a_rep = a.unsqueeze(0).repeat(k, 1, 1, 1, 1).clone()
            for vi, channels in enumerate(bv):
                for c in channels:
                    a_rep[vi, :, c, :, :] = 0.0
            xin_rep = x_in.unsqueeze(0).repeat(k, 1, 1, 1, 1)
            a_flat = a_rep.view(k * B, *a.shape[1:])
            xin_flat = xin_rep.reshape(k * B, *x_in.shape[1:])
            logits = model.continue_from(a_flat, xin_flat, block_idx)
            out_rows.append(logits.view(k, B, -1).cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def analyze_block_class(model, data, block_idx, class_idx, n_cal, require_correct, seed, max_pairs=None):
    assert not model.training, (
        "модель в .train()! BatchNorm в train-режиме считает статистики по текущему батчу - "
        "абляции разных вариантов перемешаются в одном батче и дадут ложные взаимодействия "
        "(см. docstring calibrate_batchnorm). Вызовите model.eval() перед причинным тестом.")
    x = calib_batch(data, class_idx, n_cal, seed, model=model if require_correct else None,
                     require_correct=require_correct)
    a, x_in = model.capture(x, block_idx)
    with torch.no_grad():
        baseline = model.continue_from(a, x_in, block_idx)[:, class_idx].cpu().numpy()
    tau = TAU_REL * float(np.std(baseline, ddof=1))

    n_channels = a.shape[1]
    single_variants = [(c,) for c in range(n_channels)]
    single_logits = batched_variant_logits(model, a, x_in, block_idx, single_variants)[:, :, class_idx]
    te_single = baseline[None, :] - single_logits
    te_cache = {c: te_single[c] for c in range(n_channels)}

    # --- значимость одиночных каналов (critical/harmful/neutral) - нужна, чтобы отличить
    # "пар нет, потому что одиночные эффекты ниже порога" от "пар нет при обильных одиночных" ---
    singles = []
    for c in range(n_channels):
        r = mean_test(te_single[c], tau)
        r["channel"] = c
        singles.append(r)
    fdr(singles, ALPHA, FDR_METHOD)
    n_critical = sum(r["significant"] and r["ate"] > tau for r in singles)
    n_harmful = sum(r["significant"] and r["ate"] < -tau for r in singles)
    n_neutral = n_channels - n_critical - n_harmful

    all_pairs = list(itertools.combinations(range(n_channels), 2))
    if max_pairs is not None and len(all_pairs) > max_pairs:
        rng = np.random.default_rng(seed + class_idx + block_idx)
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    pair_logits = batched_variant_logits(model, a, x_in, block_idx, pairs)[:, :, class_idx]
    rows = []
    for k, (i, j) in enumerate(pairs):
        te_joint = baseline[None, :] - pair_logits[k:k + 1]
        s_vals = te_joint[0] - te_cache[i] - te_cache[j]
        r = mean_test(s_vals, tau)
        r.update(a=i, b=j, S=r["ate"])
        rows.append(r)
    fdr(rows, ALPHA, FDR_METHOD)
    for r in rows:
        ok = r["significant"] and abs(r["S"]) > tau
        r["sign"] = "synergy" if ok and r["S"] > 0 else "subadditive" if ok and r["S"] < 0 else "additive"
    n_syn = sum(r["sign"] == "synergy" for r in rows)
    n_sub = sum(r["sign"] == "subadditive" for r in rows)
    return {"n_pairs": len(rows), "n_pairs_possible": len(all_pairs), "n_syn": n_syn, "n_sub": n_sub,
            "nonadditive_frac": (n_syn + n_sub) / len(rows),
            "n_channels": n_channels, "n_critical": n_critical, "n_harmful": n_harmful, "n_neutral": n_neutral,
            "single_sig_frac": (n_critical + n_harmful) / n_channels}


def analyze_net(model, data, label, require_correct, seed, max_pairs=None, n_cal=None):
    if n_cal is None:
        n_cal = N_CAL
    rows = []
    for block_idx in range(model.depth):
        fracs, syns, subs, sig_fracs = [], [], [], []
        n_pairs_tested = n_pairs_possible = None
        for class_idx in range(N_CLASSES):
            res = analyze_block_class(model, data, block_idx, class_idx, n_cal, require_correct, seed,
                                       max_pairs=max_pairs)
            fracs.append(res["nonadditive_frac"])
            syns.append(res["n_syn"])
            subs.append(res["n_sub"])
            sig_fracs.append(res["single_sig_frac"])
            n_pairs_tested, n_pairs_possible = res["n_pairs"], res["n_pairs_possible"]
        rows.append({"block": block_idx, "nonadditive_frac_mean": float(np.mean(fracs)),
                     "n_syn_mean": float(np.mean(syns)), "n_sub_mean": float(np.mean(subs)),
                     "single_sig_frac_mean": float(np.mean(sig_fracs)),
                     "n_pairs": n_pairs_tested, "n_pairs_possible": n_pairs_possible})
        print(f"    [{label}] block={block_idx}: доля неаддитивных={np.mean(fracs):.3f} "
              f"(syn={np.mean(syns):.2f}, sub={np.mean(subs):.2f} /{n_pairs_tested}), "
              f"singles значимых={np.mean(sig_fracs):.2f}")
    return rows


# ------------------------------------------------------------ часть 1: механизм ---
def save_mechanism_checkpoint(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MECHANISM_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": {"C": C, "DEPTH": DEPTH, "N_CAL": N_CAL, "EPOCHS": EPOCHS,
                               "N_TRAIN_SEEDS": N_TRAIN_SEEDS, "N_UNTRAIN_SEEDS": N_UNTRAIN_SEEDS,
                               "ALPHA": ALPHA, "TAU_REL": TAU_REL, "FDR_METHOD": FDR_METHOD},
                   "results": all_results}, f, indent=2, ensure_ascii=False)


def run_mechanisms(data):
    all_results = []
    for name, use_bn, use_residual in MECHANISMS:
        print(f"\n=== механизм: {name} (bn={use_bn}, residual={use_residual}) ===")

        for seed in range(N_TRAIN_SEEDS):
            t0 = time.perf_counter()
            torch.manual_seed(1000 + seed)
            model = TapNet(use_bn=use_bn, use_residual=use_residual)
            x_check = data.prep(data.xtr[:4])
            sanity_check(model, x_check)
            model = train_net(model, data, EPOCHS, LR, seed)
            acc = eval_accuracy(model, data)
            print(f"  [обученная seed={seed}] acc={acc:.3f}")
            rows = analyze_net(model, data, f"обученная seed={seed}", require_correct=True, seed=SEED)
            all_results.append({"mechanism": name, "use_bn": use_bn, "use_residual": use_residual,
                                 "net_type": "обученная", "seed": seed, "accuracy": acc, "rows": rows,
                                 "seconds": time.perf_counter() - t0})
            save_mechanism_checkpoint(all_results)

        for seed in range(N_UNTRAIN_SEEDS):
            t0 = time.perf_counter()
            torch.manual_seed(2000 + seed)
            model = TapNet(use_bn=use_bn, use_residual=use_residual)
            x_check = data.prep(data.xtr[:4])
            sanity_check(model, x_check)
            calibrate_batchnorm(model, data, seed=SEED + seed)
            acc = eval_accuracy(model, data)
            print(f"  [необученная seed={seed}] acc={acc:.3f}")
            rows = analyze_net(model, data, f"необученная seed={seed}", require_correct=False, seed=SEED)
            all_results.append({"mechanism": name, "use_bn": use_bn, "use_residual": use_residual,
                                 "net_type": "необученная", "seed": seed, "accuracy": acc, "rows": rows,
                                 "seconds": time.perf_counter() - t0})
            save_mechanism_checkpoint(all_results)

    print("\nГотово.", MECHANISM_RESULTS_PATH)
    return all_results


# --------------------------------------------------------------- часть 2: ширина ---
def save_followup_checkpoint(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FOLLOWUP_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": {"WIDTH_CONFIGS": WIDTH_CONFIGS, "EPOCHS": EPOCHS, "DEPTH": DEPTH},
                   "results": all_results}, f, indent=2, ensure_ascii=False)


def save_dropout_checkpoint(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DROPOUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": {"DROPOUT_CONFIGS": DROPOUT_CONFIGS, "DROPOUT_P": DROPOUT_P,
                               "EPOCHS": EPOCHS, "DROPOUT_EPOCHS": DROPOUT_EPOCHS,
                               "C": C, "DEPTH": DEPTH},
                   "results": all_results}, f, indent=2, ensure_ascii=False)


def run_dropout(data):
    """Часть 3, изолированно: dropout=0.1 для всех 4 механизмов при C=16 (то же DEPTH,
    тот же причинный тест). Не трогает результаты частей 1 и 2 - пишет только в
    data/tapnet_dropout_followup.json; базой "без dropout" для сравнения на графике
    служит уже готовый data/tapnet_mechanism_ablation.json (часть 1)."""
    all_results = []
    for name, use_bn, use_residual, n_tr, n_un in DROPOUT_CONFIGS:
        run_followup_one(name, C, use_bn, use_residual, n_tr, n_un, N_CAL, None,
                          data, all_results, epochs=DROPOUT_EPOCHS, dropout_p=DROPOUT_P,
                          save_fn=save_dropout_checkpoint)
    print("\nГотово.", DROPOUT_RESULTS_PATH)
    return all_results


def run_followup_one(name, c, use_bn, use_residual, n_train_seeds, n_untrain_seeds, n_cal, max_pairs,
                      data, all_results, epochs=EPOCHS, min_acc=0.5, dropout_p=0.0, save_fn=None):
    """save_fn - какой чекпойнт писать после каждого сида (часть 2: save_followup_checkpoint
    в data/tapnet_width_followup.json; часть 3: save_dropout_checkpoint в
    data/tapnet_dropout_followup.json), чтобы обе фазы не смешивали результаты в одном файле."""
    if save_fn is None:
        save_fn = save_followup_checkpoint
    print(f"\n=== {name} (C={c}, bn={use_bn}, residual={use_residual}, dropout={dropout_p}) ===")
    for seed in range(n_train_seeds):
        t0 = time.perf_counter()
        torch.manual_seed(1000 + seed)
        model = TapNet(c=c, use_bn=use_bn, use_residual=use_residual, dropout_p=dropout_p)
        x_check = data.prep(data.xtr[:4])
        sanity_check(model, x_check)
        model = train_net(model, data, epochs, LR, seed)
        acc = eval_accuracy(model, data)
        print(f"  [обученная seed={seed}] acc={acc:.3f}")
        if acc < min_acc:
            print(f"  ПРОПУСК: acc={acc:.3f} < {min_acc} - обучение не сошлось, причинный тест на "
                  f"неклассифицирующей сети бессмыслен (мало правильно классифицированных примеров на класс).")
            all_results.append({"mechanism": name, "c": c, "use_bn": use_bn, "use_residual": use_residual,
                                 "dropout_p": dropout_p, "net_type": "обученная", "seed": seed, "accuracy": acc,
                                 "rows": None, "skipped": True, "seconds": time.perf_counter() - t0})
            save_fn(all_results)
            continue
        rows = analyze_net(model, data, f"обученная seed={seed}", True, SEED, max_pairs=max_pairs, n_cal=n_cal)
        all_results.append({"mechanism": name, "c": c, "use_bn": use_bn, "use_residual": use_residual,
                             "dropout_p": dropout_p, "net_type": "обученная", "seed": seed, "accuracy": acc,
                             "rows": rows, "seconds": time.perf_counter() - t0})
        save_fn(all_results)

    for seed in range(n_untrain_seeds):
        t0 = time.perf_counter()
        torch.manual_seed(2000 + seed)
        model = TapNet(c=c, use_bn=use_bn, use_residual=use_residual, dropout_p=dropout_p)
        x_check = data.prep(data.xtr[:4])
        sanity_check(model, x_check)
        calibrate_batchnorm(model, data, seed=SEED + seed)
        acc = eval_accuracy(model, data)
        print(f"  [необученная seed={seed}] acc={acc:.3f}")
        rows = analyze_net(model, data, f"необученная seed={seed}", False, SEED, max_pairs=max_pairs, n_cal=n_cal)
        all_results.append({"mechanism": name, "c": c, "use_bn": use_bn, "use_residual": use_residual,
                             "dropout_p": dropout_p, "net_type": "необученная", "seed": seed, "accuracy": acc,
                             "rows": rows, "seconds": time.perf_counter() - t0})
        save_fn(all_results)


# --------------------------------------------------------------------------- main ---
def main():
    torch.manual_seed(SEED)
    data = Data()
    print(f"MNIST loaded: train={len(data.train_idx)}, sel={len(data.sel_idx)}")

    args = sys.argv[1:]
    if "--mechanisms-only" in args:
        phases = {"mechanisms"}
    elif "--width-only" in args:
        phases = {"width"}
    elif "--dropout-only" in args:
        phases = {"dropout"}
    elif "--skip-mechanisms" in args:
        phases = {"width", "dropout"}
    else:
        phases = {"mechanisms", "width", "dropout"}

    if "mechanisms" in phases:
        print(f"C={C}, DEPTH={DEPTH}, N_CAL={N_CAL}, EPOCHS={EPOCHS}, "
              f"seeds trained/untrained={N_TRAIN_SEEDS}/{N_UNTRAIN_SEEDS}")
        print("\n########## ЧАСТЬ 1: механизм (BatchNorm/residual) ##########")
        run_mechanisms(data)

    if "width" in phases:
        followup_results = []
        print("\n########## ЧАСТЬ 2: ширина ##########")
        for name, c, use_bn, use_residual, n_tr, n_un, n_cal, max_pairs in WIDTH_CONFIGS:
            run_followup_one(name, c, use_bn, use_residual, n_tr, n_un, n_cal, max_pairs,
                              data, followup_results, save_fn=save_followup_checkpoint)

    if "dropout" in phases:
        print("\n########## ЧАСТЬ 3: dropout ##########")
        run_dropout(data)

    print("\nВсё готово.", MECHANISM_RESULTS_PATH, FOLLOWUP_RESULTS_PATH, DROPOUT_RESULTS_PATH)
    print("Дальше: python plot_mechanism_depth_profile.py && python plot_width_profile.py "
          "&& python plot_dropout_profile.py")


if __name__ == "__main__":
    main()
