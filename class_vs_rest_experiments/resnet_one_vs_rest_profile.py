"""
Направление 1 из плана недели: "один класс против остальных" на ResNet-18.

САМОДОСТАТОЧНЫЙ СКРИПT - не зависит от других файлов проекта (synergy_experiments/),
все нужные причинный-тест функции инлайнены ниже. Нужен только локальный датасет
(DATA_ROOT - см. ниже, подмножество ImageNet: папка на класс, изображения внутри)
и стандартные библиотеки (torch, torchvision, numpy, scipy, statsmodels).

ИСПРАВЛЕНИЕ (после разбора методологии): первая версия этого скрипта считала
margin_k(x) = logit_k(x) - max_{j!=k} logit_j(x) и использовала ЭТУ величину
одновременно и для разметки роли канала, и для причинного S-теста. Это
методологическая ошибка: max() по 999 остальным классам - нелинейная, негладкая
операция, которая подмешивает в попарный тест нелинейность, не имеющую отношения
к самой сети (абляция пары каналов может незаметно поменять, какой класс "второй
по силе", и это создаёт скачок в margin сам по себе). Измерять S нужно на СЫРОМ
логите - только так гарантированно ловятся нелинейности СЕТИ (ReLU, BatchNorm,
residual), а не нелинейности способа измерения.

Исправленная методика:

1. РАЗМЕТКА РОЛИ (critical/harmful/neutral) - эпсилон ξ из C-SWAP (Bauvin, Baret,
   Ahmad, BMVC 2025): относительное изменение softmax-вероятности истинного
   класса, ξ = mean[(sigma*-sigma)/sigma]. Это одноканальный тест значимости (не
   попарный!), поэтому нелинейность softmax здесь не создаёт ложного
   "взаимодействия" между парой каналов - только искажает размер эффекта ОДНОГО
   канала. Без голосования по классам (класс один) - в отличие от стандартной
   C-SWAP-разметки, где роль - голосование (OR) значимости по всем классам.

2. ПРИЧИННЫЙ S-ТЕСТ (синергия/субаддитивность) - сырой логит истинного класса,
   TE = baseline_logit - ablated_logit, S(A,B) = TE(A)+TE(B)-TE(A,B). Никакого
   margin и никакой вероятности здесь нет.

Абляция - do(zero channel): вся карта признаков канала в точке "a" (вход в
conv2 блока, через forward-pre-hook) зануляется, логиты пересчитываются с этой
точки до конца сети. 8 точек по глубине (оба блока каждой из 4 стадий ResNet-18),
равномерно случайные пары каналов (до N_PAIRS на точку), FDR-BY + порог эффекта
tau, обученная (ImageNet-веса) vs необученная (случайная инициализация +
калибровка BatchNorm) сеть.
"""
import itertools
import json
import os
import time

import numpy as np
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from torchvision import datasets, models, transforms

# --- конфигурация: поменяйте под свою машину ---
DATA_ROOT = r"D:\cnn_mi_experiments\ate_analysis\data\imagenet_paper\val"   # папка-на-класс с картинками
OUT_DIR = r"D:\cnn_mi_experiments\class_vs_rest_experiments"

SEED = 42
ALPHA = 0.05
FDR_METHOD = "fdr_by"      # Benjamini-Yekutieli: парные тесты структурно зависимы
TAU_REL = 0.02             # порог размера эффекта, в долях std baseline
N_CAL = int(os.environ.get("OVR_N_CAL", "128"))     # калибровочных изображений на класс
RUN_CHUNK = 32             # вариантов абляции в одном батче "хвоста" сети
PROB_FLOOR = 1e-3          # защита эпсилон от деления на почти-ноль (редкие неуверенные изображения)

TARGET_CLASS = int(os.environ.get("OVR_CLASS", "0"))    # один класс, любой
N_PAIRS = int(os.environ.get("OVR_N_PAIRS", "800"))      # случайных пар на точку

MEAN_IMAGENET = (.485, .456, .406)
STD_IMAGENET = (.229, .224, .225)
TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
])

# 8 точек по глубине - ОБА блока каждой из 4 стадий ResNet-18 (не только block_idx=1,
# как в типичных профилях этого семейства экспериментов).
LAYERS_EXT = [
    (0, 0, 64, "layer1.0.conv2"), (0, 1, 64, "layer1.1.conv2"),
    (1, 0, 128, "layer2.0.conv2"), (1, 1, 128, "layer2.1.conv2"),
    (2, 0, 256, "layer3.0.conv2"), (2, 1, 256, "layer3.1.conv2"),
    (3, 0, 512, "layer4.0.conv2"), (3, 1, 512, "layer4.1.conv2"),
]


# ============================== причинный тест: инфраструктура ==============================

def get_stages(model):
    return [model.layer1, model.layer2, model.layer3, model.layer4]


def capture_block_io(model, stage_idx, block_idx, x_batch):
    """Прогоняет x_batch один раз, захватывая вход блока (для identity) и вход
    conv2 ('a' - точка абляции). Возвращает (a, identity, baseline_logits)."""
    stages = get_stages(model)
    block = stages[stage_idx][block_idx]
    captured = {}

    def pre_block(module, inp):
        captured["x"] = inp[0].detach()

    def pre_conv2(module, inp):
        captured["a"] = inp[0].detach()

    h1 = block.register_forward_pre_hook(pre_block)
    h2 = block.conv2.register_forward_pre_hook(pre_conv2)
    with torch.no_grad():
        logits = model(x_batch)
    h1.remove()
    h2.remove()
    x_in = captured["x"]
    identity = block.downsample(x_in) if block.downsample is not None else x_in
    return captured["a"], identity, logits


def run_from(model, stage_idx, block_idx, a, identity):
    """Доcчитывает сеть от входа conv2 указанного блока до логитов (эквивалентно
    полному forward с заданными (a, identity) в этой точке - проверяется sanity_check)."""
    stages = get_stages(model)
    block = stages[stage_idx][block_idx]
    out = block.bn2(block.conv2(a))
    out = out + identity
    x = block.relu(out)
    for b in stages[stage_idx][block_idx + 1:]:
        idn = b.downsample(x) if b.downsample is not None else x
        out = b.relu(b.bn1(b.conv1(x)))
        out = b.bn2(b.conv2(out))
        x = b.relu(out + idn)
    for s in stages[stage_idx + 1:]:
        for b in s:
            idn = b.downsample(x) if b.downsample is not None else x
            out = b.relu(b.bn1(b.conv1(x)))
            out = b.bn2(b.conv2(out))
            x = b.relu(out + idn)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return model.fc(x)


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def batched_variant_full_logits(model, stage_idx, block_idx, a, identity, variants, chunk=RUN_CHUNK):
    """variants: список кортежей номеров каналов для зануления. Возвращает ПОЛНЫЙ
    вектор логитов (все 1000 классов) для каждого варианта x каждого изображения."""
    B = a.shape[0]
    out_rows = []
    with torch.no_grad():
        for start in range(0, len(variants), chunk):
            batch_variants = variants[start:start + chunk]
            k = len(batch_variants)
            a_rep = a.unsqueeze(0).repeat(k, 1, 1, 1, 1).clone()
            for vi, channels in enumerate(batch_variants):
                for c in channels:
                    a_rep[vi, :, c, :, :] = 0.0
            id_rep = identity.unsqueeze(0).repeat(k, 1, 1, 1, 1)
            a_flat = a_rep.view(k * B, *a.shape[1:])
            id_flat = id_rep.reshape(k * B, *identity.shape[1:])
            logits = run_from(model, stage_idx, block_idx, a_flat, id_flat)
            out_rows.append(logits.view(k, B, -1).cpu().numpy())
    return np.concatenate(out_rows, axis=0)


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


def calibrate_batchnorm(model, dataset, n_images=256, batch_size=32, seed=SEED):
    """У свежей (необученной) сети running_mean=0/var=1 - заводские заглушки, не
    оценка реальных статистик. Прогоняем в .train() на обычных изображениях (без
    вмешательств), чтобы BatchNorm получил реальные статистики этой конкретной
    случайной инициализации, затем .eval()."""
    model.train()
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n_images, len(dataset)), replace=False)
    with torch.no_grad():
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            xb = torch.stack([dataset[i][0] for i in batch_idx])
            model(xb)
    model.eval()


def calib_batch(dataset, class_idx, n, seed):
    idxs = [i for i, (_, y) in enumerate(dataset.samples) if y == class_idx]
    rng = np.random.default_rng(seed + class_idx)
    pick = rng.choice(idxs, size=min(n, len(idxs)), replace=False)
    xs = torch.stack([dataset[i][0] for i in pick])
    return xs


def sample_all_pairs(n_channels, n, rng):
    pool = list(itertools.combinations(range(n_channels), 2))
    total = len(pool)
    if total <= n:
        return pool, total
    idx = rng.choice(total, size=n, replace=False)
    return [pool[i] for i in idx], total


def sanity_check_ext(model, dataset):
    x = calib_batch(dataset, 0, 4, SEED)
    for stage_idx, block_idx, _, name in LAYERS_EXT:
        a, identity, baseline_logits = capture_block_io(model, stage_idx, block_idx, x)
        recon = run_from(model, stage_idx, block_idx, a, identity)
        diff = (recon - baseline_logits).abs().max().item()
        assert diff < 1e-4, f"{name}: recon diff {diff} too large"
        print(f"  sanity ok: {name}, max|diff|={diff:.2e}")


# ============================== основной анализ (роль на эпсилон + S-тест на ATE) ==============================

def analyze_layer_one_class(model, dataset, stage_idx, block_idx, n_channels, class_idx, n_pairs, seed):
    x = calib_batch(dataset, class_idx, N_CAL, SEED)
    a, identity, baseline_full = capture_block_io(model, stage_idx, block_idx, x)
    baseline_full_np = baseline_full.cpu().numpy()

    baseline_logit = baseline_full_np[:, class_idx]                    # для S-теста (ATE)
    baseline_prob = softmax_np(baseline_full_np)[:, class_idx]         # для роли (эпсилон)

    keep = baseline_prob > PROB_FLOOR
    if not keep.all():
        a, identity = a[keep], identity[keep]
        baseline_logit, baseline_prob = baseline_logit[keep], baseline_prob[keep]

    tau_logit = TAU_REL * float(np.std(baseline_logit, ddof=1))
    tau_prob = TAU_REL * float(np.std(baseline_prob, ddof=1))

    single_variants = [(c,) for c in range(n_channels)]
    full_single = batched_variant_full_logits(model, stage_idx, block_idx, a, identity, single_variants)
    logit_single = full_single[:, :, class_idx]
    prob_single = softmax_np(full_single)[:, :, class_idx]

    te_single = baseline_logit[None, :] - logit_single
    rel_single = (prob_single - baseline_prob[None, :]) / baseline_prob[None, :]
    te_cache = {c: te_single[c] for c in range(n_channels)}

    # --- разметка роли на эпсилон, без голосования (класс один) ---
    singles = []
    for c in range(n_channels):
        r = mean_test(rel_single[c], tau_prob)
        r["channel"] = c
        singles.append(r)
    fdr(singles, ALPHA, FDR_METHOD)
    roles = []
    for r in singles:
        keep_sig = r["significant"] and abs(r["ate"]) > tau_prob
        role = "critical" if keep_sig and r["ate"] <= 0 else "harmful" if keep_sig and r["ate"] > 0 else "neutral"
        roles.append(role)
    n_critical = sum(r == "critical" for r in roles)
    n_harmful = sum(r == "harmful" for r in roles)
    n_neutral = sum(r == "neutral" for r in roles)

    # --- причинный S-тест на сыром логите, равномерно случайные пары ---
    rng = np.random.default_rng(seed + class_idx)
    pairs, total_possible = sample_all_pairs(n_channels, n_pairs, rng)
    rows = []
    if pairs:
        logits_full_pairs = batched_variant_full_logits(model, stage_idx, block_idx, a, identity, pairs)
        logit_pairs = logits_full_pairs[:, :, class_idx]
        for k, (i, j) in enumerate(pairs):
            te_joint = baseline_logit[None, :] - logit_pairs[k:k + 1]
            s_vals = te_cache[i] + te_cache[j] - te_joint[0]
            r = mean_test(s_vals, tau_logit)
            r.update(a=i, b=j, S=r["ate"])
            rows.append(r)
        fdr(rows, ALPHA, FDR_METHOD)
        for r in rows:
            ok = r["significant"] and abs(r["S"]) > tau_logit
            r["sign"] = "synergy" if ok and r["S"] > 0 else "subadditive" if ok and r["S"] < 0 else "additive"

    n_syn = sum(r["sign"] == "synergy" for r in rows)
    n_sub = sum(r["sign"] == "subadditive" for r in rows)
    return {
        "n_channels": n_channels, "n_images_kept": int(keep.sum()),
        "n_critical": n_critical, "n_harmful": n_harmful, "n_neutral": n_neutral,
        "n_pairs_tested": len(rows), "total_possible_pairs": total_possible,
        "coverage_pct": 100.0 * len(rows) / total_possible if total_possible else 0.0,
        "n_syn": n_syn, "n_sub": n_sub,
        "nonadditive_pairs": [(r["a"], r["b"], round(r["S"], 5)) for r in rows if r["sign"] != "additive"],
    }


RESULTS_PATH = OUT_DIR + r"\resnet_one_vs_rest_profile.json"


def save_checkpoint(rows):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": {"N_CAL": N_CAL, "TARGET_CLASS": TARGET_CLASS,
                               "N_PAIRS": N_PAIRS, "ALPHA": ALPHA, "TAU_REL": TAU_REL,
                               "FDR_METHOD": FDR_METHOD, "PROB_FLOOR": PROB_FLOOR,
                               "scoring": "роль - эпсилон C-SWAP (без голосования); "
                                          "S-тест - сырой логит истинного класса (ATE)"},
                   "rows": rows}, f, indent=2, ensure_ascii=False)


def run_for_model(model, dataset, label, rows):
    for stage_idx, block_idx, n_channels, name in LAYERS_EXT:
        t0 = time.perf_counter()
        res = analyze_layer_one_class(model, dataset, stage_idx, block_idx, n_channels, TARGET_CLASS, N_PAIRS, SEED)
        row = {"net_type": label, "layer": name, "class_idx": TARGET_CLASS, **res,
               "seconds": time.perf_counter() - t0}
        print(f"[{label}] {name} (класс {TARGET_CLASS}): "
              f"critical={res['n_critical']}/{n_channels} ({100*res['n_critical']/n_channels:.0f}%), "
              f"harmful={res['n_harmful']}/{n_channels} ({100*res['n_harmful']/n_channels:.0f}%), "
              f"neutral={res['n_neutral']}/{n_channels} ({100*res['n_neutral']/n_channels:.0f}%)")
        print(f"    пары (ATE): syn={res['n_syn']} sub={res['n_sub']} /{res['n_pairs_tested']} tested "
              f"(coverage={res['coverage_pct']:.2f}% из {res['total_possible_pairs']}) "
              f"({row['seconds']:.1f} c)")
        if res["nonadditive_pairs"]:
            print(f"    неаддитивные пары (канал_a, канал_b, S): {res['nonadditive_pairs'][:20]}"
                  + (" ..." if len(res["nonadditive_pairs"]) > 20 else ""))
        rows.append(row)
        save_checkpoint(rows)
    return rows


def main():
    torch.manual_seed(SEED)
    dataset = datasets.ImageFolder(DATA_ROOT, transform=TRANSFORM)
    print(f"dataset: {len(dataset)} images, классы из {sorted(dataset.classes)}")
    print(f"TARGET_CLASS={TARGET_CLASS}, N_CAL={N_CAL}, N_PAIRS={N_PAIRS}, точек по глубине={len(LAYERS_EXT)}")

    print("--- sanity check (обученная сеть, 8 точек) ---")
    model_trained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    sanity_check_ext(model_trained, dataset)

    rows = []
    print("\n--- обученная ResNet-18: роль на эпсилон, S-тест на ATE, один класс, 8 точек по глубине ---")
    run_for_model(model_trained, dataset, "обученная", rows)

    print("\n--- необученная (случайная инициализация) ResNet-18 ---")
    model_untrained = models.resnet18(weights=None)
    calibrate_batchnorm(model_untrained, dataset)
    model_untrained.eval()
    sanity_check_ext(model_untrained, dataset)
    run_for_model(model_untrained, dataset, "необученная", rows)

    print("\nГотово.", RESULTS_PATH)


if __name__ == "__main__":
    main()
