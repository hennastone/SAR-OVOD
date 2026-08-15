"""
Madde 6 - TP/FP guven skoru ayrismasi.

calibration.csv kova bazli "fraction_correct" veriyor; o kalibrasyonu olcer,
AYRISMAYI olcmez. Buradaki soru farkli: guven skoru dogru ve yanlis tespitleri
ne kadar iyi ayiriyor? Bir esik secilebilir mi?

Olculenler
  auroc          Guven skorunu TP/FP ayirici olarak kullanmanin ROC alani.
                 0.5 = bilgisiz, 1.0 = mukemmel ayrisma. Mann-Whitney U ile
                 hesaplanir (esitlikler ortalama rank alir).
  ks             Kolmogorov-Smirnov: iki dagilimin CDF'leri arasindaki en
                 buyuk fark. Nerede ayristiklarini da verir (ks_at_conf).
  overlap        Histogram kesisimi (0-1). 1 = dagilimlar tamamen ic ice.
  youden_j       max(TPR - FPR) ve bunu veren esik -> "en iyi tek esik".
  mean/median    Ham betimleyiciler.

Girdi : outputs/predictions/<tag>.json, data/annotations/instances_val.json
Cikti : outputs/tables/confidence_separation.csv
        outputs/tables/confidence_separation_by_band.csv
        outputs/comparison/confidence_separation.png

Ornek:
  python scripts/13_confidence_separation.py --tags baseline_a_pilot baseline_b_canonical
"""
import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# 08_error_analysis rakamla basladigi icin normal import edilemiyor;
# dosya yolundan yukleniyor. Eslestirme mantigi TEK yerde kalsin diye
# kopyalanmiyor - 08'deki sentetik testlerle dogrulanmis kod bu.
_spec = importlib.util.spec_from_file_location(
    "ea", Path(__file__).with_name("08_error_analysis.py"))
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

BANDS = ea.BANDS
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED = "#0b0b0b", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U tabanli ROC alani. sklearn'e bagimlilik yok."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # esitliklere ortalama rank ver
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ks_stat(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    grid = np.linspace(0, 1, 501)
    cdf_p = np.searchsorted(np.sort(pos), grid, side="right") / len(pos)
    cdf_n = np.searchsorted(np.sort(neg), grid, side="right") / len(neg)
    d = np.abs(cdf_p - cdf_n)
    k = int(d.argmax())
    return float(d[k]), float(grid[k])


def overlap_coef(pos: np.ndarray, neg: np.ndarray, bins: int = 50) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    hp, _ = np.histogram(pos, bins=edges, density=False)
    hn, _ = np.histogram(neg, bins=edges, density=False)
    return float(np.minimum(hp / hp.sum(), hn / hn.sum()).sum())


def youden(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    grid = np.linspace(0, 1, 501)
    tpr = 1 - np.searchsorted(np.sort(pos), grid, side="right") / len(pos)
    fpr = 1 - np.searchsorted(np.sort(neg), grid, side="right") / len(neg)
    j = tpr - fpr
    k = int(j.argmax())
    return float(j[k]), float(grid[k])


def collect(tag: str, pred_dir: Path, gt_path: Path, iou_thr: float,
            loc_min: float, min_conf: float) -> pd.DataFrame:
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f)
    img_ids = sorted(im["id"] for im in gt_data["images"])
    gt_by_img = defaultdict(list)
    for a in gt_data["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    with open(pred_dir / f"{tag}.json", encoding="utf-8") as f:
        preds = json.load(f)
    pred_by_img = defaultdict(list)
    for d in preds:
        if d["score"] >= min_conf:
            pred_by_img[d["image_id"]].append(d)

    rows = []
    for image_id in tqdm(img_ids, desc=f"eslestirme ({tag})", leave=False):
        dets, tp, fp, _ = ea.match_image(
            pred_by_img.get(image_id, []), gt_by_img[image_id], iou_thr, loc_min)
        for r in tp:
            d = dets[r["det_idx"]]
            edge = float(np.sqrt(d["bbox"][2] * d["bbox"][3]))
            rows.append({"score": d["score"], "is_tp": 1,
                         "size_band": ea.band_of(edge), "fp_type": ""})
        for r in fp:
            d = dets[r["det_idx"]]
            edge = float(np.sqrt(d["bbox"][2] * d["bbox"][3]))
            rows.append({"score": d["score"], "is_tp": 0,
                         "size_band": ea.band_of(edge), "fp_type": r["fp_type"]})
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame) -> dict:
    pos = df.loc[df.is_tp == 1, "score"].to_numpy()
    neg = df.loc[df.is_tp == 0, "score"].to_numpy()
    ks, ks_at = ks_stat(pos, neg)
    j, j_at = youden(pos, neg)
    return {
        "n_tp": len(pos), "n_fp": len(neg),
        "auroc": round(auroc(pos, neg), 4),
        "ks": round(ks, 4), "ks_at_conf": round(ks_at, 3),
        "overlap": round(overlap_coef(pos, neg), 4),
        "youden_j": round(j, 4), "youden_at_conf": round(j_at, 3),
        "tp_mean_conf": round(float(pos.mean()), 4) if len(pos) else np.nan,
        "fp_mean_conf": round(float(neg.mean()), 4) if len(neg) else np.nan,
        "tp_median_conf": round(float(np.median(pos)), 4) if len(pos) else np.nan,
        "fp_median_conf": round(float(np.median(neg)), 4) if len(neg) else np.nan,
    }


def plot(per_tag: dict[str, pd.DataFrame], out: Path) -> None:
    n = len(per_tag)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), facecolor=SURFACE, squeeze=False)
    bins = np.linspace(0, 1, 41)
    for ax, (tag, df) in zip(axes[0], per_tag.items()):
        ax.set_facecolor(SURFACE)
        for mask, color, label in ((df.is_tp == 1, SERIES[2], "dogru pozitif"),
                                   (df.is_tp == 0, SERIES[1], "yanlis pozitif")):
            v = df.loc[mask, "score"]
            if len(v) == 0:
                continue
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.18, zorder=3)
            ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2,
                    color=color, label=label, zorder=4)
        m = metrics(df)
        ax.set_title(f"{tag}\nAUROC {m['auroc']:.3f}  ortusme {m['overlap']:.2f}",
                     color=INK, fontsize=11)
        ax.set_xlabel("guven skoru", color=INK_MUTED)
        ax.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
        ax.tick_params(colors=INK_MUTED)
    axes[0][0].set_ylabel("yogunluk", color=INK_MUTED)
    axes[0][0].legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.suptitle("TP/FP guven skoru ayrismasi", color=INK, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--gt", type=Path, default=Path("data/annotations/instances_val.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tables"))
    p.add_argument("--fig-dir", type=Path, default=Path("outputs/comparison"))
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--loc-min", type=float, default=0.1)
    p.add_argument("--min-conf", type=float, default=0.05,
                   help="bu esigin altindaki tespitler analize alinmaz")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows, band_rows, per_tag = [], [], {}
    for tag in args.tags:
        if not (args.pred_dir / f"{tag}.json").exists():
            print(f"[atlandi] {tag}: tahmin dosyasi yok")
            continue
        df = collect(tag, args.pred_dir, args.gt, args.iou, args.loc_min, args.min_conf)
        per_tag[tag] = df
        overall_rows.append({"tag": tag, "size_band": "all", **metrics(df)})
        for b in BANDS:
            sub = df[df.size_band == b]
            if len(sub):
                band_rows.append({"tag": tag, "size_band": b, **metrics(sub)})
        m = overall_rows[-1]
        print(f"{tag:26s} AUROC {m['auroc']:.4f}  KS {m['ks']:.4f} @{m['ks_at_conf']:.2f}  "
              f"ortusme {m['overlap']:.4f}  en iyi esik {m['youden_at_conf']:.2f}")

    if not overall_rows:
        raise SystemExit("hicbir tag islenemedi")

    pd.DataFrame(overall_rows).to_csv(args.out_dir / "confidence_separation.csv", index=False)
    pd.DataFrame(band_rows).to_csv(args.out_dir / "confidence_separation_by_band.csv", index=False)
    plot(per_tag, args.fig_dir / "confidence_separation.png")
    print(f"\nCiktilar: {args.out_dir}/confidence_separation*.csv, "
          f"{args.fig_dir}/confidence_separation.png")


if __name__ == "__main__":
    main()
