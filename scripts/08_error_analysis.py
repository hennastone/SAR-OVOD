"""
Hata analizi: yanlis pozitif / yanlis negatif toplama, kirpinti kaydi, kalibrasyon.

Girdi : outputs/predictions/<tag>.json, data/annotations/instances_val.json, data/images/val/
Cikti : outputs/error_analysis/<tag>/
          manifest.csv                 tum hatalar (kirpinti kaydedilmese de)
          fp_summary.csv               FP sayilari: tip x bant x sinif
          fn_summary.csv               FN sayilari: bant x sinif
          calibration.csv              guven bini basina istatistik + ECE
          figures/fp_type_breakdown.png
          figures/fp_conf_size.png
          figures/reliability_diagram.png
          crops/<hata_tipi>/<sinif>/<bant>/*.png

FP alt tipleri (TIDE tarzi) - "baskin hata modu ne" sorusunu ancak bu ayrim cevaplar:
  background       hicbir GT ile anlamli ortusme yok  (kopuk, yansima, kaya...)
  class_confusion  BASKA sinifta bir GT'yi dogru bulmus, sinifi yanlis
  localization     dogru sinifta GT var ama IoU esigin altinda (kutu kotu)
  duplicate        dogru sinifta GT'yi bulmus ama o GT zaten daha yuksek skorlu
                   bir tespitle eslesmis (mukerrer)

Ornek:
  python scripts/08_error_analysis.py --tag baseline_a_pilot --conf 0.25
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

BANDS = ["<16", "16-32", "32-64", ">64"]
BAND_EDGES = [0, 16, 32, 64, np.inf]
# '<' ve '>' Windows'ta dizin adinda kullanilamaz; kabukta da kacis gerektirir
BAND_DIR = {"<16": "lt16", "16-32": "16-32", "32-64": "32-64", ">64": "gt64"}
FP_TYPES = ["background", "class_confusion", "localization", "duplicate"]

# dataviz referans paleti - kategorik slot 1..4 (acik tema)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_MUTED = "#0b0b0b", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def band_of(edge: float) -> str:
    return BANDS[int(np.searchsorted(BAND_EDGES, edge, side="right")) - 1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xywh, b: (M,4) xywh -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1 = a[:, 0, None], a[:, 1, None]
    ax2, ay2 = ax1 + a[:, 2, None], ay1 + a[:, 3, None]
    bx1, by1 = b[None, :, 0], b[None, :, 1]
    bx2, by2 = bx1 + b[None, :, 2], by1 + b[None, :, 3]

    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2, None] * a[:, 3, None] + b[None, :, 2] * b[None, :, 3] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_image(dets: list[dict], gts: list[dict], iou_thr: float, loc_min: float):
    """Tek goruntu icin eslestirme. (tp_records, fp_records, fn_records) doner."""
    dets = sorted(dets, key=lambda d: -d["score"])
    d_box = np.array([d["bbox"] for d in dets], dtype=np.float32).reshape(-1, 4)
    g_box = np.array([g["bbox"] for g in gts], dtype=np.float32).reshape(-1, 4)
    d_cat = np.array([d["category_id"] for d in dets], dtype=np.int64)
    g_cat = np.array([g["category_id"] for g in gts], dtype=np.int64)

    ious = iou_matrix(d_box, g_box)
    gt_taken = np.zeros(len(gts), dtype=bool)

    tp, fp = [], []
    for i, det in enumerate(dets):
        same = np.where(g_cat == d_cat[i])[0]
        # 1) ayni sinifta, henuz eslesmemis en iyi GT
        best_j, best_iou = -1, 0.0
        for j in same:
            if not gt_taken[j] and ious[i, j] > best_iou:
                best_j, best_iou = j, float(ious[i, j])

        if best_j >= 0 and best_iou >= iou_thr:
            gt_taken[best_j] = True
            tp.append({"det_idx": i, "gt_idx": best_j, "iou": best_iou})
            continue

        # 2) FP - alt tipini belirle
        same_best_iou = float(ious[i, same].max()) if len(same) else 0.0
        other = np.where(g_cat != d_cat[i])[0]
        other_best_iou = float(ious[i, other].max()) if len(other) else 0.0
        other_best_j = int(other[ious[i, other].argmax()]) if len(other) else -1

        if same_best_iou >= iou_thr:
            fp_type, conf_with = "duplicate", int(d_cat[i])
        elif other_best_iou >= iou_thr:
            fp_type, conf_with = "class_confusion", int(g_cat[other_best_j])
        elif same_best_iou >= loc_min:
            fp_type, conf_with = "localization", int(d_cat[i])
        else:
            fp_type, conf_with = "background", -1

        fp.append({"det_idx": i, "fp_type": fp_type, "confused_with": conf_with,
                   "best_iou": max(same_best_iou, other_best_iou)})

    fn = [{"gt_idx": j} for j in range(len(gts)) if not gt_taken[j]]
    return dets, tp, fp, fn


def crop_and_save(img: Image.Image, bbox, out_path: Path, context: float, min_px: int):
    """Baglam icin genisletilmis kirpinti. Cok kucuk kirpintilar okunabilir olsun
    diye buyutulur (LANCZOS) - bu gorsel inceleme icindir, olcum icin degil."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    half_w, half_h = max(w * context / 2, 16), max(h * context / 2, 16)
    x1 = int(max(0, cx - half_w))
    y1 = int(max(0, cy - half_h))
    x2 = int(min(img.width, cx + half_w))
    y2 = int(min(img.height, cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img.crop((x1, y1, x2, y2))
    if min(crop.size) < min_px:
        scale = min_px / min(crop.size)
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True


# ---------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED)


def plot_fp_breakdown(fp_df: pd.DataFrame, out: Path, tag: str):
    counts = (fp_df.groupby(["size_band", "fp_type"]).size()
              .unstack("fp_type").reindex(BANDS).fillna(0))
    for t in FP_TYPES:
        if t not in counts:
            counts[t] = 0
    counts = counts[FP_TYPES]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    _style(ax)
    bottom = np.zeros(len(BANDS))
    for i, t in enumerate(FP_TYPES):
        vals = counts[t].values
        ax.bar(BANDS, vals, bottom=bottom, color=SERIES[i], label=t,
               zorder=3, edgecolor=SURFACE, linewidth=2)
        bottom += vals
    ax.set_xlabel("nesne boyut bandi (sqrt(w*h), piksel)", color=INK_MUTED)
    ax.set_ylabel("yanlis pozitif sayisi", color=INK_MUTED)
    ax.set_title(f"Yanlis pozitiflerin tipi ve boyutu - {tag}", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_fp_conf_size(fp_df: pd.DataFrame, tp_df: pd.DataFrame, out: Path, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    _style(axes[0])
    bins = np.linspace(0, 1, 21)
    # Ust uste binen dolu histogramlar birbirini kapatip sayilari yanlis
    # okutuyor; ana hat + hafif dolgu ikisini de gorunur birakir.
    for vals, color, label in ((tp_df["score"], SERIES[2], "dogru pozitif"),
                               (fp_df["score"], SERIES[1], "yanlis pozitif")):
        axes[0].hist(vals, bins=bins, color=color, alpha=0.18, zorder=3)
        axes[0].hist(vals, bins=bins, histtype="step", linewidth=2,
                     color=color, label=label, zorder=4)
    axes[0].set_xlabel("guven skoru", color=INK_MUTED)
    axes[0].set_ylabel("tespit sayisi", color=INK_MUTED)
    axes[0].set_title("Guven skoru dagilimi", color=INK, fontsize=12)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9)

    _style(axes[1])
    for i, t in enumerate(FP_TYPES):
        sub = fp_df[fp_df.fp_type == t]
        if len(sub) == 0:
            continue
        axes[1].scatter(sub["edge_len"], sub["score"], s=8, alpha=0.35,
                        color=SERIES[i], label=t, zorder=3, edgecolors="none")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("tespit kenar uzunlugu sqrt(w*h) (piksel, log)", color=INK_MUTED)
    axes[1].set_ylabel("guven skoru", color=INK_MUTED)
    axes[1].set_title("Yanlis pozitif: boyut vs guven", color=INK, fontsize=12)
    axes[1].legend(frameon=False, labelcolor=INK, fontsize=9, markerscale=2)

    fig.suptitle(f"Yanlis pozitif profili - {tag}", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(calib: pd.DataFrame, ece: float, out: Path, tag: str):
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, color=AXIS,
            label="mukemmel kalibrasyon", zorder=2)
    m = calib["n"] > 0
    ax.plot(calib.loc[m, "mean_conf"], calib.loc[m, "accuracy"], marker="o",
            markersize=8, linewidth=2, color=SERIES[0], label="model",
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ortalama guven skoru", color=INK_MUTED)
    ax.set_ylabel("gercek dogruluk (TP orani)", color=INK_MUTED)
    ax.set_title(f"Guvenilirlik diyagrami - {tag}\nECE = {ece:.4f}", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--gt", type=Path, default=Path("data/annotations/instances_val.json"))
    p.add_argument("--img-dir", type=Path, default=Path("data/images/val"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/error_analysis"))
    p.add_argument("--conf", type=float, default=0.25, help="calisma esigi (hata sayimi icin)")
    p.add_argument("--iou", type=float, default=0.5, help="TP kabul IoU esigi")
    p.add_argument("--loc-min", type=float, default=0.1,
                   help="bu IoU'nun altindaki ayni-sinif ortusmeleri 'background' sayilir")
    p.add_argument("--calib-min-conf", type=float, default=0.05,
                   help="kalibrasyona dahil edilecek minimum guven")
    p.add_argument("--calib-bins", type=int, default=10)
    p.add_argument("--context", type=float, default=3.0, help="kirpinti baglam carpani")
    p.add_argument("--crop-min-px", type=int, default=128, help="kirpintilar bu boyuta buyutulur")
    p.add_argument("--max-crops", type=int, default=60,
                   help="klasor basina en fazla kirpinti (0 = kirpinti kaydetme)")
    p.add_argument("--no-crops", action="store_true")
    args = p.parse_args()

    out_root = args.out_dir / args.tag
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    with open(args.gt, encoding="utf-8") as f:
        gt_data = json.load(f)
    cat_name = {c["id"]: c["name"] for c in gt_data["categories"]}
    img_info = {im["id"]: im for im in gt_data["images"]}
    gt_by_img = defaultdict(list)
    for a in gt_data["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    with open(args.pred_dir / f"{args.tag}.json", encoding="utf-8") as f:
        preds = json.load(f)
    pred_by_img = defaultdict(list)
    for d in preds:
        pred_by_img[d["image_id"]].append(d)

    fp_rows, fn_rows, tp_rows, calib_rows = [], [], [], []

    for image_id in tqdm(sorted(img_info), desc=f"eslestirme ({args.tag})"):
        gts = gt_by_img[image_id]
        all_dets = pred_by_img.get(image_id, [])

        # --- calisma esigi ile hata sayimi
        dets = [d for d in all_dets if d["score"] >= args.conf]
        dets, tp, fp, fn = match_image(dets, gts, args.iou, args.loc_min)

        for r in tp:
            d = dets[r["det_idx"]]
            edge = float(np.sqrt(d["bbox"][2] * d["bbox"][3]))
            tp_rows.append({"image_id": image_id, "category_id": d["category_id"],
                            "category": cat_name[d["category_id"]], "score": d["score"],
                            "edge_len": edge, "size_band": band_of(edge), "iou": r["iou"]})
        for r in fp:
            d = dets[r["det_idx"]]
            edge = float(np.sqrt(d["bbox"][2] * d["bbox"][3]))
            fp_rows.append({"image_id": image_id, "file_name": img_info[image_id]["file_name"],
                            "category_id": d["category_id"], "category": cat_name[d["category_id"]],
                            "score": d["score"], "bbox": d["bbox"], "edge_len": edge,
                            "size_band": band_of(edge), "fp_type": r["fp_type"],
                            "best_iou": r["best_iou"],
                            "confused_with": cat_name.get(r["confused_with"], "")})
        for r in fn:
            g = gts[r["gt_idx"]]
            edge = float(np.sqrt(g["bbox"][2] * g["bbox"][3]))
            fn_rows.append({"image_id": image_id, "file_name": img_info[image_id]["file_name"],
                            "category_id": g["category_id"], "category": cat_name[g["category_id"]],
                            "bbox": g["bbox"], "edge_len": edge, "size_band": band_of(edge)})

        # --- kalibrasyon: dusuk esikle ayri eslestirme (bin'ler dolsun)
        cdets = [d for d in all_dets if d["score"] >= args.calib_min_conf]
        cdets, ctp, cfp, _ = match_image(cdets, gts, args.iou, args.loc_min)
        for r in ctp:
            calib_rows.append({"score": cdets[r["det_idx"]]["score"], "correct": 1})
        for r in cfp:
            calib_rows.append({"score": cdets[r["det_idx"]]["score"], "correct": 0})

    fp_df = pd.DataFrame(fp_rows)
    fn_df = pd.DataFrame(fn_rows)
    tp_df = pd.DataFrame(tp_rows)
    calib_df = pd.DataFrame(calib_rows)

    # ---------------- tablolar
    manifest = pd.concat([
        fp_df.assign(error_kind="FP") if len(fp_df) else pd.DataFrame(),
        fn_df.assign(error_kind="FN", score=np.nan, fp_type="", best_iou=np.nan,
                     confused_with="") if len(fn_df) else pd.DataFrame(),
    ], ignore_index=True)
    manifest.to_csv(out_root / "manifest.csv", index=False)

    if len(fp_df):
        (fp_df.groupby(["fp_type", "size_band", "category"]).size()
         .rename("count").reset_index()
         .to_csv(out_root / "fp_summary.csv", index=False))
    if len(fn_df):
        (fn_df.groupby(["size_band", "category"]).size()
         .rename("count").reset_index()
         .to_csv(out_root / "fn_summary.csv", index=False))

    # ---------------- kalibrasyon + ECE
    edges = np.linspace(0, 1, args.calib_bins + 1)
    calib_df["bin"] = np.clip(np.digitize(calib_df["score"], edges) - 1, 0, args.calib_bins - 1)
    grp = calib_df.groupby("bin")
    calib = pd.DataFrame({
        "bin_low": edges[:-1], "bin_high": edges[1:],
        "n": grp.size().reindex(range(args.calib_bins), fill_value=0),
        "mean_conf": grp["score"].mean().reindex(range(args.calib_bins)),
        "accuracy": grp["correct"].mean().reindex(range(args.calib_bins)),
    })
    n_tot = calib["n"].sum()
    ece = float((calib["n"] / n_tot * (calib["accuracy"] - calib["mean_conf"]).abs()).sum()) if n_tot else float("nan")
    calib.to_csv(out_root / "calibration.csv", index=False)

    # ---------------- grafikler
    figs = out_root / "figures"
    if len(fp_df):
        plot_fp_breakdown(fp_df, figs / "fp_type_breakdown.png", args.tag)
        plot_fp_conf_size(fp_df, tp_df, figs / "fp_conf_size.png", args.tag)
    plot_reliability(calib, ece, figs / "reliability_diagram.png", args.tag)

    # ---------------- kirpintilar
    if not args.no_crops and args.max_crops > 0:
        buckets = defaultdict(list)
        for _, r in fp_df.iterrows():
            buckets[("fp_" + r["fp_type"], r["category"], r["size_band"])].append(r)
        for _, r in fn_df.iterrows():
            buckets[("fn", r["category"], r["size_band"])].append(r)

        # bucket ici siralama: FP'de yuksek guven once (en zararlilari),
        # FN'de buyuk nesne once (kacirilmasi en sasirtici olanlar)
        by_image = defaultdict(list)
        for (kind, cat, band), rows in buckets.items():
            rows = sorted(rows, key=lambda r: -r["score"] if kind != "fn" else -r["edge_len"])
            for i, r in enumerate(rows[: args.max_crops]):
                conf_s = f"conf{r['score']:.2f}_" if kind != "fn" else ""
                fname = f"{i:03d}_{conf_s}size{r['edge_len']:.0f}px_img{r['image_id']}.png"
                by_image[r["file_name"]].append(
                    (out_root / "crops" / kind / cat / BAND_DIR[band] / fname, r["bbox"]))

        for file_name, items in tqdm(by_image.items(), desc="kirpinti"):
            with Image.open(args.img_dir / file_name) as im:
                im = im.convert("RGB")
                for path, bbox in items:
                    crop_and_save(im, bbox, path, args.context, args.crop_min_px)

    # ---------------- ozet
    lines = [f"=== {args.tag} | conf>={args.conf}, IoU>={args.iou} ===", ""]
    lines.append(f"TP: {len(tp_df)}   FP: {len(fp_df)}   FN: {len(fn_df)}")
    if len(tp_df) + len(fp_df):
        lines.append(f"precision: {len(tp_df)/(len(tp_df)+len(fp_df)):.4f}   "
                     f"recall: {len(tp_df)/(len(tp_df)+len(fn_df)):.4f}")
    lines.append("")
    if len(fp_df):
        lines.append("Yanlis pozitif tipi (baskin hata modu):")
        share = fp_df["fp_type"].value_counts(normalize=True)
        for t in FP_TYPES:
            lines.append(f"  {t:16s} {int(fp_df['fp_type'].eq(t).sum()):6d}  "
                         f"({share.get(t, 0)*100:5.1f}%)")
        lines.append("")
        lines.append("FP tipi x boyut bandi:")
        lines.append(pd.crosstab(fp_df["size_band"], fp_df["fp_type"])
                     .reindex(BANDS).fillna(0).astype(int).to_string())
        lines.append("")
        cc = fp_df[fp_df.fp_type == "class_confusion"]
        if len(cc):
            lines.append("Sinif karisikligi (tahmin -> gercek):")
            lines.append(pd.crosstab(cc["category"], cc["confused_with"]).to_string())
            lines.append("")
    if len(fn_df):
        lines.append("Yanlis negatif, boyut bandina gore:")
        lines.append(fn_df["size_band"].value_counts().reindex(BANDS).fillna(0)
                     .astype(int).to_string())
        lines.append("")
    lines.append(f"ECE (guven kalibrasyon hatasi): {ece:.4f}")

    text = "\n".join(lines)
    (out_root / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nCiktilar: {out_root}/")


if __name__ == "__main__":
    main()
