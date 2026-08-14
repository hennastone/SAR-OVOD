"""
Dagilmis ciktilari outputs/tables/ altinda kalici, tek noktadan okunabilir
CSV'lere toplar. YENI DENEY CALISTIRMAZ - her sayi ya mevcut bir cikti
dosyasindan okunur ya da onlardan aritmetikle turetilir.

Girdi (hepsi mevcut):
  outputs/eda/*.csv
  outputs/metrics/<tag>_per_class.csv, <tag>_overall.csv
  outputs/error_analysis/<tag>/{fp_summary,fn_summary,calibration,manifest}.csv
  outputs/predictions/<tag>_meta.json
  data/annotations/instances_{train,val}.json   <- yalnizca sayim icin okunur

Cikti: outputs/tables/*.csv

Turetme notlari:
  - TP = GT - FN  (her GT ya eslesir ya kacar; ozdeslik 08'in TP sayisiyla
    dogrulanir, uyusmazsa hata verir)
  - precision = TP / (TP + FP), recall = TP / (TP + FN)
  - FP'nin bandi/sinifi TESPIT kutusuna, FN'ninki GT kutusuna gore. Bir tespit
    X bandinda olup Y bandindaki bir GT'ye ait olabilir; bant bazli precision
    icin dogal tanim budur ama recall ile tam simetrik degildir.
  - precision/recall conf>=0.25 & IoU>=0.5'te; AP/AR tum guven araliginda.
    Sutun adlari bunu tasiyor.
  - Olculmemis alanlar BOS birakilir, uydurulmaz.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BANDS = ["<16", "16-32", "32-64", ">64"]
CONF_THR, IOU_THR = 0.25, 0.5

MODELS = {
    "baseline_a_pilot": {"model": "YOLO11s fine-tune", "kind": "closed_set", "query_set": ""},
    "baseline_b_canonical": {"model": "YOLO-World v2-s", "kind": "open_vocab", "query_set": "canonical"},
    "baseline_b_attributed": {"model": "YOLO-World v2-s", "kind": "open_vocab", "query_set": "attributed"},
}


# ----------------------------------------------------------------- A1/A2
def dataset_stats(ann_dir: Path, out: Path) -> None:
    rows = []
    for split in ["train", "val"]:
        with open(ann_dir / f"instances_{split}.json", encoding="utf-8") as f:
            d = json.load(f)
        cat = {c["id"]: c["name"] for c in d["categories"]}
        n_images_total = len(d["images"])
        n_inst_total = len(d["annotations"])

        per_cat_inst, per_cat_imgs = {}, {}
        for a in d["annotations"]:
            name = cat[a["category_id"]]
            per_cat_inst[name] = per_cat_inst.get(name, 0) + 1
            per_cat_imgs.setdefault(name, set()).add(a["image_id"])

        for name in sorted(per_cat_inst, key=lambda k: -per_cat_inst[k]):
            n_inst = per_cat_inst[name]
            n_img = len(per_cat_imgs[name])
            rows.append({
                "split": split, "category": name,
                "n_instances": n_inst,
                "n_images_containing": n_img,
                "pct_images_containing": round(100 * n_img / n_images_total, 2),
                "mean_instances_per_containing_image": round(n_inst / n_img, 3),
                "pct_of_split_instances": round(100 * n_inst / n_inst_total, 2),
            })
        rows.append({
            "split": split, "category": "ALL",
            "n_instances": n_inst_total,
            "n_images_containing": n_images_total,
            "pct_images_containing": 100.0,
            "mean_instances_per_containing_image": round(n_inst_total / n_images_total, 3),
            "pct_of_split_instances": 100.0,
        })
    pd.DataFrame(rows).to_csv(out, index=False)


def size_distribution(eda_dir: Path, out: Path) -> None:
    df = pd.read_csv(eda_dir / "size_band_by_class.csv")
    long = df.melt(id_vars=["split", "category_name"], value_vars=BANDS,
                   var_name="size_band", value_name="n_instances")
    long = long.rename(columns={"category_name": "category"})
    long["size_band"] = pd.Categorical(long["size_band"], categories=BANDS, ordered=True)
    long = long.sort_values(["split", "category", "size_band"])
    tot = long.groupby("split")["n_instances"].transform("sum")
    long["pct_of_split"] = (100 * long["n_instances"] / tot).round(3)
    long.to_csv(out, index=False)


# ----------------------------------------------------------------- A3/A4
def counts_by_class_band(tag: str, ea_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp = pd.read_csv(ea_dir / tag / "fp_summary.csv")
    fp = fp.groupby(["category", "size_band"])["count"].sum().rename("FP").reset_index()
    fn = pd.read_csv(ea_dir / tag / "fn_summary.csv")
    fn = fn.groupby(["category", "size_band"])["count"].sum().rename("FN").reset_index()
    return fp, fn


def gt_by_class_band(eda_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(eda_dir / "size_band_by_class.csv")
    df = df[df.split == "val"]
    long = df.melt(id_vars=["category_name"], value_vars=BANDS,
                   var_name="size_band", value_name="n_gt")
    return long.rename(columns={"category_name": "category"})


def results_table(tag: str, metrics_dir: Path, ea_dir: Path, eda_dir: Path) -> pd.DataFrame:
    ap = pd.read_csv(metrics_dir / f"{tag}_per_class.csv")
    ap = ap[ap.size_band != "all"].rename(columns={
        "mAP@50-95": "AP@50-95", "mAP@50": "AP@50", "AR@50-95": "AR@50-95"})

    gt = gt_by_class_band(eda_dir)
    fp, fn = counts_by_class_band(tag, ea_dir)

    df = (gt.merge(fn, on=["category", "size_band"], how="left")
            .merge(fp, on=["category", "size_band"], how="left")
            .merge(ap, on=["category", "size_band"], how="left"))
    df[["FP", "FN"]] = df[["FP", "FN"]].fillna(0).astype(int)
    df["TP"] = df["n_gt"] - df["FN"]

    denom_p = df["TP"] + df["FP"]
    denom_r = df["TP"] + df["FN"]
    df[f"precision@conf{CONF_THR}"] = (df["TP"] / denom_p).where(denom_p > 0).round(4)
    df[f"recall@conf{CONF_THR}"] = (df["TP"] / denom_r).where(denom_r > 0).round(4)
    for c in ("AP@50", "AP@50-95", "AR@50-95"):
        df[c] = df[c].round(4)

    df["size_band"] = pd.Categorical(df["size_band"], categories=BANDS, ordered=True)
    return df.sort_values(["category", "size_band"])


def verify_tp(tag: str, df: pd.DataFrame, ea_dir: Path) -> None:
    """08'in summary.txt'sindeki TP ile turetilen TP toplamini karsilastirir."""
    txt = (ea_dir / tag / "summary.txt").read_text(encoding="utf-8")
    line = next((l for l in txt.splitlines() if l.startswith("TP:")), None)
    if line is None:
        return
    reported = int(line.split()[1])
    derived = int(df["TP"].sum())
    if reported != derived:
        raise SystemExit(f"{tag}: TP tutarsiz - 08 raporu {reported}, turetilen {derived}")
    print(f"  {tag}: TP dogrulandi ({derived})")


# ----------------------------------------------------------------- A5
def errors_summary(ea_dir: Path, out: Path) -> None:
    rows = []
    for tag, meta in MODELS.items():
        man = pd.read_csv(ea_dir / tag / "manifest.csv")
        fp = man[man.error_kind == "FP"]
        fn = man[man.error_kind == "FN"]

        for band in BANDS:
            sub_fn = fn[fn.size_band == band]
            rows.append({
                "model": meta["model"], "query_set": meta["query_set"], "tag": tag,
                "size_band": band, "error_type": "FN", "fp_subtype": "",
                "count": len(sub_fn),
                "mean_confidence": "",  # FN'de tespit yok -> guven skoru tanimsiz
                "mean_edge_px": round(sub_fn["edge_len"].mean(), 2) if len(sub_fn) else "",
            })
            band_fp = fp[fp.size_band == band]
            for sub in ["background", "class_confusion", "localization", "duplicate"]:
                s = band_fp[band_fp.fp_type == sub]
                rows.append({
                    "model": meta["model"], "query_set": meta["query_set"], "tag": tag,
                    "size_band": band, "error_type": "FP", "fp_subtype": sub,
                    "count": len(s),
                    "mean_confidence": round(s["score"].mean(), 4) if len(s) else "",
                    "mean_edge_px": round(s["edge_len"].mean(), 2) if len(s) else "",
                })
            rows.append({
                "model": meta["model"], "query_set": meta["query_set"], "tag": tag,
                "size_band": band, "error_type": "FP", "fp_subtype": "ALL",
                "count": len(band_fp),
                "mean_confidence": round(band_fp["score"].mean(), 4) if len(band_fp) else "",
                "mean_edge_px": round(band_fp["edge_len"].mean(), 2) if len(band_fp) else "",
            })
    pd.DataFrame(rows).to_csv(out, index=False)


# ----------------------------------------------------------------- A6
def calibration(ea_dir: Path, out: Path) -> None:
    frames = []
    for tag, meta in MODELS.items():
        c = pd.read_csv(ea_dir / tag / "calibration.csv")
        c.insert(0, "tag", tag)
        c.insert(0, "query_set", meta["query_set"])
        c.insert(0, "model", meta["model"])
        frames.append(c)
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "n": "n_detections", "mean_conf": "mean_confidence", "accuracy": "fraction_correct"})
    df["mean_confidence"] = df["mean_confidence"].round(4)
    df["fraction_correct"] = df["fraction_correct"].round(4)
    df.to_csv(out, index=False)


# ----------------------------------------------------------------- A7
def timing(pred_dir: Path, out: Path) -> None:
    rows = []
    for tag, meta in MODELS.items():
        with open(pred_dir / f"{tag}_meta.json", encoding="utf-8") as f:
            m = json.load(f)
        sp = m.get("mean_speed_ms", {})
        total = sum(sp.values()) if sp else None
        rows.append({
            "model": meta["model"],
            "query_set": meta["query_set"],
            "tag": tag,
            "imgsz": m["imgsz"],
            "hardware": "NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41",
            "batch_size": 16,           # 04/06'da --chunk varsayilani
            "preprocess_ms": sp.get("preprocess", ""),
            "inference_ms": sp.get("inference", ""),
            "postprocess_ms": sp.get("postprocess", ""),
            "total_ms_per_image": round(total, 2) if total else "",
            "throughput_img_per_s_wallclock": round(m["n_images"] / m["wall_seconds"], 2),
            # --- asagidakiler OLCULMEDI (adim 5 kosulmadi) ---
            "latency_ms_batch1": "",
            "fps_batch1": "",
            "warmup_excluded": "",
            "measurement_note": ("kaba gosterge: chunk=16, isinma turlari haric tutulmadi, "
                                 "batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali."),
        })
    pd.DataFrame(rows).to_csv(out, index=False)


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir", type=Path, default=Path("data/annotations"))
    p.add_argument("--eda-dir", type=Path, default=Path("outputs/eda"))
    p.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--ea-dir", type=Path, default=Path("outputs/error_analysis"))
    p.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/tables"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats(args.ann_dir, args.out_dir / "dataset_stats.csv")
    size_distribution(args.eda_dir, args.out_dir / "size_distribution.csv")

    print("TP ozdesligi kontrolu:")
    closed = results_table("baseline_a_pilot", args.metrics_dir, args.ea_dir, args.eda_dir)
    verify_tp("baseline_a_pilot", closed, args.ea_dir)
    closed.insert(0, "model", MODELS["baseline_a_pilot"]["model"])
    closed.to_csv(args.out_dir / "results_closed_set.csv", index=False)

    ov = []
    for tag in ["baseline_b_canonical", "baseline_b_attributed"]:
        df = results_table(tag, args.metrics_dir, args.ea_dir, args.eda_dir)
        verify_tp(tag, df, args.ea_dir)
        df.insert(0, "query_set", MODELS[tag]["query_set"])
        df.insert(0, "model", MODELS[tag]["model"])
        ov.append(df)
    pd.concat(ov, ignore_index=True).to_csv(args.out_dir / "results_openvocab.csv", index=False)

    errors_summary(args.ea_dir, args.out_dir / "errors_summary.csv")
    calibration(args.ea_dir, args.out_dir / "calibration.csv")
    timing(args.pred_dir, args.out_dir / "timing.csv")

    print(f"\nCiktilar: {args.out_dir}/")
    for f in sorted(args.out_dir.glob("*.csv")):
        print(f"  {f.name:28s} {sum(1 for _ in open(f)) - 1:5d} satir")


if __name__ == "__main__":
    main()
