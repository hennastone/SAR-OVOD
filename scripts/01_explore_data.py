"""
SeaDronesSee (Object Detection v2) - anotasyon kesfi.

Girdi : data/annotations/instances_{train,val}.json (COCO formati)
Cikti : outputs/eda/ altina csv + png

Boyut metrigi: sqrt(bbox_w * bbox_h) ("esdeger kare kenari"). Bbox kare
olmadigi icin bu bir tasarim tercihi - kullanicidan onay alindi.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZE_BINS = [0, 16, 32, 64, np.inf]
SIZE_LABELS = ["<16", "16-32", "32-64", ">64"]


def load_split(json_path: Path, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    images = pd.DataFrame(coco["images"])
    images["altitude"] = images["meta"].apply(
        lambda m: m.get("height_above_takeoff(meter)") if isinstance(m, dict) else np.nan
    )
    images = images[["id", "file_name", "width", "height", "altitude"]].rename(
        columns={"id": "image_id", "width": "img_w", "height": "img_h"}
    )
    images["split"] = split_name

    anns = pd.DataFrame(coco["annotations"])
    anns["bbox_w"] = anns["bbox"].apply(lambda b: b[2])
    anns["bbox_h"] = anns["bbox"].apply(lambda b: b[3])
    anns["edge_len"] = np.sqrt(anns["bbox_w"] * anns["bbox_h"])
    anns["size_band"] = pd.cut(anns["edge_len"], bins=SIZE_BINS, labels=SIZE_LABELS, right=False)
    anns["category_name"] = anns["category_id"].map(cat_name)
    anns["split"] = split_name
    anns = anns[["split", "image_id", "category_id", "category_name",
                 "bbox_w", "bbox_h", "area", "edge_len", "size_band"]]

    return images, anns


def class_distribution(anns: pd.DataFrame, out_dir: Path) -> None:
    table = anns.groupby(["split", "category_name"]).size().unstack("split", fill_value=0)
    table["total"] = table.sum(axis=1)
    table.sort_values("total", ascending=False).to_csv(out_dir / "class_distribution.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    table.drop(columns="total").plot(kind="bar", ax=ax)
    ax.set_ylabel("nesne sayisi")
    ax.set_title("Sinif dagilimi (split bazinda)")
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def objects_per_image(anns: pd.DataFrame, images: pd.DataFrame, out_dir: Path) -> None:
    counts = anns.groupby(["split", "image_id"]).size().rename("n_objects").reset_index()
    # anotasyonu olmayan goruntu varsa 0 olarak eklensin
    counts = images[["split", "image_id"]].merge(counts, on=["split", "image_id"], how="left")
    counts["n_objects"] = counts["n_objects"].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split, g in counts.groupby("split"):
        ax.hist(g["n_objects"], bins=range(0, int(counts["n_objects"].max()) + 2),
                alpha=0.5, label=split)
    ax.set_xlabel("goruntu basina nesne sayisi")
    ax.set_ylabel("goruntu sayisi")
    ax.legend()
    ax.set_title("Goruntu basina nesne sayisi dagilimi")
    fig.tight_layout()
    fig.savefig(out_dir / "objects_per_image_hist.png", dpi=150)
    plt.close(fig)

    counts.groupby("split")["n_objects"].describe().to_csv(out_dir / "objects_per_image_stats.csv")


def object_area_hist(anns: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, g in anns.groupby("split"):
        ax.hist(g["area"], bins=np.logspace(np.log10(max(g["area"].min(), 1)),
                                             np.log10(g["area"].max()), 50),
                alpha=0.5, label=split)
    ax.set_xscale("log")
    ax.set_xlabel("nesne alani (piksel^2, log)")
    ax.set_ylabel("nesne sayisi")
    ax.legend()
    ax.set_title("Nesne alani dagilimi")
    fig.tight_layout()
    fig.savefig(out_dir / "object_area_hist.png", dpi=150)
    plt.close(fig)


def size_bands(anns: pd.DataFrame, out_dir: Path) -> None:
    table = anns.groupby(["split", "size_band"], observed=True).size().unstack("size_band", fill_value=0)
    table = table[SIZE_LABELS]
    table.loc["total"] = table.sum()
    table.to_csv(out_dir / "size_band_counts.csv")

    by_class = anns.groupby(["split", "category_name", "size_band"], observed=True).size().unstack(
        "size_band", fill_value=0)[SIZE_LABELS]
    by_class.to_csv(out_dir / "size_band_by_class.csv")


def altitude_hist(images: pd.DataFrame, out_dir: Path) -> None:
    missing = images["altitude"].isna().sum()
    total = len(images)
    fig, ax = plt.subplots(figsize=(8, 5))
    for split, g in images.groupby("split"):
        vals = g["altitude"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.5, label=split)
    ax.set_xlabel("irtifa (m, height_above_takeoff)")
    ax.set_ylabel("goruntu sayisi")
    ax.legend()
    ax.set_title(f"Irtifa dagilimi (meta eksik: {missing}/{total} goruntu)")
    fig.tight_layout()
    fig.savefig(out_dir / "altitude_hist.png", dpi=150)
    plt.close(fig)


def image_resolution_table(images: pd.DataFrame, out_dir: Path) -> None:
    table = images.groupby(["split", "img_w", "img_h"]).size().rename("n_images").reset_index()
    table.sort_values(["split", "n_images"], ascending=[True, False]).to_csv(
        out_dir / "image_resolution_counts.csv", index=False
    )


def write_summary(images: pd.DataFrame, anns: pd.DataFrame, out_dir: Path) -> None:
    lines = []
    for split, g in images.groupby("split"):
        n_ann = len(anns[anns["split"] == split])
        lines.append(f"{split}: {len(g)} goruntu, {n_ann} nesne, "
                      f"{g['altitude'].isna().sum()} goruntude irtifa metasi yok")
    text = "\n".join(lines)
    print(text)
    (out_dir / "eda_summary.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_tr, ann_tr = load_split(args.ann_dir / "instances_train.json", "train")
    img_val, ann_val = load_split(args.ann_dir / "instances_val.json", "val")

    images = pd.concat([img_tr, img_val], ignore_index=True)
    anns = pd.concat([ann_tr, ann_val], ignore_index=True)

    class_distribution(anns, args.out_dir)
    objects_per_image(anns, images, args.out_dir)
    object_area_hist(anns, args.out_dir)
    size_bands(anns, args.out_dir)
    altitude_hist(images, args.out_dir)
    image_resolution_table(images, args.out_dir)
    write_summary(images, anns, args.out_dir)

    print(f"\nCiktilar: {args.out_dir}/")


if __name__ == "__main__":
    main()
