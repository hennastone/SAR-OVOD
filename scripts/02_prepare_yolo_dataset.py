"""
SeaDronesSee anotasyonlarini Ultralytics YOLO formatina cevirir.

SeaDronesSee anotasyonlarini COCO JSON *formatinda* dagitiyor (instances_*.json).
Burada COCO veri seti degil, sadece o dosya formati kastediliyor - sinif listesi
tamamen SeaDronesSee'nin 5 sinifi.

Girdi : data/annotations/instances_{train,val}.json, data/images/{train,val}/
Cikti : outputs/yolo_dataset/
          images/{train,val}/   (link; data/ kopyalanmaz, degistirilmez)
          labels/{train,val}/   (*.txt)
          data.yaml             (ultralytics dataset tanimi)
          class_mapping.json    (YOLO idx <-> SeaDronesSee category_id, eval icin)

Not: SeaDronesSee'nin 'ignored' (id=0) sinifi anotasyonlarda hic gecmiyor, atlaniyor.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

# SeaDronesSee category_id -> (YOLO class index, isim). id=0 'ignored' kullanilmiyor.
SDS_TO_YOLO = {
    1: (0, "swimmer"),
    2: (1, "boat"),
    3: (2, "jetski"),
    4: (3, "life_saving_appliances"),
    5: (4, "buoy"),
}


def link_tree(src_dir: Path, dst_dir: Path, file_names: list[str], mode: str) -> str:
    """Goruntuleri dst_dir altina baglar. data/ salt-okunur kalir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    order = {"auto": ["symlink", "hardlink", "copy"]}.get(mode, [mode])

    used = None
    for attempt in order:
        try:
            probe_src, probe_dst = src_dir / file_names[0], dst_dir / file_names[0]
            if probe_dst.exists() or probe_dst.is_symlink():
                probe_dst.unlink()
            if attempt == "symlink":
                os.symlink(probe_src.resolve(), probe_dst)
            elif attempt == "hardlink":
                os.link(probe_src, probe_dst)
            else:
                shutil.copy2(probe_src, probe_dst)
            used = attempt
            break
        except OSError:
            continue
    if used is None:
        raise RuntimeError(f"{src_dir} -> {dst_dir} baglanamadi (symlink/hardlink/copy hepsi basarisiz)")

    for name in file_names[1:]:
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            continue
        if used == "symlink":
            os.symlink((src_dir / name).resolve(), dst)
        elif used == "hardlink":
            os.link(src_dir / name, dst)
        else:
            shutil.copy2(src_dir / name, dst)
    return used


def convert_split(ann_path: Path, img_src: Path, out_root: Path, split: str,
                  link_mode: str) -> dict:
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    imgs = {im["id"]: im for im in coco["images"]}
    per_image: dict[int, list[str]] = {i: [] for i in imgs}

    skipped_cat = 0
    clipped = 0
    degenerate = 0

    for a in coco["annotations"]:
        if a["category_id"] not in SDS_TO_YOLO:
            skipped_cat += 1
            continue
        cls = SDS_TO_YOLO[a["category_id"]][0]
        im = imgs[a["image_id"]]
        iw, ih = im["width"], im["height"]

        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(iw), x2), min(float(ih), y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            clipped += 1
        if cx2 <= cx1 or cy2 <= cy1:
            degenerate += 1
            continue

        cx = ((cx1 + cx2) / 2) / iw
        cy = ((cy1 + cy2) / 2) / ih
        nw = (cx2 - cx1) / iw
        nh = (cy2 - cy1) / ih
        per_image[a["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    lbl_dir = out_root / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_id, lines in per_image.items():
        stem = Path(imgs[img_id]["file_name"]).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    file_names = [im["file_name"] for im in coco["images"]]
    used_mode = link_tree(img_src, out_root / "images" / split, file_names, link_mode)

    return {
        "split": split,
        "images": len(imgs),
        "annotations_written": sum(len(v) for v in per_image.values()),
        "skipped_unmapped_category": skipped_cat,
        "clipped_to_bounds": clipped,
        "dropped_degenerate": degenerate,
        "link_mode": used_mode,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out-root", type=Path, default=Path("outputs/yolo_dataset"))
    p.add_argument("--link-mode", choices=["auto", "symlink", "hardlink", "copy"],
                   default="auto", help="auto: symlink -> hardlink -> copy sirasiyla dener")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    stats = []
    for split in ["train", "val"]:
        stats.append(convert_split(
            args.data_root / "annotations" / f"instances_{split}.json",
            args.data_root / "images" / split,
            args.out_root, split, args.link_mode,
        ))

    names = {idx: name for _, (idx, name) in sorted(SDS_TO_YOLO.items(), key=lambda kv: kv[1][0])}
    data_yaml = {
        "path": str(args.out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    with open(args.out_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    with open(args.out_root / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({
            "sds_to_yolo": {str(k): v[0] for k, v in SDS_TO_YOLO.items()},
            "yolo_to_sds": {str(v[0]): k for k, v in SDS_TO_YOLO.items()},
            "names": {str(k): v for k, v in names.items()},
        }, f, indent=2)

    for s in stats:
        print(s)
    print(f"\ndata.yaml -> {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
