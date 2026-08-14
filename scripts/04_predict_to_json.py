"""
Bir detector'u SeaDronesSee val setinde kosturup tahmin dosyasi uretir.

Baseline A ve B icin ORTAK giris noktasi - ayni tahmin/eval yolundan gectikleri
icin metrikler karsilastirilabilir kalir.

Cikti, degerlendirme kutuphanesinin (pycocotools) bekledigi standart detection
JSON semasinda yazilir: [{image_id, category_id, bbox:[x,y,w,h], score}, ...].
image_id'ler SeaDronesSee instances_val.json'daki id'lerdir.

Girdi : agirlik (.pt), data/annotations/instances_val.json, data/images/val/
Cikti : outputs/predictions/<tag>.json          (detection listesi)
        outputs/predictions/<tag>_meta.json     (kosum ayarlari + ham hiz notu)

Not: conf esigi bilerek cok dusuk (0.001). mAP'in precision-recall egrisini tam
kurabilmesi icin gerekli; esikleme hata analizinde ayrica yapilir.
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

from predict_common import run_chunked


def load_image_index(val_json: Path) -> dict[str, int]:
    with open(val_json, encoding="utf-8") as f:
        coco = json.load(f)
    return {im["file_name"]: im["id"] for im in coco["images"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help=".pt agirlik yolu veya model adi")
    p.add_argument("--tag", required=True, help="cikti dosya adi (orn. baseline_a_pilot)")
    p.add_argument("--val-json", type=Path, default=Path("data/annotations/instances_val.json"))
    p.add_argument("--img-dir", type=Path, default=Path("data/images/val"))
    p.add_argument("--class-map", type=Path,
                   default=Path("outputs/yolo_dataset/class_mapping.json"),
                   help="model sinif idx -> SeaDronesSee category_id esleme dosyasi")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU esigi")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--chunk", type=int, default=16, help="tek seferde islenen goruntu sayisi (bellek)")
    p.add_argument("--limit", type=int, default=None, help="hizli test icin ilk N goruntu")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_index = load_image_index(args.val_json)
    with open(args.class_map, encoding="utf-8") as f:
        cmap = json.load(f)
    yolo_to_sds = {int(k): int(v) for k, v in cmap["yolo_to_sds"].items()}

    paths = [args.img_dir / n for n in img_index]
    if args.limit:
        paths = paths[: args.limit]

    model = YOLO(args.weights)

    detections = []
    speeds = []
    t0 = time.perf_counter()
    for path, xyxy, conf, cls, speed in run_chunked(
        model, paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, device=args.device, chunk=args.chunk,
        desc=args.tag,
    ):
        image_id = img_index[Path(path).name]
        speeds.append(speed)
        for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cls):
            if c not in yolo_to_sds:
                continue
            detections.append({
                "image_id": image_id,
                "category_id": yolo_to_sds[c],
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(sc), 5),
            })
    wall = time.perf_counter() - t0

    out_json = args.out_dir / f"{args.tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    meta = {
        "tag": args.tag,
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_nms": args.iou,
        "max_det": args.max_det,
        "n_images": len(paths),
        "n_detections": len(detections),
        "wall_seconds": round(wall, 2),
        "mean_speed_ms": {
            k: round(sum(s[k] for s in speeds) / len(speeds), 2)
            for k in speeds[0]
        } if speeds else {},
        "note": "hiz sadece kaba gostergedir; resmi olcum 08_benchmark_speed.py",
    }
    with open(args.out_dir / f"{args.tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nTahminler: {out_json}")
    print(f"Sonraki adim: python scripts/05_eval_detection.py --pred {out_json} --tag {args.tag}")


if __name__ == "__main__":
    main()
