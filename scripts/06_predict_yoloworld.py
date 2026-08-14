"""
Baseline B - acik kelime dagarcikli detector: YOLO-World zero-shot.

Girdi : scripts/prompt_sets.json, data/annotations/instances_val.json, data/images/val/
Cikti : outputs/predictions/<tag>.json + <tag>_meta.json
        (04_predict_to_json.py ile AYNI sema -> 05_eval_detection.py dogrudan calisir)

Kritik nokta - cok-e-bir prompt eslemesi:
  'attributed' setinde bir sinifa birden fazla ifade esleniyor. Modelin kendi NMS'i
  prompt-sinifi bazinda calistigi icin ayni nesne uzerinde iki farkli ifade birden
  atesleyebiliyor; SeaDronesSee sinifina eslendikten sonra bunlar ayni sinifta
  mukerrer kutu haline gelir ve yanlis pozitif olarak sayilir. Bu yuzden esleme
  SONRASI sinif-bazli NMS uygulanir. 'canonical' setinde esleme 1:1 oldugu icin
  bu adim etkisizdir (yine de calisir, zarari yok).

Ornek:
  python scripts/06_predict_yoloworld.py --prompt-set canonical  --tag baseline_b_canonical  --imgsz 640
  python scripts/06_predict_yoloworld.py --prompt-set attributed --tag baseline_b_attributed --imgsz 640
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.ops import batched_nms
from ultralytics import YOLOWorld

from predict_common import run_chunked


def load_prompt_set(path: Path, name: str) -> tuple[list[str], list[int], str]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if name not in cfg or name.startswith("_"):
        raise SystemExit(f"'{name}' sorgu seti yok. Mevcut: "
                         f"{[k for k in cfg if not k.startswith('_')]}")
    entry = cfg[name]
    texts = [p["text"] for p in entry["prompts"]]
    cat_ids = [p["category_id"] for p in entry["prompts"]]
    return texts, cat_ids, entry.get("description", "")


def load_image_index(val_json: Path) -> dict[str, int]:
    with open(val_json, encoding="utf-8") as f:
        data = json.load(f)
    return {im["file_name"]: im["id"] for im in data["images"]}


def merge_by_class(xyxy, conf, cat_ids, iou_thres: float):
    """Prompt -> SeaDronesSee sinifi eslemesi sonrasi sinif-bazli NMS."""
    if len(xyxy) == 0:
        return xyxy, conf, cat_ids
    boxes = torch.as_tensor(xyxy, dtype=torch.float32)
    scores = torch.as_tensor(conf, dtype=torch.float32)
    idxs = torch.as_tensor(cat_ids, dtype=torch.int64)
    keep = batched_nms(boxes, scores, idxs, iou_thres)
    return boxes[keep].numpy(), scores[keep].numpy(), idxs[keep].numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-set", required=True, help="prompt_sets.json icindeki set adi")
    p.add_argument("--tag", required=True)
    p.add_argument("--prompts", type=Path, default=Path("scripts/prompt_sets.json"))
    p.add_argument("--model", default="yolov8s-worldv2.pt")
    p.add_argument("--val-json", type=Path, default=Path("data/annotations/instances_val.json"))
    p.add_argument("--img-dir", type=Path, default=Path("data/images/val"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/predictions"))
    p.add_argument("--imgsz", type=int, default=640, help="Baseline A ile AYNI olmali")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU esigi (model ici + esleme sonrasi)")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--chunk", type=int, default=16, help="tek seferde islenen goruntu sayisi (bellek)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    texts, prompt_to_cat, desc = load_prompt_set(args.prompts, args.prompt_set)
    img_index = load_image_index(args.val_json)
    paths = [args.img_dir / n for n in img_index]
    if args.limit:
        paths = paths[: args.limit]

    model = YOLOWorld(args.model)
    model.set_classes(texts)

    detections = []
    speeds = []
    n_before = n_after = 0
    t0 = time.perf_counter()
    for path, xyxy, conf, prompt_idx, speed in run_chunked(
        model, paths, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
        max_det=args.max_det, device=args.device, chunk=args.chunk,
        desc=args.tag,
    ):
        image_id = img_index[Path(path).name]
        speeds.append(speed)
        if len(xyxy) == 0:
            continue
        cat_ids = [prompt_to_cat[i] for i in prompt_idx]

        n_before += len(xyxy)
        xyxy, conf, cat_ids = merge_by_class(xyxy, conf, cat_ids, args.iou)
        n_after += len(xyxy)

        for (x1, y1, x2, y2), sc, c in zip(xyxy, conf, cat_ids):
            detections.append({
                "image_id": image_id,
                "category_id": int(c),
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
        "baseline": "B (YOLO-World zero-shot)",
        "model": args.model,
        "prompt_set": args.prompt_set,
        "prompt_set_description": desc,
        "prompts": [{"text": t, "category_id": c} for t, c in zip(texts, prompt_to_cat)],
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_nms": args.iou,
        "max_det": args.max_det,
        "n_images": len(paths),
        "n_detections": len(detections),
        "dedup_removed_by_class_nms": n_before - n_after,
        "wall_seconds": round(wall, 2),
        "mean_speed_ms": {
            k: round(sum(s[k] for s in speeds) / len(speeds), 2) for k in speeds[0]
        } if speeds else {},
        "note": "hiz kaba gostergedir; resmi olcum 08_benchmark_speed.py",
    }
    with open(args.out_dir / f"{args.tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in meta.items() if k != "prompts"}, indent=2, ensure_ascii=False))
    print(f"\nTahminler: {out_json}")
    print(f"Sonraki adim: python scripts/05_eval_detection.py --pred {out_json} --tag {args.tag}")


if __name__ == "__main__":
    main()
