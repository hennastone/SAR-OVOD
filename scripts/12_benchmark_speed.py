"""
Adim 5 - gecikme/FPS olcumu. batch=1, isinma turlari HARIC.

Neden ayri script: 04/06'nin meta dosyalarindaki hiz sayilari chunk=16 ile
alinmisti ve isinma dahildi; karsilastirmali iddia icin kullanilamaz.

Olcum yontemi:
  - Her cagri oncesi ve sonrasi torch.cuda.synchronize() -> GPU asenkron
    kuyrugu zamanlamayi kirletmesin.
  - Ilk --warmup goruntu olculur ama ATILIR (cuDNN autotune, bellek ayirma,
    CUDA context ilk kurulum maliyeti buraya dusuyor).
  - Uctan uca gecikme raporlanir (on isleme + cikarim + NMS), cunku
    operasyonel olarak anlamli olan bu.

Cikti : outputs/tables/timing_batch1.csv  (model x imgsz basina bir satir)
        outputs/logs/benchmark_speed.log uzerinden cagrilirsa ayrica log

Ornek:
  python scripts/12_benchmark_speed.py \
      --spec "YOLO11s@640=outputs/runs/baseline_a_pilot_640/weights/best.pt" \
      --spec "YOLO-World-s@640=yolov8s-worldv2.pt" \
      --n 300 --warmup 50
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO, YOLOWorld


def parse_spec(spec: str) -> tuple[str, int, str]:
    """'ad@imgsz=agirlik' -> (ad, imgsz, agirlik)"""
    name_size, weights = spec.split("=", 1)
    name, size = name_size.rsplit("@", 1)
    return name, int(size), weights


def load_model(weights: str, prompts: Path | None, prompt_set: str | None):
    if prompts and prompt_set:
        with open(prompts, encoding="utf-8") as f:
            cfg = json.load(f)
        texts = [p["text"] for p in cfg[prompt_set]["prompts"]]
        m = YOLOWorld(weights)
        m.set_classes(texts)
        return m, len(texts)
    return YOLO(weights), None


def bench(model, paths, imgsz, conf, iou, max_det, device, warmup, n) -> dict:
    lat_ms = []
    total = warmup + n
    for i in range(total):
        p = str(paths[i % len(paths)])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=p, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:                      # isinma turlarini AT
            lat_ms.append((t1 - t0) * 1000)

    lat_ms.sort()
    mean = statistics.mean(lat_ms)
    return {
        "n_measured": len(lat_ms),
        "warmup_excluded": warmup,
        "latency_ms_mean": round(mean, 3),
        "latency_ms_median": round(statistics.median(lat_ms), 3),
        "latency_ms_p95": round(lat_ms[int(0.95 * len(lat_ms)) - 1], 3),
        "latency_ms_min": round(lat_ms[0], 3),
        "latency_ms_max": round(lat_ms[-1], 3),
        "latency_ms_std": round(statistics.pstdev(lat_ms), 3),
        "fps_batch1": round(1000.0 / mean, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", action="append", required=True,
                   help="'ad@imgsz=agirlik'; birden fazla kez verilebilir")
    p.add_argument("--prompt-set", default=None,
                   help="verilirse tum spec'ler YOLO-World olarak yuklenir")
    p.add_argument("--prompts", type=Path, default=Path("scripts/prompt_sets.json"))
    p.add_argument("--img-dir", type=Path, default=Path("data/images/val"))
    p.add_argument("--out", type=Path, default=Path("outputs/tables/timing_batch1.csv"))
    p.add_argument("--n", type=int, default=300, help="olculecek goruntu sayisi")
    p.add_argument("--warmup", type=int, default=50, help="atilacak isinma turu")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    paths = sorted(args.img_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"{args.img_dir} icinde goruntu yok")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    hw = (f"{gpu}, torch {torch.__version__}, "
          f"{platform.system()} {platform.release()}")

    rows = []
    for spec in args.spec:
        name, imgsz, weights = parse_spec(spec)
        model, n_prompts = load_model(weights, args.prompts if args.prompt_set else None,
                                      args.prompt_set)
        print(f"[bench] {name} @ {imgsz}  ({weights})")
        r = bench(model, paths, imgsz, args.conf, args.iou, args.max_det,
                  args.device, args.warmup, args.n)
        rows.append({
            "model": name, "imgsz": imgsz, "weights": weights,
            "batch_size": 1, "hardware": hw,
            "n_prompts": n_prompts if n_prompts else "",
            "conf": args.conf, "iou_nms": args.iou, **r,
            "method": "torch.cuda.synchronize() ile uctan uca; isinma turlari haric",
        })
        print(f"         {r['latency_ms_mean']:.2f} ms  ->  {r['fps_batch1']:.1f} FPS "
              f"(medyan {r['latency_ms_median']:.2f}, p95 {r['latency_ms_p95']:.2f})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nCikti: {args.out}")


if __name__ == "__main__":
    main()
