"""
Baseline A - kapali kume detector: Ultralytics YOLO fine-tune (SeaDronesSee).

Girdi : outputs/yolo_dataset/data.yaml  (02_coco_to_yolo.py uretir)
Cikti : outputs/runs/<name>/weights/best.pt + ultralytics'in kendi
        egitim grafikleri/loglari

Preset:
  pilot -> imgsz=640,  epochs=10   (pipeline dogrulama, sonuclar bilimsel degil)
  full  -> imgsz=1280, epochs=100  (asil kosu)

Ornek:
  python scripts/03_train_yolo.py --preset pilot
  python scripts/03_train_yolo.py --preset full --model yolo11s.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

PRESETS = {
    "pilot": {"imgsz": 640, "epochs": 10, "batch": 16},
    # batch=8 elle sabit: 12GB VRAM'de AutoBatch (-1) 1280 icin batch=2 seciyor
    # (tahmini de hatali - 5.5G ongorup 2.5G kullaniyor). batch=2 hem %33 daha
    # yavas hem gradyan gurultusu yuksek. Olculen: batch=8 -> 8.3G VRAM, GPU %80,
    # 32.8 goruntu/sn, epoch ~295 sn (validation dahil).
    "full": {"imgsz": 1280, "epochs": 100, "batch": 8},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="pilot")
    p.add_argument("--data", type=Path, default=Path("outputs/yolo_dataset/data.yaml"))
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--project", type=Path, default=Path("outputs/runs"))
    p.add_argument("--name", default=None, help="varsayilan: baseline_a_<preset>_<imgsz>")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # preset override
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=30)
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in ("imgsz", "epochs", "batch"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    name = args.name or f"baseline_a_{args.preset}_{cfg['imgsz']}"
    # DIKKAT: goreli 'project' yolu ultralytics tarafindan kendi RUNS_DIR'i
    # (./runs/detect/) altina gomuluyor. Mutlak yol vermek sart.
    project = args.project.resolve()

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        val=True,
    )

    weights = project / name / "weights" / "best.pt"
    print(f"\nEn iyi agirlik: {weights}")
    print(f"Sonraki adim: python scripts/04_predict_to_json.py --weights {weights} "
          f"--imgsz {cfg['imgsz']} --tag baseline_a_{args.preset}")


if __name__ == "__main__":
    main()
