"""
SeaDronesSee val setinde detection degerlendirmesi + boyut bandi kirilimi.

Metrikler pycocotools ile hesaplanir - bu, nesne tespitinde mAP'in fiili standart
kutuphanesi; SeaDronesSee verisi disinda hicbir veri kullanilmaz.

Boyut bantlari sqrt(w*h) kenar uzunlugu uzerinden tanimli; pycocotools areaRng
alan cinsinden calistigi icin bantlar alan karesine cevrilir:
    <16px   -> area <  256
    16-32px -> 256  <= area < 1024
    32-64px -> 1024 <= area < 4096
    >64px   -> area >= 4096

Girdi : outputs/predictions/<tag>.json, data/annotations/instances_val.json
Cikti : outputs/metrics/<tag>_overall.csv     (genel + boyut bandi)
        outputs/metrics/<tag>_per_class.csv   (sinif x boyut bandi mAP)
        outputs/metrics/<tag>_summary.txt

Baseline A ve B ayni script'ten gectigi icin sonuclar dogrudan karsilastirilabilir.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# (etiket, [alan_min, alan_max])
#
# Iki bant tanimi destekleniyor:
#   sqrt : bu projenin tercihi - sqrt(w*h) kenar uzunlugu bantlari
#   coco : literaturle kiyaslanabilirlik icin standart COCO esikleri
#          (small < 32^2, medium 32^2-96^2, large > 96^2)
# Hakemler COCO tablosunu isteyecegi icin ikisi de uretiliyor; sayilar
# ayni kosudan, sadece alan araliklari farkli.
BAND_SETS = {
    "sqrt": [
        ("all", [0.0, 1e10]),
        ("<16", [0.0, 256.0]),
        ("16-32", [256.0, 1024.0]),
        ("32-64", [1024.0, 4096.0]),
        (">64", [4096.0, 1e10]),
    ],
    "coco": [
        ("all", [0.0, 1e10]),
        ("small", [0.0, 1024.0]),        # < 32^2
        ("medium", [1024.0, 9216.0]),    # 32^2 - 96^2
        ("large", [9216.0, 1e10]),       # > 96^2
    ],
}

AREA_RNG = BAND_SETS["sqrt"]   # main() secime gore degistirir


def load_gt(gt_path: Path) -> COCO:
    """SeaDronesSee anotasyonlarinda 'iscrowd' alani yok; pycocotools bunu zorunlu
    tutuyor. Eksikse 0 atanir (veri setinde crowd bolgesi tanimli degil)."""
    gt = COCO(str(gt_path))
    for ann in gt.dataset.get("annotations", []):
        ann.setdefault("iscrowd", 0)
    for ann in gt.anns.values():
        ann.setdefault("iscrowd", 0)
    return gt


def build_eval(gt: COCO, pred_path: Path, cat_ids: list[int]) -> COCOeval:
    with open(pred_path, encoding="utf-8") as f:
        dets = json.load(f)
    if not dets:
        raise SystemExit(f"{pred_path} bos - tahmin yok, eval yapilamaz")
    dt = gt.loadRes(dets)

    e = COCOeval(gt, dt, iouType="bbox")
    e.params.areaRng = [r for _, r in AREA_RNG]
    e.params.areaRngLbl = [lbl for lbl, _ in AREA_RNG]
    # nesne/goruntu en fazla 16; tek maxDets yeterli ve tablolari sadelestirir
    e.params.maxDets = [100]
    # DIKKAT: catIds evaluate()'ten ONCE atanmali - precision dizisinin sinif
    # ekseni bu listeye gore olusuyor, sonra atanirsa indeksler kayiyor.
    e.params.catIds = cat_ids
    e.evaluate()
    e.accumulate()
    return e


def _mean(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(np.mean(x)) if x.size else float("nan")


def extract(e: COCOeval, cat_ids: list[int], cat_names: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # precision: [T(iou), R(recall), K(cat), A(area), M(maxDet)]
    prec = e.eval["precision"]
    rec = e.eval["recall"]  # [T, K, A, M]
    iou_thrs = e.params.iouThrs
    i50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    i75 = int(np.argmin(np.abs(iou_thrs - 0.75)))

    # --- genel: boyut bandi basina ---
    rows = []
    for a, (lbl, _) in enumerate(AREA_RNG):
        rows.append({
            "size_band": lbl,
            "mAP@50-95": _mean(prec[:, :, :, a, 0]),
            "mAP@50": _mean(prec[i50, :, :, a, 0]),
            "mAP@75": _mean(prec[i75, :, :, a, 0]),
            "AR@50-95": _mean(rec[:, :, a, 0]),
            "n_gt": sum(
                1 for ann in e.cocoGt.loadAnns(e.cocoGt.getAnnIds(catIds=cat_ids))
                if AREA_RNG[a][1][0] <= ann["area"] < AREA_RNG[a][1][1]
            ),
        })
    overall = pd.DataFrame(rows)

    # --- sinif x boyut bandi ---
    rows = []
    for k, cid in enumerate(cat_ids):
        for a, (lbl, _) in enumerate(AREA_RNG):
            rows.append({
                "category": cat_names[cid],
                "size_band": lbl,
                "mAP@50-95": _mean(prec[:, :, k, a, 0]),
                "mAP@50": _mean(prec[i50, :, k, a, 0]),
                "AR@50-95": _mean(rec[:, k, a, 0]),
            })
    per_class = pd.DataFrame(rows)
    return overall, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--gt", type=Path, default=Path("data/annotations/instances_val.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--bands", choices=list(BAND_SETS), default="sqrt",
                   help="sqrt: sqrt(w*h) kenar bantlari | coco: standart AP_S/M/L")
    args = p.parse_args()

    # coco secilirse dosya adlarina son ek gelir; sqrt varsayilani mevcut
    # dosya adlarini degistirmez (geriye donuk uyumluluk).
    global AREA_RNG
    AREA_RNG = BAND_SETS[args.bands]
    suffix = "" if args.bands == "sqrt" else f"_{args.bands}"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(args.gt)
    cat_ids = sorted(c for c in gt.getCatIds() if len(gt.getAnnIds(catIds=[c])) > 0)
    cat_names = {c["id"]: c["name"] for c in gt.loadCats(cat_ids)}

    e = build_eval(gt, args.pred, cat_ids)
    overall, per_class = extract(e, cat_ids, cat_names)

    overall.to_csv(args.out_dir / f"{args.tag}{suffix}_overall.csv", index=False)
    per_class.to_csv(args.out_dir / f"{args.tag}{suffix}_per_class.csv", index=False)

    lines = [f"=== {args.tag} ===", "", "Boyut bandina gore (tum siniflar):",
             overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
             "", "Sinif x boyut bandi (mAP@50):",
             per_class.pivot(index="category", columns="size_band", values="mAP@50")[
                 [lbl for lbl, _ in AREA_RNG]
             ].to_string(float_format=lambda v: f"{v:.4f}")]
    text = "\n".join(lines)
    (args.out_dir / f"{args.tag}{suffix}_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nCiktilar: {args.out_dir}/{args.tag}{suffix}_*")


if __name__ == "__main__":
    main()
