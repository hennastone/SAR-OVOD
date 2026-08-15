"""
Gozetimsiz kosu surucusu. Tek komutla baslatilir, PC basinda kimse olmadan
bastan sona akar, kaldigi yerden devam eder.

    python scripts/run_unattended.py

Neden bu var: 8+ saatlik is bir sohbet oturumuna baglanmamali. Bu script
dogrudan isletim sisteminde calisir; Claude Code kapansa da devam eder.

Ozellikler
  - Asamalar sirali; her asama kendi log dosyasina yazar (outputs/logs/)
  - Durum outputs/run_state.json icinde tutulur -> tekrar calistirilinca
    tamamlanmis asamalar ATLANIR (resume)
  - required=False olan asama patlarsa kosu devam eder, sadece isaretlenir
  - required=True olan asama patlarsa kosu durur (sonraki asamalar ona bagli)
  - Her asama bitiminde durum diske yazilir; elektrik giderse ilerleme kaybolmaz

Bayraklar
  --only train_full           sadece belirli asama(lar)i kosar
  --skip speed                belirli asama(lar)i atlar
  --redo eval_a_full          tamamlanmis sayilsa bile yeniden kosar
  --list                      asamalari ve durumlarini yazip cikar
  --dry-run                   komutlari yazdirir, calistirmaz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOGS = ROOT / "outputs" / "logs"
STATE = ROOT / "outputs" / "run_state.json"

FULL_W = "outputs/runs/baseline_a_full_1280/weights/best.pt"


def s(name, args, required=True, note=""):
    return {"name": name, "args": args, "required": required, "note": note}


# Sira onemli: her asama kendinden oncekinin ciktisina dayanabiliyor.
STAGES = [
    s("train_full", ["scripts/03_train_yolo.py", "--preset", "full",
                     "--name", "baseline_a_full_1280", "--workers", "8"],
      note="~8.2 saat. Madde 1 - bloklayici."),

    s("predict_a_full", ["scripts/04_predict_to_json.py", "--weights", FULL_W,
                         "--imgsz", "1280", "--tag", "baseline_a_full"],
      note="1280 agirligiyla val tahminleri"),

    s("eval_a_full", ["scripts/05_eval_detection.py",
                      "--pred", "outputs/predictions/baseline_a_full.json",
                      "--tag", "baseline_a_full"],
      note="Boyut bandi metrikleri. <16 bandi hala olu mu?"),

    # Madde 7: literaturle kiyaslanabilir standart bantlar
    s("eval_a_full_coco", ["scripts/05_eval_detection.py",
                           "--pred", "outputs/predictions/baseline_a_full.json",
                           "--tag", "baseline_a_full", "--bands", "coco"],
      required=False, note="Madde 7 - AP_S / AP_M / AP_L"),

    # Baseline B'yi de 1280'de kosuyoruz - A ile ayni olcekte olmazsa
    # karsilastirma gecersiz.
    s("predict_b_canon_1280", ["scripts/06_predict_yoloworld.py",
                               "--prompt-set", "canonical",
                               "--tag", "baseline_b_canonical_1280", "--imgsz", "1280"],
      required=False),
    s("predict_b_attr_1280", ["scripts/06_predict_yoloworld.py",
                              "--prompt-set", "attributed",
                              "--tag", "baseline_b_attributed_1280", "--imgsz", "1280"],
      required=False),
    s("eval_b_canon_1280", ["scripts/05_eval_detection.py",
                            "--pred", "outputs/predictions/baseline_b_canonical_1280.json",
                            "--tag", "baseline_b_canonical_1280"], required=False),
    s("eval_b_attr_1280", ["scripts/05_eval_detection.py",
                           "--pred", "outputs/predictions/baseline_b_attributed_1280.json",
                           "--tag", "baseline_b_attributed_1280"], required=False),
    s("eval_b_canon_coco", ["scripts/05_eval_detection.py",
                            "--pred", "outputs/predictions/baseline_b_canonical_1280.json",
                            "--tag", "baseline_b_canonical_1280", "--bands", "coco"],
      required=False),
    s("eval_b_attr_coco", ["scripts/05_eval_detection.py",
                           "--pred", "outputs/predictions/baseline_b_attributed_1280.json",
                           "--tag", "baseline_b_attributed_1280", "--bands", "coco"],
      required=False),

    # Madde 6: guven skoru TP/FP ayrismasi (GPU'suz, tahmin dosyalarindan)
    s("conf_separation", ["scripts/13_confidence_separation.py", "--tags",
                          "baseline_a_full", "baseline_b_canonical_1280",
                          "baseline_b_attributed_1280"],
      required=False, note="Madde 6 - AUROC/KS/ortusme + en iyi esik"),

    # Madde 5: guven esigi taramasi. 0.25 ana esik, digerleri duyarlilik icin.
    s("errors_a_full_c025", ["scripts/08_error_analysis.py",
                             "--tag", "baseline_a_full", "--conf", "0.25"],
      note="localization payi dustu mu? net_izole FN kayboldu mu?"),
    s("errors_a_full_c010", ["scripts/08_error_analysis.py", "--tag", "baseline_a_full",
                             "--conf", "0.10", "--no-crops"], required=False),
    s("errors_a_full_c040", ["scripts/08_error_analysis.py", "--tag", "baseline_a_full",
                             "--conf", "0.40", "--no-crops"], required=False),
    s("errors_a_full_c050", ["scripts/08_error_analysis.py", "--tag", "baseline_a_full",
                             "--conf", "0.50", "--no-crops"], required=False),

    # Madde 2: gecikme/FPS. Egitim bittikten SONRA kosmali - GPU'yu tek
    # basina kullanmasi sart, yoksa olcum gecersiz.
    s("speed", ["scripts/12_benchmark_speed.py",
                "--spec", f"YOLO11s@1280={FULL_W}",
                "--spec", "YOLO11s@640=outputs/runs/baseline_a_pilot_640/weights/best.pt",
                "--n", "300", "--warmup", "50"],
      required=False, note="Madde 2. batch=1, isinma haric."),

    s("compare", ["scripts/07_compare_baselines.py",
                  "--tags", "baseline_a_full", "baseline_b_canonical_1280",
                  "baseline_b_attributed_1280",
                  "--labels", "A: YOLO11s @1280", "B: YOLO-World kanonik",
                  "B: YOLO-World oznitelikli"], required=False),

    s("tables", ["scripts/09_build_tables.py"], required=False),
    s("export_md", ["scripts/11_export_tables_md.py"], required=False),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"started": now(), "stages": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st["updated"] = now()
    STATE.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")


def run_stage(stage: dict, st: dict, dry: bool) -> bool:
    name = stage["name"]
    log = LOGS / f"{name}.log"
    # -u : tamponsuz. Olmadan asama loglari saatlerce bos gorunur ve
    # uzaktan ilerleme takip edilemez.
    cmd = [PY, "-u"] + stage["args"]

    print(f"\n{'='*70}\n[{now()}] ASAMA: {name}")
    if stage["note"]:
        print(f"  not: {stage['note']}")
    print(f"  komut: {' '.join(cmd)}")
    print(f"  log  : {log}")

    if dry:
        return True

    LOGS.mkdir(parents=True, exist_ok=True)
    st["stages"][name] = {"status": "running", "started": now(), "log": str(log)}
    save_state(st)

    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"# {name}\n# {' '.join(cmd)}\n# baslangic {now()}\n\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0

    ok = rc == 0
    st["stages"][name] = {
        "status": "ok" if ok else "failed",
        "returncode": rc,
        "seconds": round(dt, 1),
        "finished": now(),
        "log": str(log),
        "required": stage["required"],
    }
    save_state(st)

    print(f"  sonuc: {'OK' if ok else 'HATA (rc=%d)' % rc}  ({dt/60:.1f} dk)")
    if not ok:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print("  --- log sonu ---")
        for line in tail:
            print("  " + line)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--skip", nargs="+", default=[])
    p.add_argument("--redo", nargs="+", default=[])
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    st = load_state()

    if args.list:
        print(f"{'asama':26s} {'durum':10s} {'sure':>8s}  not")
        for stg in STAGES:
            info = st["stages"].get(stg["name"], {})
            dur = f"{info.get('seconds', 0)/60:.1f} dk" if info.get("seconds") else "-"
            req = "" if stg["required"] else "(opsiyonel) "
            print(f"{stg['name']:26s} {info.get('status','-'):10s} {dur:>8s}  {req}{stg['note']}")
        return

    selected = [x for x in STAGES
                if (args.only is None or x["name"] in args.only)
                and x["name"] not in args.skip]

    print(f"[{now()}] gozetimsiz kosu basliyor - {len(selected)} asama")
    print(f"durum dosyasi: {STATE}")

    t_all = time.perf_counter()
    for stg in selected:
        name = stg["name"]
        done = st["stages"].get(name, {}).get("status") == "ok"
        if done and name not in args.redo:
            print(f"\n[atlandi] {name} zaten tamamlanmis "
                  f"({st['stages'][name].get('seconds', 0)/60:.1f} dk). "
                  f"Tekrar icin: --redo {name}")
            continue

        ok = run_stage(stg, st, args.dry_run)
        if not ok and stg["required"]:
            print(f"\n[DURDU] '{name}' zorunlu asamasi basarisiz. "
                  f"Sonraki asamalar ona bagli oldugu icin kosu kesildi.")
            print(f"Duzeltip ayni komutu tekrar calistirin - "
                  f"tamamlanan asamalar atlanacak.")
            sys.exit(1)
        if not ok:
            print(f"[devam] '{name}' opsiyonel, kosu suruyor.")

    total = time.perf_counter() - t_all
    print(f"\n{'='*70}\n[{now()}] KOSU BITTI - toplam {total/3600:.2f} saat")
    okc = sum(1 for v in st["stages"].values() if v.get("status") == "ok")
    bad = [k for k, v in st["stages"].items() if v.get("status") == "failed"]
    print(f"basarili: {okc}   basarisiz: {len(bad)}")
    if bad:
        print("basarisiz asamalar:", ", ".join(bad))
    print(f"\nDurum ozeti icin: python scripts/run_unattended.py --list")


if __name__ == "__main__":
    main()
