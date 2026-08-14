"""
Baseline'lari yan yana karsilastirir: boyut bandina gore bozulma egrisi + tablo.

Girdi : outputs/metrics/<tag>_overall.csv ve <tag>_per_class.csv (05'in ciktilari)
Cikti : outputs/comparison/size_degradation.png
        outputs/comparison/overall_comparison.csv
        outputs/comparison/per_class_map50.csv

Ornek:
  python scripts/07_compare_baselines.py \
      --tags baseline_a_pilot baseline_b_canonical baseline_b_attributed \
      --labels "A: YOLO11s fine-tune" "B: YOLO-World kanonik" "B: YOLO-World oznitelikli"

Renkler: dataviz referans paletinin dogrulanmis ilk uc kategorik slotu
(blue/orange/aqua). Bu ucluk all-pairs gecer - slot sayisi artarsa palet yeniden
dogrulanmali. Statik tez figuru oldugu icin tek (acik) tema.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BANDS = ["<16", "16-32", "32-64", ">64"]

# dataviz referans paleti - kategorik slot 1/2/3 (acik tema)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def load(tags: list[str], metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, per_class = [], []
    for t in tags:
        o = pd.read_csv(metrics_dir / f"{t}_overall.csv")
        o.insert(0, "tag", t)
        overall.append(o)
        pc = pd.read_csv(metrics_dir / f"{t}_per_class.csv")
        pc.insert(0, "tag", t)
        per_class.append(pc)
    return pd.concat(overall, ignore_index=True), pd.concat(per_class, ignore_index=True)


def degradation_plot(overall: pd.DataFrame, tags: list[str], labels: list[str],
                     out_path: Path) -> None:
    n_gt = (overall[overall.tag == tags[0]]
            .set_index("size_band")["n_gt"].reindex(BANDS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    for ax, metric in zip(axes, ["mAP@50", "mAP@50-95"]):
        ax.set_facecolor(SURFACE)
        for i, (tag, label) in enumerate(zip(tags, labels)):
            row = overall[overall.tag == tag].set_index("size_band")[metric].reindex(BANDS)
            ax.plot(BANDS, row.values, marker="o", markersize=8, linewidth=2,
                    color=SERIES[i], label=label, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            # secici dogrudan etiket: sadece en sagdaki nokta
            ax.annotate(f"{row.values[-1]:.2f}", (len(BANDS) - 1, row.values[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        color=INK, fontsize=9, va="center")

        ax.set_title(metric, color=INK, fontsize=12, pad=10)
        ax.set_xlabel("nesne boyut bandi  (sqrt(w*h), piksel)", color=INK_MUTED, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.tick_params(colors=INK_MUTED)
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([f"{b}\n(n={int(n_gt[b])})" for b in BANDS], fontsize=9)
        ax.margins(x=0.12)

    axes[0].set_ylabel("mAP", color=INK_MUTED, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.suptitle("Hedef boyutu kuculdukce performans bozulmasi - SeaDronesSee val",
                 color=INK, fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None, help="grafik icin okunakli isimler")
    p.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/comparison"))
    args = p.parse_args()

    labels = args.labels or args.tags
    if len(labels) != len(args.tags):
        raise SystemExit("--labels sayisi --tags ile ayni olmali")
    if len(args.tags) > len(SERIES):
        raise SystemExit(f"En fazla {len(SERIES)} seri destekleniyor "
                         "(palet bu kadari icin dogrulandi); fazlasi icin palet yeniden dogrulanmali")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall, per_class = load(args.tags, args.metrics_dir)

    wide = overall.pivot(index="size_band", columns="tag",
                         values=["mAP@50", "mAP@50-95", "AR@50-95"]).reindex(BANDS + ["all"])
    wide.to_csv(args.out_dir / "overall_comparison.csv")

    pc = per_class.pivot_table(index=["category", "size_band"], columns="tag", values="mAP@50")
    pc.to_csv(args.out_dir / "per_class_map50.csv")

    degradation_plot(overall, args.tags, labels, args.out_dir / "size_degradation.png")

    print(wide.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nCiktilar: {args.out_dir}/")


if __name__ == "__main__":
    main()
