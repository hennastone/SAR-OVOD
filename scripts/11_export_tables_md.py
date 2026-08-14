"""
outputs/tables/*.csv iceriklerini tek bir markdown dosyasina dokur.

Neden: outputs/ dizini (predictions, manifest, figurler) Claude Project
kapasitesini asiyor ve secici klasor bazinda calistigi icin tablolara
ulasilamiyor. notes/ ise kucuk. Bu script CSV'leri notes/tables.md icine
markdown tablolari olarak yaziyor - boylece notes/ secildiginde tum sayilar
projede olur.

CSV'ler kaynak (makine tarafindan okunabilir), bu dosya turev. Tablolar
degisirse yeniden calistir.

Girdi : outputs/tables/*.csv
Cikti : notes/tables.md
"""
import argparse
from pathlib import Path

import pandas as pd

# (dosya, baslik, aciklama). Sirali.
SECTIONS = [
    ("dataset_stats.csv", "Veri seti istatistikleri",
     "Sinif basina ornek sayisi, o sinifi iceren goruntu sayisi ve goruntu basina "
     "ortalama nesne. Kaynak: `data/annotations/instances_*.json`."),
    ("size_distribution.csv", "Boyut bandi x sinif dagilimi",
     "Bant tanimi `sqrt(w*h)` (esdeger kare kenari), piksel."),
    ("results_closed_set.csv", "Baseline A - kapali kume (YOLO11s fine-tune @640)",
     "AP/AR tum guven araliginda; precision/recall `conf>=0.25 & IoU>=0.5`'te. "
     "`TP = n_gt - FN` ozdesliginden turetildi. FP'nin bandi/sinifi TESPIT "
     "kutusuna, FN'ninki GT kutusuna gore."),
    ("results_openvocab.csv", "Baseline B - acik kelime dagarcikli (YOLO-World v2-s @640, zero-shot)",
     "Ayni kirilim, ek olarak `query_set` sutunu (canonical / attributed)."),
    ("errors_summary.csv", "Hata ozeti: model x bant x hata tipi",
     "FP alt tipleri TIDE tarzi. FN'de `mean_confidence` bos - tespit olmadigi "
     "icin guven skoru tanimsiz."),
    ("calibration.csv", "Guven skoru kalibrasyonu",
     "0.1'lik kovalar. `fraction_correct` = o kovadaki tespitlerin TP orani. "
     "Kalibrasyon eslestirmesi `conf>=0.05` ile yapildi, bu yuzden ilk kova "
     "yalnizca 0.05-0.10 araligini kapsar."),
    ("timing.csv", "Hiz (KISMI - bkz. not)",
     "**`latency_ms_batch1`, `fps_batch1`, `warmup_excluded` OLCULMEDI** (adim 5 "
     "kosulmadi). Dolu sutunlar tahmin kosusunun yan urunu: chunk=16, batch=1 "
     "degil, isinma turlari haric tutulmadi. Karsilastirmali hiz iddiasi icin "
     "kullanilamaz."),
]

# Bu ikisi ham (kirpinti basina bir satir) - ozetlenerek yazilir
TAXONOMY = [
    ("fp_visual_taxonomy_sample.csv", "fp_background_index.csv",
     "Yanlis pozitif gorsel taksonomi (ornek)",
     "`background` alt tipinden incelenen ornek. Kategoriler onceden "
     "belirlenmedi. Orneklem yanli: kirpintilar guven skoruna gore azalan "
     "siralanip klasor basina ilk 60 kaydedildi, ben de hepsine bakmadim. "
     "Ham etiketler: `outputs/tables/fp_visual_taxonomy_sample.csv`."),
    ("fn_visual_taxonomy_sample.csv", "fn_swimmer_index.csv",
     "Kacirilan swimmer'lar gorsel taksonomi (ornek)",
     "Orneklem yanli: kirpintilar boyuta gore azalan siralanip ilk 60 "
     "kaydedildi; ben de iki uctan (en buyukler + en kucukler) baktim, "
     "orta boy temsil edilmiyor."),
]


def md_table(df: pd.DataFrame) -> str:
    df = df.fillna("")
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def taxonomy_summary(tables: Path, sheets: Path, label_csv: str, index_csv: str) -> pd.DataFrame:
    lab = pd.read_csv(tables / label_csv)
    idx = pd.read_csv(sheets / index_csv)
    df = lab.merge(idx, on="idx")
    agg = {"n": ("idx", "size"), "mean_edge_px": ("edge_px", "mean")}
    if df["conf"].notna().any():
        agg["mean_conf"] = ("conf", "mean")
    g = df.groupby("visual_category").agg(**agg).round(2)
    g["pct_of_sample"] = (100 * g["n"] / len(df)).round(1)
    return g.sort_values("n", ascending=False).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", type=Path, default=Path("outputs/tables"))
    p.add_argument("--sheets", type=Path,
                   default=Path("outputs/error_analysis/baseline_a_pilot/contact_sheets"))
    p.add_argument("--out", type=Path, default=Path("notes/tables.md"))
    args = p.parse_args()

    parts = [
        "# Sonuc tablolari",
        "",
        "`outputs/tables/*.csv` dosyalarinin markdown hali. **Kaynak CSV'lerdir**; "
        "bu dosya `scripts/11_export_tables_md.py` ile uretilir, elle duzenlenmez.",
        "",
        "Tum sonuclar **640 pilot kosusuna** ait (10 epoch). 1280 tam kosu "
        "yapilmadi. Degerlendirme **val** bolumunde (1547 goruntu, 9630 nesne); "
        "SeaDronesSee test etiketleri degerlendirme sunucusunda oldugu icin test "
        "sonucu uretilmedi.",
        "",
        "Yorumlar ve baglam: [findings.md](findings.md) · "
        "veri seti detaylari: [data-notes.md](data-notes.md) · "
        "gorsel hata analizi: [error-taxonomy.md](error-taxonomy.md)",
        "",
    ]

    for fname, title, desc in SECTIONS:
        path = args.tables / fname
        if not path.exists():
            parts += [f"## {title}", "", f"_`{fname}` bulunamadi._", ""]
            continue
        parts += [f"## {title}", "", desc, "", f"`outputs/tables/{fname}`", "",
                  md_table(pd.read_csv(path)), ""]

    for label_csv, index_csv, title, desc in TAXONOMY:
        if not (args.tables / label_csv).exists() or not (args.sheets / index_csv).exists():
            continue
        g = taxonomy_summary(args.tables, args.sheets, label_csv, index_csv)
        n = int(g["n"].sum())
        parts += [f"## {title}", "", desc, "", f"Incelenen ornek: **n = {n}**", "",
                  md_table(g), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    kb = args.out.stat().st_size / 1024
    print(f"{args.out}  ({kb:.1f} KB)")


if __name__ == "__main__":
    main()
