# Sonuc tablolari

`outputs/tables/*.csv` dosyalarinin markdown hali. **Kaynak CSV'lerdir**; bu dosya `scripts/11_export_tables_md.py` ile uretilir, elle duzenlenmez.

Tum sonuclar **640 pilot kosusuna** ait (10 epoch). 1280 tam kosu yapilmadi. Degerlendirme **val** bolumunde (1547 goruntu, 9630 nesne); SeaDronesSee test etiketleri degerlendirme sunucusunda oldugu icin test sonucu uretilmedi.

Yorumlar ve baglam: [findings.md](findings.md) · veri seti detaylari: [data-notes.md](data-notes.md) · gorsel hata analizi: [error-taxonomy.md](error-taxonomy.md)

## Veri seti istatistikleri

Sinif basina ornek sayisi, o sinifi iceren goruntu sayisi ve goruntu basina ortalama nesne. Kaynak: `data/annotations/instances_*.json`.

`outputs/tables/dataset_stats.csv`

| split | category | n_instances | n_images_containing | pct_images_containing | mean_instances_per_containing_image | pct_of_split_instances |
|---|---|---|---|---|---|---|
| train | swimmer | 37096 | 6898 | 77.25 | 5.378 | 64.22 |
| train | boat | 13022 | 5870 | 65.73 | 2.218 | 22.55 |
| train | buoy | 4389 | 3722 | 41.68 | 1.179 | 7.6 |
| train | jetski | 2330 | 2328 | 26.07 | 1.001 | 4.03 |
| train | life_saving_appliances | 923 | 665 | 7.45 | 1.388 | 1.6 |
| train | ALL | 57760 | 8930 | 100.0 | 6.468 | 100.0 |
| val | swimmer | 6206 | 1286 | 83.13 | 4.826 | 64.44 |
| val | boat | 2214 | 915 | 59.15 | 2.42 | 22.99 |
| val | buoy | 560 | 352 | 22.75 | 1.591 | 5.82 |
| val | life_saving_appliances | 330 | 191 | 12.35 | 1.728 | 3.43 |
| val | jetski | 320 | 320 | 20.69 | 1.0 | 3.32 |
| val | ALL | 9630 | 1547 | 100.0 | 6.225 | 100.0 |

## Boyut bandi x sinif dagilimi

Bant tanimi `sqrt(w*h)` (esdeger kare kenari), piksel.

`outputs/tables/size_distribution.csv`

| split | category | size_band | n_instances | pct_of_split |
|---|---|---|---|---|
| train | boat | <16 | 27 | 0.047 |
| train | boat | 16-32 | 1313 | 2.273 |
| train | boat | 32-64 | 2750 | 4.761 |
| train | boat | >64 | 8932 | 15.464 |
| train | buoy | <16 | 352 | 0.609 |
| train | buoy | 16-32 | 897 | 1.553 |
| train | buoy | 32-64 | 1818 | 3.148 |
| train | buoy | >64 | 1322 | 2.289 |
| train | jetski | <16 | 21 | 0.036 |
| train | jetski | 16-32 | 260 | 0.45 |
| train | jetski | 32-64 | 865 | 1.498 |
| train | jetski | >64 | 1184 | 2.05 |
| train | life_saving_appliances | <16 | 77 | 0.133 |
| train | life_saving_appliances | 16-32 | 550 | 0.952 |
| train | life_saving_appliances | 32-64 | 291 | 0.504 |
| train | life_saving_appliances | >64 | 5 | 0.009 |
| train | swimmer | <16 | 2707 | 4.687 |
| train | swimmer | 16-32 | 15567 | 26.951 |
| train | swimmer | 32-64 | 14587 | 25.255 |
| train | swimmer | >64 | 4235 | 7.332 |
| val | boat | <16 | 1 | 0.01 |
| val | boat | 16-32 | 165 | 1.713 |
| val | boat | 32-64 | 395 | 4.102 |
| val | boat | >64 | 1653 | 17.165 |
| val | buoy | <16 | 55 | 0.571 |
| val | buoy | 16-32 | 251 | 2.606 |
| val | buoy | 32-64 | 81 | 0.841 |
| val | buoy | >64 | 173 | 1.796 |
| val | jetski | <16 | 0 | 0.0 |
| val | jetski | 16-32 | 8 | 0.083 |
| val | jetski | 32-64 | 84 | 0.872 |
| val | jetski | >64 | 228 | 2.368 |
| val | life_saving_appliances | <16 | 5 | 0.052 |
| val | life_saving_appliances | 16-32 | 231 | 2.399 |
| val | life_saving_appliances | 32-64 | 94 | 0.976 |
| val | life_saving_appliances | >64 | 0 | 0.0 |
| val | swimmer | <16 | 121 | 1.256 |
| val | swimmer | 16-32 | 2129 | 22.108 |
| val | swimmer | 32-64 | 3197 | 33.198 |
| val | swimmer | >64 | 759 | 7.882 |

## Baseline A - kapali kume (YOLO11s fine-tune @640)

AP/AR tum guven araliginda; precision/recall `conf>=0.25 & IoU>=0.5`'te. `TP = n_gt - FN` ozdesliginden turetildi. FP'nin bandi/sinifi TESPIT kutusuna, FN'ninki GT kutusuna gore.

`outputs/tables/results_closed_set.csv`

| model | category | size_band | n_gt | FN | FP | AP@50-95 | AP@50 | AR@50-95 | TP | precision@conf0.25 | recall@conf0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO11s fine-tune | boat | 16-32 | 165 | 69 | 25 | 0.3925 | 0.6291 | 0.4188 | 96 | 0.7934 | 0.5818 |
| YOLO11s fine-tune | boat | 32-64 | 395 | 62 | 85 | 0.5004 | 0.8674 | 0.5633 | 333 | 0.7967 | 0.843 |
| YOLO11s fine-tune | boat | >64 | 1653 | 31 | 82 | 0.6926 | 0.9713 | 0.7555 | 1622 | 0.9519 | 0.9812 |
| YOLO11s fine-tune | buoy | <16 | 55 | 53 | 0 | 0.0101 | 0.0663 | 0.0164 | 2 | 1.0 | 0.0364 |
| YOLO11s fine-tune | buoy | 16-32 | 251 | 168 | 120 | 0.0947 | 0.2957 | 0.1239 | 83 | 0.4089 | 0.3307 |
| YOLO11s fine-tune | buoy | 32-64 | 81 | 8 | 5 | 0.6156 | 0.9076 | 0.6815 | 73 | 0.9359 | 0.9012 |
| YOLO11s fine-tune | buoy | >64 | 173 | 4 | 9 | 0.7163 | 0.9621 | 0.7867 | 169 | 0.9494 | 0.9769 |
| YOLO11s fine-tune | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO11s fine-tune | jetski | 16-32 | 8 | 0 | 6 | 0.4904 | 0.9571 | 0.5125 | 8 | 0.5714 | 1.0 |
| YOLO11s fine-tune | jetski | 32-64 | 84 | 12 | 20 | 0.4861 | 0.8505 | 0.556 | 72 | 0.7826 | 0.8571 |
| YOLO11s fine-tune | jetski | >64 | 228 | 12 | 64 | 0.5606 | 0.8831 | 0.6719 | 216 | 0.7714 | 0.9474 |
| YOLO11s fine-tune | life_saving_appliances | <16 | 5 | 3 | 0 | 0.2446 | 0.4059 | 0.24 | 2 | 1.0 | 0.4 |
| YOLO11s fine-tune | life_saving_appliances | 16-32 | 231 | 179 | 16 | 0.0864 | 0.2243 | 0.1091 | 52 | 0.7647 | 0.2251 |
| YOLO11s fine-tune | life_saving_appliances | 32-64 | 94 | 80 | 25 | 0.0634 | 0.1238 | 0.0947 | 14 | 0.359 | 0.1489 |
| YOLO11s fine-tune | life_saving_appliances | >64 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO11s fine-tune | swimmer | <16 | 121 | 93 | 15 | 0.0812 | 0.2345 | 0.1065 | 28 | 0.6512 | 0.2314 |
| YOLO11s fine-tune | swimmer | 16-32 | 2129 | 804 | 398 | 0.2337 | 0.5985 | 0.2819 | 1325 | 0.769 | 0.6224 |
| YOLO11s fine-tune | swimmer | 32-64 | 3197 | 478 | 1275 | 0.3226 | 0.7745 | 0.4583 | 2719 | 0.6808 | 0.8505 |
| YOLO11s fine-tune | swimmer | >64 | 759 | 138 | 228 | 0.3409 | 0.751 | 0.5204 | 621 | 0.7314 | 0.8182 |

## Baseline B - acik kelime dagarcikli (YOLO-World v2-s @640, zero-shot)

Ayni kirilim, ek olarak `query_set` sutunu (canonical / attributed).

`outputs/tables/results_openvocab.csv`

| model | query_set | category | size_band | n_gt | FN | FP | AP@50-95 | AP@50 | AR@50-95 | TP | precision@conf0.25 | recall@conf0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World v2-s | canonical | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | boat | 16-32 | 165 | 164 | 2 | 0.0074 | 0.0256 | 0.0806 | 1 | 0.3333 | 0.0061 |
| YOLO-World v2-s | canonical | boat | 32-64 | 395 | 376 | 66 | 0.0558 | 0.1612 | 0.2501 | 19 | 0.2235 | 0.0481 |
| YOLO-World v2-s | canonical | boat | >64 | 1653 | 635 | 244 | 0.4231 | 0.694 | 0.5811 | 1018 | 0.8067 | 0.6158 |
| YOLO-World v2-s | canonical | buoy | <16 | 55 | 55 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | buoy | 16-32 | 251 | 248 | 61 | 0.0004 | 0.0023 | 0.0108 | 3 | 0.0469 | 0.012 |
| YOLO-World v2-s | canonical | buoy | 32-64 | 81 | 78 | 114 | 0.0252 | 0.0557 | 0.4852 | 3 | 0.0256 | 0.037 |
| YOLO-World v2-s | canonical | buoy | >64 | 173 | 160 | 82 | 0.1105 | 0.1848 | 0.615 | 13 | 0.1368 | 0.0751 |
| YOLO-World v2-s | canonical | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | canonical | jetski | 16-32 | 8 | 8 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | jetski | 32-64 | 84 | 84 | 0 | 0.0004 | 0.0019 | 0.0238 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | jetski | >64 | 228 | 228 | 15 | 0.0041 | 0.0081 | 0.0877 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | canonical | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | life_saving_appliances | 16-32 | 231 | 231 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | life_saving_appliances | 32-64 | 94 | 94 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | life_saving_appliances | >64 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | canonical | swimmer | <16 | 121 | 121 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | swimmer | 16-32 | 2129 | 2129 | 0 | 0.0082 | 0.0188 | 0.0072 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | swimmer | 32-64 | 3197 | 3197 | 0 | 0.0246 | 0.0648 | 0.0327 | 0 |  | 0.0 |
| YOLO-World v2-s | canonical | swimmer | >64 | 759 | 759 | 0 | 0.1025 | 0.2673 | 0.1876 | 0 |  | 0.0 |
| YOLO-World v2-s | attributed | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | attributed | boat | 16-32 | 165 | 165 | 59 | 0.0033 | 0.0171 | 0.0394 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | attributed | boat | 32-64 | 395 | 276 | 239 | 0.0975 | 0.2093 | 0.2286 | 119 | 0.3324 | 0.3013 |
| YOLO-World v2-s | attributed | boat | >64 | 1653 | 467 | 1277 | 0.3535 | 0.6293 | 0.566 | 1186 | 0.4815 | 0.7175 |
| YOLO-World v2-s | attributed | buoy | <16 | 55 | 55 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | attributed | buoy | 16-32 | 251 | 247 | 96 | 0.0005 | 0.0033 | 0.01 | 4 | 0.04 | 0.0159 |
| YOLO-World v2-s | attributed | buoy | 32-64 | 81 | 73 | 497 | 0.006 | 0.0164 | 0.2827 | 8 | 0.0158 | 0.0988 |
| YOLO-World v2-s | attributed | buoy | >64 | 173 | 173 | 102 | 0.0021 | 0.0045 | 0.0636 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | attributed | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | attributed | jetski | 16-32 | 8 | 8 | 50 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | attributed | jetski | 32-64 | 84 | 61 | 392 | 0.011 | 0.0306 | 0.2381 | 23 | 0.0554 | 0.2738 |
| YOLO-World v2-s | attributed | jetski | >64 | 228 | 188 | 477 | 0.0167 | 0.0325 | 0.2965 | 40 | 0.0774 | 0.1754 |
| YOLO-World v2-s | attributed | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | attributed | life_saving_appliances | 16-32 | 231 | 231 | 3 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | attributed | life_saving_appliances | 32-64 | 94 | 94 | 4 | 0.0003 | 0.001 | 0.0032 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | attributed | life_saving_appliances | >64 | 0 | 0 | 2 |  |  |  | 0 | 0.0 |  |
| YOLO-World v2-s | attributed | swimmer | <16 | 121 | 121 | 0 | 0.0032 | 0.0218 | 0.004 | 0 |  | 0.0 |
| YOLO-World v2-s | attributed | swimmer | 16-32 | 2129 | 1863 | 90 | 0.0681 | 0.1784 | 0.1022 | 266 | 0.7472 | 0.1249 |
| YOLO-World v2-s | attributed | swimmer | 32-64 | 3197 | 2496 | 339 | 0.1025 | 0.2595 | 0.1773 | 701 | 0.674 | 0.2193 |
| YOLO-World v2-s | attributed | swimmer | >64 | 759 | 413 | 919 | 0.129 | 0.3001 | 0.4034 | 346 | 0.2735 | 0.4559 |

## Hata ozeti: model x bant x hata tipi

FP alt tipleri TIDE tarzi. FN'de `mean_confidence` bos - tespit olmadigi icin guven skoru tanimsiz.

`outputs/tables/errors_summary.csv`

| model | query_set | tag | size_band | error_type | fp_subtype | count | mean_confidence | mean_edge_px |
|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FN |  | 150 |  | 14.06 |
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FP | background | 9 | 0.3667 | 11.68 |
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FP | class_confusion | 1 | 0.3589 | 15.9 |
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FP | localization | 1 | 0.3839 | 14.6 |
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FP | duplicate | 4 | 0.3232 | 12.05 |
| YOLO11s fine-tune |  | baseline_a_pilot | <16 | FP | ALL | 15 | 0.3557 | 12.25 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FN |  | 1220 |  | 23.99 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FP | background | 90 | 0.3882 | 25.04 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FP | class_confusion | 22 | 0.3768 | 26.39 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FP | localization | 411 | 0.4099 | 27.52 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FP | duplicate | 42 | 0.3223 | 27.7 |
| YOLO11s fine-tune |  | baseline_a_pilot | 16-32 | FP | ALL | 565 | 0.3986 | 27.1 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FN |  | 640 |  | 41.97 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FP | background | 145 | 0.439 | 43.03 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FP | class_confusion | 53 | 0.5433 | 43.52 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FP | localization | 959 | 0.4804 | 43.71 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FP | duplicate | 253 | 0.3367 | 43.5 |
| YOLO11s fine-tune |  | baseline_a_pilot | 32-64 | FP | ALL | 1410 | 0.4527 | 43.59 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FN |  | 185 |  | 158.07 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FP | background | 33 | 0.5545 | 168.86 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FP | class_confusion | 61 | 0.6408 | 334.71 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FP | localization | 224 | 0.5458 | 124.34 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FP | duplicate | 65 | 0.378 | 161.47 |
| YOLO11s fine-tune |  | baseline_a_pilot | >64 | FP | ALL | 383 | 0.5332 | 167.98 |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FN |  | 182 |  | 13.97 |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FP | background | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FP | class_confusion | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FP | localization | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | <16 | FP | ALL | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FN |  | 2780 |  | 25.57 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FP | background | 6 | 0.2952 | 28.52 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FP | class_confusion | 56 | 0.3426 | 26.22 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FP | localization | 1 | 0.4886 | 31.39 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | 16-32 | FP | ALL | 63 | 0.3404 | 26.52 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FN |  | 3829 |  | 43.61 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FP | background | 57 | 0.3286 | 51.56 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FP | class_confusion | 122 | 0.3365 | 44.5 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FP | localization | 1 | 0.2611 | 37.69 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical | 32-64 | FP | ALL | 180 | 0.3336 | 46.69 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FN |  | 1782 |  | 132.26 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FP | background | 61 | 0.3536 | 116.4 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FP | class_confusion | 200 | 0.427 | 148.09 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FP | localization | 65 | 0.4741 | 130.63 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FP | duplicate | 15 | 0.3101 | 142.17 |
| YOLO-World v2-s | canonical | baseline_b_canonical | >64 | FP | ALL | 341 | 0.4177 | 138.83 |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FN |  | 182 |  | 13.97 |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FP | background | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FP | class_confusion | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FP | localization | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed | <16 | FP | ALL | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FN |  | 2514 |  | 25.28 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FP | background | 65 | 0.402 | 28.52 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FP | class_confusion | 154 | 0.4354 | 28.11 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FP | localization | 60 | 0.4456 | 27.96 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FP | duplicate | 19 | 0.4473 | 28.95 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 16-32 | FP | ALL | 298 | 0.4309 | 28.22 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FN |  | 3000 |  | 43.47 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FP | background | 329 | 0.4326 | 46.46 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FP | class_confusion | 844 | 0.4907 | 43.64 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FP | localization | 197 | 0.5119 | 47.12 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FP | duplicate | 101 | 0.5017 | 43.44 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 32-64 | FP | ALL | 1471 | 0.4813 | 44.72 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FN |  | 1241 |  | 116.03 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FP | background | 729 | 0.4742 | 202.31 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FP | class_confusion | 1143 | 0.6278 | 138.31 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FP | localization | 622 | 0.5675 | 149.37 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FP | duplicate | 283 | 0.5503 | 152.5 |
| YOLO-World v2-s | attributed | baseline_b_attributed | >64 | FP | ALL | 2777 | 0.5661 | 159.03 |

## Guven skoru kalibrasyonu

0.1'lik kovalar. `fraction_correct` = o kovadaki tespitlerin TP orani. Kalibrasyon eslestirmesi `conf>=0.05` ile yapildi, bu yuzden ilk kova yalnizca 0.05-0.10 araligini kapsar.

`outputs/tables/calibration.csv`

| model | query_set | tag | bin_low | bin_high | n_detections | mean_confidence | fraction_correct |
|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune |  | baseline_a_pilot | 0.0 | 0.1 | 2105 | 0.071 | 0.0276 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.1 | 0.2 | 1775 | 0.1444 | 0.0518 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.2 | 0.3 | 1012 | 0.2468 | 0.1166 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.3 | 0.4 | 939 | 0.3508 | 0.2939 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.4 | 0.5 | 1099 | 0.4537 | 0.5851 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.5 | 0.6000000000000001 | 1717 | 0.5523 | 0.7676 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.6000000000000001 | 0.7000000000000001 | 2233 | 0.6517 | 0.8607 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.7000000000000001 | 0.8 | 1639 | 0.7443 | 0.9378 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.8 | 0.9 | 1639 | 0.848 | 0.975 |
| YOLO11s fine-tune |  | baseline_a_pilot | 0.9 | 1.0 | 75 | 0.9132 | 1.0 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.0 | 0.1 | 1917 | 0.0715 | 0.1007 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.1 | 0.2 | 1624 | 0.142 | 0.1755 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.2 | 0.3 | 649 | 0.2447 | 0.2388 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.3 | 0.4 | 330 | 0.3451 | 0.3667 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.4 | 0.5 | 218 | 0.445 | 0.6193 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.5 | 0.6000000000000001 | 174 | 0.5505 | 0.7069 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.6000000000000001 | 0.7000000000000001 | 272 | 0.6548 | 0.9375 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.7000000000000001 | 0.8 | 259 | 0.7425 | 0.9846 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.8 | 0.9 | 114 | 0.8396 | 0.9035 |
| YOLO-World v2-s | canonical | baseline_b_canonical | 0.9 | 1.0 | 14 | 0.9216 | 0.2143 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.0 | 0.1 | 4039 | 0.0709 | 0.0587 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.1 | 0.2 | 3320 | 0.1431 | 0.0985 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.2 | 0.3 | 1665 | 0.2471 | 0.1393 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.3 | 0.4 | 1137 | 0.3495 | 0.1917 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.4 | 0.5 | 937 | 0.4484 | 0.2134 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.5 | 0.6000000000000001 | 857 | 0.5507 | 0.2812 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.6000000000000001 | 0.7000000000000001 | 825 | 0.6487 | 0.3285 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.7000000000000001 | 0.8 | 780 | 0.7499 | 0.3769 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.8 | 0.9 | 805 | 0.8512 | 0.5615 |
| YOLO-World v2-s | attributed | baseline_b_attributed | 0.9 | 1.0 | 1133 | 0.9498 | 0.7917 |

## Hiz (KISMI - bkz. not)

**`latency_ms_batch1`, `fps_batch1`, `warmup_excluded` OLCULMEDI** (adim 5 kosulmadi). Dolu sutunlar tahmin kosusunun yan urunu: chunk=16, batch=1 degil, isinma turlari haric tutulmadi. Karsilastirmali hiz iddiasi icin kullanilamaz.

`outputs/tables/timing.csv`

| model | query_set | tag | imgsz | hardware | batch_size | preprocess_ms | inference_ms | postprocess_ms | total_ms_per_image | throughput_img_per_s_wallclock | latency_ms_batch1 | fps_batch1 | warmup_excluded | measurement_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune |  | baseline_a_pilot | 640 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 1.07 | 1.64 | 1.45 | 4.16 | 16.58 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |
| YOLO-World v2-s | canonical | baseline_b_canonical | 640 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 1.07 | 1.87 | 1.55 | 4.49 | 16.65 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |
| YOLO-World v2-s | attributed | baseline_b_attributed | 640 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 1.07 | 1.88 | 1.58 | 4.53 | 16.61 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |

## Yanlis pozitif gorsel taksonomi (ornek)

`background` alt tipinden incelenen ornek. Kategoriler onceden belirlenmedi. Orneklem yanli: kirpintilar guven skoruna gore azalan siralanip klasor basina ilk 60 kaydedildi, ben de hepsine bakmadim. Ham etiketler: `outputs/tables/fp_visual_taxonomy_sample.csv`.

Incelenen ornek: **n = 81**

| visual_category | n | mean_edge_px | mean_conf | pct_of_sample |
|---|---|---|---|---|
| belirsiz | 39 | 24.1 | 0.35 | 48.1 |
| gercek_anotasyonsuz_insan | 20 | 122.25 | 0.44 | 24.7 |
| gercek_anotasyonsuz_deniz_araci | 8 | 74.38 | 0.49 | 9.9 |
| gercek_anotasyonsuz_tekne | 7 | 62.86 | 0.72 | 8.6 |
| dalga_kopugu_parilti | 4 | 36.0 | 0.33 | 4.9 |
| gercek_anotasyonsuz_can_simidi | 3 | 32.33 | 0.67 | 3.7 |

## Kacirilan swimmer'lar gorsel taksonomi (ornek)

Orneklem yanli: kirpintilar boyuta gore azalan siralanip ilk 60 kaydedildi; ben de iki uctan (en buyukler + en kucukler) baktim, orta boy temsil edilmiyor.

Incelenen ornek: **n = 48**

| visual_category | n | mean_edge_px | pct_of_sample |
|---|---|---|---|
| sadece_bas | 18 | 14.0 | 37.5 |
| net_izole | 8 | 31.25 | 16.7 |
| dusuk_kontrast | 6 | 31.0 | 12.5 |
| dusuk_isik | 5 | 14.0 | 10.4 |
| kopuk_icinde | 5 | 27.8 | 10.4 |
| kume_halinde | 4 | 31.0 | 8.3 |
| batik_golge | 2 | 31.0 | 4.2 |
