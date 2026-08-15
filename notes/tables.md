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

| model | run | imgsz | category | size_band | n_gt | FN | FP | AP@50-95 | AP@50 | AR@50-95 | TP | precision@conf0.25 | recall@conf0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune | full@1280 | 1280 | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO11s fine-tune | full@1280 | 1280 | boat | 16-32 | 165 | 6 | 22 | 0.6637 | 0.9519 | 0.6982 | 159 | 0.8785 | 0.9636 |
| YOLO11s fine-tune | full@1280 | 1280 | boat | 32-64 | 395 | 42 | 21 | 0.6233 | 0.9049 | 0.6709 | 353 | 0.9439 | 0.8937 |
| YOLO11s fine-tune | full@1280 | 1280 | boat | >64 | 1653 | 24 | 58 | 0.784 | 0.9834 | 0.8331 | 1629 | 0.9656 | 0.9855 |
| YOLO11s fine-tune | full@1280 | 1280 | buoy | <16 | 55 | 4 | 15 | 0.375 | 0.9359 | 0.4636 | 51 | 0.7727 | 0.9273 |
| YOLO11s fine-tune | full@1280 | 1280 | buoy | 16-32 | 251 | 24 | 36 | 0.4174 | 0.8796 | 0.5104 | 227 | 0.8631 | 0.9044 |
| YOLO11s fine-tune | full@1280 | 1280 | buoy | 32-64 | 81 | 1 | 5 | 0.6717 | 0.9923 | 0.742 | 80 | 0.9412 | 0.9877 |
| YOLO11s fine-tune | full@1280 | 1280 | buoy | >64 | 173 | 3 | 6 | 0.7343 | 0.971 | 0.815 | 170 | 0.9659 | 0.9827 |
| YOLO11s fine-tune | full@1280 | 1280 | jetski | <16 | 0 | 0 | 2 |  |  |  | 0 | 0.0 |  |
| YOLO11s fine-tune | full@1280 | 1280 | jetski | 16-32 | 8 | 1 | 2 | 0.6518 | 0.9505 | 0.7 | 7 | 0.7778 | 0.875 |
| YOLO11s fine-tune | full@1280 | 1280 | jetski | 32-64 | 84 | 7 | 18 | 0.5994 | 0.8839 | 0.7071 | 77 | 0.8105 | 0.9167 |
| YOLO11s fine-tune | full@1280 | 1280 | jetski | >64 | 228 | 23 | 52 | 0.6135 | 0.8944 | 0.6851 | 205 | 0.7977 | 0.8991 |
| YOLO11s fine-tune | full@1280 | 1280 | life_saving_appliances | <16 | 5 | 2 | 1 | 0.2243 | 0.604 | 0.38 | 3 | 0.75 | 0.6 |
| YOLO11s fine-tune | full@1280 | 1280 | life_saving_appliances | 16-32 | 231 | 102 | 21 | 0.3278 | 0.5852 | 0.3619 | 129 | 0.86 | 0.5584 |
| YOLO11s fine-tune | full@1280 | 1280 | life_saving_appliances | 32-64 | 94 | 52 | 24 | 0.2756 | 0.4741 | 0.3436 | 42 | 0.6364 | 0.4468 |
| YOLO11s fine-tune | full@1280 | 1280 | life_saving_appliances | >64 | 0 | 0 | 2 |  |  |  | 0 | 0.0 |  |
| YOLO11s fine-tune | full@1280 | 1280 | swimmer | <16 | 121 | 29 | 31 | 0.2949 | 0.7749 | 0.3524 | 92 | 0.748 | 0.7603 |
| YOLO11s fine-tune | full@1280 | 1280 | swimmer | 16-32 | 2129 | 353 | 323 | 0.3423 | 0.8004 | 0.4414 | 1776 | 0.8461 | 0.8342 |
| YOLO11s fine-tune | full@1280 | 1280 | swimmer | 32-64 | 3197 | 342 | 497 | 0.4172 | 0.8709 | 0.5313 | 2855 | 0.8517 | 0.893 |
| YOLO11s fine-tune | full@1280 | 1280 | swimmer | >64 | 759 | 148 | 150 | 0.4014 | 0.7903 | 0.5291 | 611 | 0.8029 | 0.805 |
| YOLO11s fine-tune | pilot@640 | 640 | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO11s fine-tune | pilot@640 | 640 | boat | 16-32 | 165 | 69 | 25 | 0.3925 | 0.6291 | 0.4188 | 96 | 0.7934 | 0.5818 |
| YOLO11s fine-tune | pilot@640 | 640 | boat | 32-64 | 395 | 62 | 85 | 0.5004 | 0.8674 | 0.5633 | 333 | 0.7967 | 0.843 |
| YOLO11s fine-tune | pilot@640 | 640 | boat | >64 | 1653 | 31 | 82 | 0.6926 | 0.9713 | 0.7555 | 1622 | 0.9519 | 0.9812 |
| YOLO11s fine-tune | pilot@640 | 640 | buoy | <16 | 55 | 53 | 0 | 0.0101 | 0.0663 | 0.0164 | 2 | 1.0 | 0.0364 |
| YOLO11s fine-tune | pilot@640 | 640 | buoy | 16-32 | 251 | 168 | 120 | 0.0947 | 0.2957 | 0.1239 | 83 | 0.4089 | 0.3307 |
| YOLO11s fine-tune | pilot@640 | 640 | buoy | 32-64 | 81 | 8 | 5 | 0.6156 | 0.9076 | 0.6815 | 73 | 0.9359 | 0.9012 |
| YOLO11s fine-tune | pilot@640 | 640 | buoy | >64 | 173 | 4 | 9 | 0.7163 | 0.9621 | 0.7867 | 169 | 0.9494 | 0.9769 |
| YOLO11s fine-tune | pilot@640 | 640 | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO11s fine-tune | pilot@640 | 640 | jetski | 16-32 | 8 | 0 | 6 | 0.4904 | 0.9571 | 0.5125 | 8 | 0.5714 | 1.0 |
| YOLO11s fine-tune | pilot@640 | 640 | jetski | 32-64 | 84 | 12 | 20 | 0.4861 | 0.8505 | 0.556 | 72 | 0.7826 | 0.8571 |
| YOLO11s fine-tune | pilot@640 | 640 | jetski | >64 | 228 | 12 | 64 | 0.5606 | 0.8831 | 0.6719 | 216 | 0.7714 | 0.9474 |
| YOLO11s fine-tune | pilot@640 | 640 | life_saving_appliances | <16 | 5 | 3 | 0 | 0.2446 | 0.4059 | 0.24 | 2 | 1.0 | 0.4 |
| YOLO11s fine-tune | pilot@640 | 640 | life_saving_appliances | 16-32 | 231 | 179 | 16 | 0.0864 | 0.2243 | 0.1091 | 52 | 0.7647 | 0.2251 |
| YOLO11s fine-tune | pilot@640 | 640 | life_saving_appliances | 32-64 | 94 | 80 | 25 | 0.0634 | 0.1238 | 0.0947 | 14 | 0.359 | 0.1489 |
| YOLO11s fine-tune | pilot@640 | 640 | life_saving_appliances | >64 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO11s fine-tune | pilot@640 | 640 | swimmer | <16 | 121 | 93 | 15 | 0.0812 | 0.2345 | 0.1065 | 28 | 0.6512 | 0.2314 |
| YOLO11s fine-tune | pilot@640 | 640 | swimmer | 16-32 | 2129 | 804 | 398 | 0.2337 | 0.5985 | 0.2819 | 1325 | 0.769 | 0.6224 |
| YOLO11s fine-tune | pilot@640 | 640 | swimmer | 32-64 | 3197 | 478 | 1275 | 0.3226 | 0.7745 | 0.4583 | 2719 | 0.6808 | 0.8505 |
| YOLO11s fine-tune | pilot@640 | 640 | swimmer | >64 | 759 | 138 | 228 | 0.3409 | 0.751 | 0.5204 | 621 | 0.7314 | 0.8182 |

## Baseline B - acik kelime dagarcikli (YOLO-World v2-s @640, zero-shot)

Ayni kirilim, ek olarak `query_set` sutunu (canonical / attributed).

`outputs/tables/results_openvocab.csv`

| model | run | imgsz | query_set | category | size_band | n_gt | FN | FP | AP@50-95 | AP@50 | AR@50-95 | TP | precision@conf0.25 | recall@conf0.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World v2-s | full@1280 | 1280 | canonical | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | boat | 16-32 | 165 | 158 | 88 | 0.008 | 0.0202 | 0.2345 | 7 | 0.0737 | 0.0424 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | boat | 32-64 | 395 | 162 | 169 | 0.1934 | 0.534 | 0.4058 | 233 | 0.5796 | 0.5899 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | boat | >64 | 1653 | 481 | 323 | 0.53 | 0.7424 | 0.725 | 1172 | 0.7839 | 0.709 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | buoy | <16 | 55 | 54 | 0 | 0.0063 | 0.021 | 0.0291 | 1 | 1.0 | 0.0182 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | buoy | 16-32 | 251 | 224 | 172 | 0.0167 | 0.0551 | 0.2223 | 27 | 0.1357 | 0.1076 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | buoy | 32-64 | 81 | 79 | 78 | 0.0293 | 0.046 | 0.6444 | 2 | 0.025 | 0.0247 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | buoy | >64 | 173 | 122 | 72 | 0.282 | 0.3796 | 0.7954 | 51 | 0.4146 | 0.2948 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | full@1280 | 1280 | canonical | jetski | 16-32 | 8 | 8 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | jetski | 32-64 | 84 | 84 | 2 | 0.0038 | 0.0108 | 0.0988 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | jetski | >64 | 228 | 228 | 32 | 0.0082 | 0.0179 | 0.1496 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | life_saving_appliances | 16-32 | 231 | 231 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | life_saving_appliances | 32-64 | 94 | 94 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | life_saving_appliances | >64 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | full@1280 | 1280 | canonical | swimmer | <16 | 121 | 121 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | swimmer | 16-32 | 2129 | 2129 | 0 | 0.036 | 0.0888 | 0.0476 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | swimmer | 32-64 | 3197 | 3197 | 0 | 0.0693 | 0.1739 | 0.1088 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | canonical | swimmer | >64 | 759 | 759 | 0 | 0.1526 | 0.3538 | 0.2664 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | boat | <16 | 1 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | boat | 16-32 | 165 | 152 | 343 | 0.003 | 0.0096 | 0.1109 | 13 | 0.0365 | 0.0788 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | boat | 32-64 | 395 | 131 | 1362 | 0.1436 | 0.3756 | 0.3937 | 264 | 0.1624 | 0.6684 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | boat | >64 | 1653 | 356 | 2174 | 0.4521 | 0.6505 | 0.7287 | 1297 | 0.3737 | 0.7846 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | buoy | <16 | 55 | 55 | 3 | 0.0028 | 0.0138 | 0.0509 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | buoy | 16-32 | 251 | 192 | 311 | 0.0274 | 0.092 | 0.2394 | 59 | 0.1595 | 0.2351 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | buoy | 32-64 | 81 | 81 | 126 | 0.0139 | 0.0217 | 0.3815 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | buoy | >64 | 173 | 172 | 17 | 0.0092 | 0.0129 | 0.1202 | 1 | 0.0556 | 0.0058 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | jetski | <16 | 0 | 0 | 3 |  |  |  | 0 | 0.0 |  |
| YOLO-World v2-s | full@1280 | 1280 | attributed | jetski | 16-32 | 8 | 6 | 208 | 0.0063 | 0.0117 | 0.525 | 2 | 0.0095 | 0.25 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | jetski | 32-64 | 84 | 52 | 646 | 0.0267 | 0.0667 | 0.3512 | 32 | 0.0472 | 0.381 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | jetski | >64 | 228 | 203 | 482 | 0.0183 | 0.037 | 0.3899 | 25 | 0.0493 | 0.1096 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0058 | 0.0083 | 0.14 | 0 |  | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | life_saving_appliances | 16-32 | 231 | 231 | 8 | 0.0 | 0.0001 | 0.0004 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | life_saving_appliances | 32-64 | 94 | 93 | 3 | 0.0109 | 0.0339 | 0.0723 | 1 | 0.25 | 0.0106 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | life_saving_appliances | >64 | 0 | 0 | 3 |  |  |  | 0 | 0.0 |  |
| YOLO-World v2-s | full@1280 | 1280 | attributed | swimmer | <16 | 121 | 120 | 12 | 0.0029 | 0.0107 | 0.0452 | 1 | 0.0769 | 0.0083 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | swimmer | 16-32 | 2129 | 1596 | 609 | 0.1017 | 0.267 | 0.2257 | 533 | 0.4667 | 0.2504 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | swimmer | 32-64 | 3197 | 1950 | 989 | 0.1765 | 0.4224 | 0.3593 | 1247 | 0.5577 | 0.3901 |
| YOLO-World v2-s | full@1280 | 1280 | attributed | swimmer | >64 | 759 | 290 | 947 | 0.1605 | 0.3643 | 0.4897 | 469 | 0.3312 | 0.6179 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | boat | 16-32 | 165 | 164 | 2 | 0.0074 | 0.0256 | 0.0806 | 1 | 0.3333 | 0.0061 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | boat | 32-64 | 395 | 376 | 66 | 0.0558 | 0.1612 | 0.2501 | 19 | 0.2235 | 0.0481 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | boat | >64 | 1653 | 635 | 244 | 0.4231 | 0.694 | 0.5811 | 1018 | 0.8067 | 0.6158 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | buoy | <16 | 55 | 55 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | buoy | 16-32 | 251 | 248 | 61 | 0.0004 | 0.0023 | 0.0108 | 3 | 0.0469 | 0.012 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | buoy | 32-64 | 81 | 78 | 114 | 0.0252 | 0.0557 | 0.4852 | 3 | 0.0256 | 0.037 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | buoy | >64 | 173 | 160 | 82 | 0.1105 | 0.1848 | 0.615 | 13 | 0.1368 | 0.0751 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | pilot@640 | 640 | canonical | jetski | 16-32 | 8 | 8 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | jetski | 32-64 | 84 | 84 | 0 | 0.0004 | 0.0019 | 0.0238 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | jetski | >64 | 228 | 228 | 15 | 0.0041 | 0.0081 | 0.0877 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | life_saving_appliances | 16-32 | 231 | 231 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | life_saving_appliances | 32-64 | 94 | 94 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | life_saving_appliances | >64 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | pilot@640 | 640 | canonical | swimmer | <16 | 121 | 121 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | swimmer | 16-32 | 2129 | 2129 | 0 | 0.0082 | 0.0188 | 0.0072 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | swimmer | 32-64 | 3197 | 3197 | 0 | 0.0246 | 0.0648 | 0.0327 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | canonical | swimmer | >64 | 759 | 759 | 0 | 0.1025 | 0.2673 | 0.1876 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | boat | <16 | 1 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | boat | 16-32 | 165 | 165 | 59 | 0.0033 | 0.0171 | 0.0394 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | boat | 32-64 | 395 | 276 | 239 | 0.0975 | 0.2093 | 0.2286 | 119 | 0.3324 | 0.3013 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | boat | >64 | 1653 | 467 | 1277 | 0.3535 | 0.6293 | 0.566 | 1186 | 0.4815 | 0.7175 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | buoy | <16 | 55 | 55 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | buoy | 16-32 | 251 | 247 | 96 | 0.0005 | 0.0033 | 0.01 | 4 | 0.04 | 0.0159 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | buoy | 32-64 | 81 | 73 | 497 | 0.006 | 0.0164 | 0.2827 | 8 | 0.0158 | 0.0988 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | buoy | >64 | 173 | 173 | 102 | 0.0021 | 0.0045 | 0.0636 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | jetski | <16 | 0 | 0 | 0 |  |  |  | 0 |  |  |
| YOLO-World v2-s | pilot@640 | 640 | attributed | jetski | 16-32 | 8 | 8 | 50 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | jetski | 32-64 | 84 | 61 | 392 | 0.011 | 0.0306 | 0.2381 | 23 | 0.0554 | 0.2738 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | jetski | >64 | 228 | 188 | 477 | 0.0167 | 0.0325 | 0.2965 | 40 | 0.0774 | 0.1754 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | life_saving_appliances | <16 | 5 | 5 | 0 | 0.0 | 0.0 | 0.0 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | life_saving_appliances | 16-32 | 231 | 231 | 3 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | life_saving_appliances | 32-64 | 94 | 94 | 4 | 0.0003 | 0.001 | 0.0032 | 0 | 0.0 | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | life_saving_appliances | >64 | 0 | 0 | 2 |  |  |  | 0 | 0.0 |  |
| YOLO-World v2-s | pilot@640 | 640 | attributed | swimmer | <16 | 121 | 121 | 0 | 0.0032 | 0.0218 | 0.004 | 0 |  | 0.0 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | swimmer | 16-32 | 2129 | 1863 | 90 | 0.0681 | 0.1784 | 0.1022 | 266 | 0.7472 | 0.1249 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | swimmer | 32-64 | 3197 | 2496 | 339 | 0.1025 | 0.2595 | 0.1773 | 701 | 0.674 | 0.2193 |
| YOLO-World v2-s | pilot@640 | 640 | attributed | swimmer | >64 | 759 | 413 | 919 | 0.129 | 0.3001 | 0.4034 | 346 | 0.2735 | 0.4559 |

## Hata ozeti: model x bant x hata tipi

FP alt tipleri TIDE tarzi. FN'de `mean_confidence` bos - tespit olmadigi icin guven skoru tanimsiz.

`outputs/tables/errors_summary.csv`

| model | query_set | tag | size_band | error_type | fp_subtype | count | mean_confidence | mean_edge_px |
|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune |  | baseline_a_full | <16 | FN |  | 36 |  | 14.26 |
| YOLO11s fine-tune |  | baseline_a_full | <16 | FP | background | 23 | 0.4331 | 13.8 |
| YOLO11s fine-tune |  | baseline_a_full | <16 | FP | class_confusion | 1 | 0.3928 | 13.65 |
| YOLO11s fine-tune |  | baseline_a_full | <16 | FP | localization | 10 | 0.4915 | 13.94 |
| YOLO11s fine-tune |  | baseline_a_full | <16 | FP | duplicate | 15 | 0.3958 | 14.98 |
| YOLO11s fine-tune |  | baseline_a_full | <16 | FP | ALL | 49 | 0.4328 | 14.19 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FN |  | 486 |  | 25.35 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FP | background | 103 | 0.4988 | 23.45 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FP | class_confusion | 9 | 0.4684 | 26.15 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FP | localization | 255 | 0.5329 | 26.25 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FP | duplicate | 37 | 0.3295 | 23.84 |
| YOLO11s fine-tune |  | baseline_a_full | 16-32 | FP | ALL | 404 | 0.5042 | 25.31 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FN |  | 444 |  | 43.67 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FP | background | 77 | 0.5215 | 45.14 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FP | class_confusion | 30 | 0.3984 | 39.98 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FP | localization | 421 | 0.5607 | 45.26 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FP | duplicate | 37 | 0.3358 | 46.4 |
| YOLO11s fine-tune |  | baseline_a_full | 32-64 | FP | ALL | 565 | 0.532 | 45.04 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FN |  | 198 |  | 201.31 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FP | background | 64 | 0.5246 | 178.2 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FP | class_confusion | 45 | 0.6208 | 402.91 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FP | localization | 147 | 0.6101 | 134.34 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FP | duplicate | 12 | 0.3913 | 140.14 |
| YOLO11s fine-tune |  | baseline_a_full | >64 | FP | ALL | 268 | 0.5817 | 190.17 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FN |  | 181 |  | 13.98 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FP | background | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FP | class_confusion | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FP | localization | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | <16 | FP | ALL | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FN |  | 2750 |  | 25.61 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FP | background | 74 | 0.3796 | 27.48 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FP | class_confusion | 168 | 0.3759 | 27.22 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FP | localization | 18 | 0.3842 | 22.89 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 16-32 | FP | ALL | 260 | 0.3776 | 27.0 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FN |  | 3616 |  | 43.64 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FP | background | 49 | 0.332 | 44.3 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FP | class_confusion | 191 | 0.3679 | 49.36 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FP | localization | 7 | 0.5222 | 42.44 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FP | duplicate | 2 | 0.2937 | 53.78 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 32-64 | FP | ALL | 249 | 0.3645 | 48.2 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FN |  | 1590 |  | 132.65 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FP | background | 49 | 0.4219 | 93.14 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FP | class_confusion | 289 | 0.5187 | 150.91 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FP | localization | 51 | 0.5504 | 133.02 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FP | duplicate | 38 | 0.34 | 116.24 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | >64 | FP | ALL | 427 | 0.4955 | 139.06 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FN |  | 181 |  | 13.97 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FP | background | 12 | 0.3849 | 13.88 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FP | class_confusion | 2 | 0.3957 | 15.07 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FP | localization | 5 | 0.3648 | 14.88 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FP | duplicate | 0 |  |  |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | <16 | FP | ALL | 19 | 0.3807 | 14.27 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FN |  | 2177 |  | 25.19 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FP | background | 329 | 0.4997 | 27.13 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FP | class_confusion | 642 | 0.5451 | 27.14 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FP | localization | 405 | 0.5555 | 25.35 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FP | duplicate | 103 | 0.5609 | 27.26 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 16-32 | FP | ALL | 1479 | 0.539 | 26.66 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FN |  | 2307 |  | 44.08 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FP | background | 1011 | 0.4771 | 46.54 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FP | class_confusion | 1211 | 0.5559 | 46.22 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FP | localization | 645 | 0.5124 | 48.67 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FP | duplicate | 259 | 0.4874 | 43.12 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 32-64 | FP | ALL | 3126 | 0.5158 | 46.57 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FN |  | 1021 |  | 119.11 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FP | background | 851 | 0.4952 | 250.11 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FP | class_confusion | 1149 | 0.6629 | 130.59 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FP | localization | 1116 | 0.5194 | 123.82 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FP | duplicate | 507 | 0.5186 | 128.92 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | >64 | FP | ALL | 3623 | 0.5591 | 156.34 |
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
| YOLO11s fine-tune |  | baseline_a_full | 0.0 | 0.1 | 1356 | 0.0705 | 0.0406 |
| YOLO11s fine-tune |  | baseline_a_full | 0.1 | 0.2 | 851 | 0.1385 | 0.0917 |
| YOLO11s fine-tune |  | baseline_a_full | 0.2 | 0.3 | 420 | 0.2449 | 0.2119 |
| YOLO11s fine-tune |  | baseline_a_full | 0.3 | 0.4 | 437 | 0.3534 | 0.4531 |
| YOLO11s fine-tune |  | baseline_a_full | 0.4 | 0.5 | 655 | 0.4528 | 0.6794 |
| YOLO11s fine-tune |  | baseline_a_full | 0.5 | 0.6000000000000001 | 1036 | 0.5549 | 0.7876 |
| YOLO11s fine-tune |  | baseline_a_full | 0.6000000000000001 | 0.7000000000000001 | 1835 | 0.6521 | 0.8708 |
| YOLO11s fine-tune |  | baseline_a_full | 0.7000000000000001 | 0.8 | 2776 | 0.7521 | 0.9226 |
| YOLO11s fine-tune |  | baseline_a_full | 0.8 | 0.9 | 2451 | 0.8529 | 0.9825 |
| YOLO11s fine-tune |  | baseline_a_full | 0.9 | 1.0 | 390 | 0.9107 | 0.9974 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.0 | 0.1 | 2717 | 0.0708 | 0.078 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.1 | 0.2 | 1893 | 0.1401 | 0.1067 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.2 | 0.3 | 824 | 0.2448 | 0.199 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.3 | 0.4 | 441 | 0.3461 | 0.3243 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.4 | 0.5 | 291 | 0.4494 | 0.4777 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.5 | 0.6000000000000001 | 213 | 0.5464 | 0.6901 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.6000000000000001 | 0.7000000000000001 | 208 | 0.651 | 0.7212 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.7000000000000001 | 0.8 | 311 | 0.7556 | 0.8585 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.8 | 0.9 | 567 | 0.8499 | 0.903 |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 0.9 | 1.0 | 47 | 0.921 | 0.8298 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.0 | 0.1 | 6932 | 0.0717 | 0.0603 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.1 | 0.2 | 5839 | 0.1431 | 0.0875 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.2 | 0.3 | 3072 | 0.2455 | 0.1214 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.3 | 0.4 | 2023 | 0.3474 | 0.1453 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.4 | 0.5 | 1564 | 0.4474 | 0.179 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.5 | 0.6000000000000001 | 1334 | 0.5485 | 0.2219 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.6000000000000001 | 0.7000000000000001 | 1203 | 0.6495 | 0.2394 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.7000000000000001 | 0.8 | 1184 | 0.7492 | 0.3488 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.8 | 0.9 | 1357 | 0.8507 | 0.4503 |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 0.9 | 1.0 | 2206 | 0.9548 | 0.7221 |
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

## Standart COCO bantlari (AP_S / AP_M / AP_L)

Literaturle kiyaslanabilirlik icin. Alan esikleri: small < 32^2, medium 32^2-96^2, large > 96^2. Ayni kosumdan, sadece alan araliklari farkli - `all` satiri sqrt tablosuyla birebir ayni olmali.

`outputs/tables/results_coco_bands.csv`

| tag | model | query_set | size_band | mAP@50-95 | mAP@50 | mAP@75 | AR@50-95 | n_gt |
|---|---|---|---|---|---|---|---|---|
| baseline_a_full | YOLO11s fine-tune |  | all | 0.5225198168303696 | 0.8363763319794276 | 0.5335344425127958 | 0.594390100511191 | 9630 |
| baseline_a_full | YOLO11s fine-tune |  | small | 0.472374208447662 | 0.8263386002215772 | 0.4873132232993042 | 0.538997514593416 | 2966 |
| baseline_a_full | YOLO11s fine-tune |  | medium | 0.5283200374665987 | 0.829634823628432 | 0.5540978346613195 | 0.6137995009108878 | 4796 |
| baseline_a_full | YOLO11s fine-tune |  | large | 0.663083204148535 | 0.9002483011597381 | 0.7553495031304364 | 0.7340946251387003 | 1868 |
| baseline_b_canonical_1280 | YOLO-World v2-s | canonical | all | 0.1092975253282732 | 0.1830008735172729 | 0.1161675709079565 | 0.2619978390441946 | 9630 |
| baseline_b_canonical_1280 | YOLO-World v2-s | canonical | small | 0.0111921791594243 | 0.030698065828071 | 0.004993772357397 | 0.0931539323678785 | 2966 |
| baseline_b_canonical_1280 | YOLO-World v2-s | canonical | medium | 0.0669608695309793 | 0.1420852912599637 | 0.0523212953995354 | 0.2806891076287325 | 4796 |
| baseline_b_canonical_1280 | YOLO-World v2-s | canonical | large | 0.2785691849528226 | 0.4250699635510818 | 0.3147132526250009 | 0.5358638699281764 | 1868 |
| baseline_b_attributed_1280 | YOLO-World v2-s | attributed | all | 0.1039520835504764 | 0.1905581023408304 | 0.101472814955762 | 0.3112886446374662 | 9630 |
| baseline_b_attributed_1280 | YOLO-World v2-s | attributed | small | 0.0255901593011349 | 0.070435320095289 | 0.0119165001355663 | 0.2120567821790123 | 2966 |
| baseline_b_attributed_1280 | YOLO-World v2-s | attributed | medium | 0.0569799716714202 | 0.1388738298261979 | 0.0346481692190366 | 0.2858375977836498 | 4796 |
| baseline_b_attributed_1280 | YOLO-World v2-s | attributed | large | 0.203480394354131 | 0.3328058104799714 | 0.2282714323590935 | 0.4723556590716996 | 1868 |
| baseline_a_pilot | YOLO11s fine-tune |  | all | 0.3774570605321172 | 0.6515784886598304 | 0.3747851522546985 | 0.4474596380461456 | 9630 |
| baseline_a_pilot | YOLO11s fine-tune |  | small | 0.2551636344947676 | 0.5275578821374843 | 0.2069618375365513 | 0.2835420347313136 | 2966 |
| baseline_a_pilot | YOLO11s fine-tune |  | medium | 0.41669034404585 | 0.7199184525634087 | 0.4206277317992055 | 0.4980357702491567 | 4796 |
| baseline_a_pilot | YOLO11s fine-tune |  | large | 0.6132664593966934 | 0.9011631793655102 | 0.6839647686017126 | 0.7181720201192745 | 1868 |

## TP/FP guven skoru ayrismasi

Kalibrasyondan FARKLI bir soru: guven skoru dogru ve yanlis tespitleri ne kadar ayiriyor? `auroc` 0.5 = bilgisiz, 1.0 = mukemmel. `overlap` histogram kesisimi (1 = tamamen ic ice). `youden_at_conf` en iyi tek esik.

`outputs/tables/confidence_separation.csv`

| tag | size_band | n_tp | n_fp | auroc | ks | ks_at_conf | overlap | youden_j | youden_at_conf | tp_mean_conf | fp_mean_conf | tp_median_conf | fp_median_conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_a_full | all | 8637 | 3570 | 0.9224 | 0.7028 | 0.508 | 0.2997 | 0.7028 | 0.508 | 0.7116 | 0.2594 | 0.746 | 0.1424 |
| baseline_b_canonical_1280 | all | 1975 | 5537 | 0.8489 | 0.5911 | 0.276 | 0.4106 | 0.5911 | 0.276 | 0.5323 | 0.1631 | 0.5851 | 0.1095 |
| baseline_b_attributed_1280 | all | 5077 | 21637 | 0.7858 | 0.4443 | 0.484 | 0.5558 | 0.4443 | 0.484 | 0.6175 | 0.278 | 0.7217 | 0.1733 |

## TP/FP ayrismasi - boyut bandina gore

Ayni metrikler, nesne boyutuna gore kirilim.

`outputs/tables/confidence_separation_by_band.csv`

| tag | size_band | n_tp | n_fp | auroc | ks | ks_at_conf | overlap | youden_j | youden_at_conf | tp_mean_conf | fp_mean_conf | tp_median_conf | fp_median_conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_a_full | <16 | 129 | 95 | 0.9244 | 0.7851 | 0.454 | 0.2299 | 0.7851 | 0.454 | 0.6228 | 0.2817 | 0.6401 | 0.2506 |
| baseline_a_full | 16-32 | 2185 | 1144 | 0.9012 | 0.6726 | 0.378 | 0.3286 | 0.6726 | 0.378 | 0.6244 | 0.2466 | 0.6481 | 0.1381 |
| baseline_a_full | 32-64 | 3653 | 1736 | 0.9222 | 0.7141 | 0.508 | 0.2873 | 0.7141 | 0.508 | 0.6867 | 0.2436 | 0.7289 | 0.1284 |
| baseline_a_full | >64 | 2670 | 595 | 0.9483 | 0.7507 | 0.754 | 0.2437 | 0.7507 | 0.754 | 0.8215 | 0.3268 | 0.8597 | 0.2168 |
| baseline_b_canonical_1280 | <16 | 3 | 24 | 0.5972 | 0.3333 | 0.14 | 0.5417 | 0.3333 | 0.14 | 0.0942 | 0.0824 | 0.0784 | 0.0779 |
| baseline_b_canonical_1280 | 16-32 | 97 | 1593 | 0.6823 | 0.3016 | 0.142 | 0.6494 | 0.3016 | 0.142 | 0.2503 | 0.1537 | 0.1923 | 0.1111 |
| baseline_b_canonical_1280 | 32-64 | 316 | 2066 | 0.8152 | 0.6057 | 0.272 | 0.3999 | 0.6057 | 0.272 | 0.3782 | 0.1391 | 0.4226 | 0.1034 |
| baseline_b_canonical_1280 | >64 | 1559 | 1854 | 0.8427 | 0.5658 | 0.374 | 0.4269 | 0.5658 | 0.374 | 0.582 | 0.1991 | 0.7173 | 0.1173 |
| baseline_b_attributed_1280 | <16 | 10 | 159 | 0.8132 | 0.5434 | 0.32 | 0.2692 | 0.5434 | 0.32 | 0.3974 | 0.1451 | 0.3807 | 0.1063 |
| baseline_b_attributed_1280 | 16-32 | 991 | 3291 | 0.6777 | 0.2689 | 0.408 | 0.7135 | 0.2689 | 0.408 | 0.4928 | 0.3107 | 0.4857 | 0.2172 |
| baseline_b_attributed_1280 | 32-64 | 1942 | 7656 | 0.7448 | 0.3784 | 0.478 | 0.6154 | 0.3784 | 0.478 | 0.5531 | 0.2814 | 0.5944 | 0.1895 |
| baseline_b_attributed_1280 | >64 | 2134 | 10531 | 0.8587 | 0.5872 | 0.544 | 0.4111 | 0.5872 | 0.544 | 0.735 | 0.2674 | 0.9218 | 0.154 |

## Hiz - batch=1, isinma haric (ADIM 5)

`12_benchmark_speed.py` ciktisi. torch.cuda.synchronize() ile uctan uca olcum, ilk turlar atilir. Asagidaki `timing.csv`'nin aksine bu karsilastirmali iddia icin kullanilabilir.

`outputs/tables/timing_batch1.csv`

| model | imgsz | weights | batch_size | hardware | n_prompts | conf | iou_nms | n_measured | warmup_excluded | latency_ms_mean | latency_ms_median | latency_ms_p95 | latency_ms_min | latency_ms_max | latency_ms_std | fps_batch1 | method |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s | 1280 | outputs/runs/baseline_a_full_1280/weights/best.pt | 1 | NVIDIA GeForce RTX 5070, torch 2.12.0.dev20260408+cu128, Windows 11 |  | 0.25 | 0.7 | 300 | 50 | 27.971 | 26.886 | 47.658 | 12.702 | 64.803 | 8.264 | 35.75 | torch.cuda.synchronize() ile uctan uca; isinma turlari haric |
| YOLO11s | 640 | outputs/runs/baseline_a_pilot_640/weights/best.pt | 1 | NVIDIA GeForce RTX 5070, torch 2.12.0.dev20260408+cu128, Windows 11 |  | 0.25 | 0.7 | 300 | 50 | 22.959 | 22.331 | 42.038 | 9.224 | 61.258 | 7.737 | 43.56 | torch.cuda.synchronize() ile uctan uca; isinma turlari haric |

## Hiz - kaba gosterge (KISMI - bkz. not)

**`latency_ms_batch1`, `fps_batch1`, `warmup_excluded` OLCULMEDI** (adim 5 kosulmadi). Dolu sutunlar tahmin kosusunun yan urunu: chunk=16, batch=1 degil, isinma turlari haric tutulmadi. Karsilastirmali hiz iddiasi icin kullanilamaz.

`outputs/tables/timing.csv`

| model | query_set | tag | imgsz | hardware | batch_size | preprocess_ms | inference_ms | postprocess_ms | total_ms_per_image | throughput_img_per_s_wallclock | latency_ms_batch1 | fps_batch1 | warmup_excluded | measurement_note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s fine-tune |  | baseline_a_full | 1280 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 3.66 | 6.07 | 1.2 | 10.93 | 15.19 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |
| YOLO-World v2-s | canonical | baseline_b_canonical_1280 | 1280 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 3.73 | 6.99 | 1.23 | 11.95 | 15.08 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |
| YOLO-World v2-s | attributed | baseline_b_attributed_1280 | 1280 | NVIDIA RTX 5070 (12.2 GB), torch 2.12.0.dev+cu128, ultralytics 8.4.41 | 16 | 3.76 | 7.02 | 1.2 | 11.98 | 15.18 |  |  |  | kaba gosterge: chunk=16, isinma turlari haric tutulmadi, batch=1 degil. Resmi gecikme/FPS icin adim 5 kosulmali. |
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
