# Bulgu günlüğü

SeaDronesSee (Object Detection v2) keşif deneyi. En yeni kayıt üstte.

Ortam notu: bu oturumdaki tüm koşular Windows 11 / RTX 5070 (12.2 GB) / torch
2.12.0.dev+cu128 / ultralytics 8.4.41 üzerinde yapıldı — CLAUDE.md'de yazan Ubuntu
makinesi değil. Mutlak süreler o makinede farklı çıkabilir; oranlar korunur.

---

## 2026-08-14 — Hata analizi: FP taksonomisi, kırpıntılar, kalibrasyon

- **Çalıştırılan:**
  `scripts/08_error_analysis.py --tag {baseline_a_pilot, baseline_b_canonical, baseline_b_attributed} --conf 0.25`
  (varsayılanlar: IoU 0.5, loc_min 0.1, context 3.0, crop_min_px 128, max_crops 60)

- **Sonuç:** FP alt tipi dağılımı (TIDE tarzı taksonomi):

  | FP tipi | A: fine-tune | B: kanonik | B: öznitelikli |
  |---|---|---|---|
  | localization | **67.2%** (1595) | 11.5% (67) | 19.3% (879) |
  | class_confusion | 5.8% (137) | **64.7%** (378) | **47.1%** (2141) |
  | background | 11.7% (277) | 21.2% (124) | 24.7% (1123) |
  | duplicate | 15.3% (364) | 2.6% (15) | 8.9% (403) |

  Baseline A: TP 7435, FP 2373, FN 2195 → precision 0.758, recall 0.772.

  Yanlış negatif, boyut bandına göre:

  | Bant | A | B kanonik | B öznitelikli |
  |---|---|---|---|
  | `<16` | 150 | 182 (bandın tamamı) | 182 (bandın tamamı) |
  | `16-32` | 1220 | 2780 | 2514 |
  | `32-64` | 640 | 3829 | 3000 |
  | `>64` | 185 | 1782 | 1241 |
  | toplam | 2195 | 8573 | 6937 |

  Sınıf karışıklığı (satır = tahmin, sütun = gerçek), öne çıkanlar:
  - A: `swimmer ← life_saving_appliances` 53
  - B kanonik: `buoy ← swimmer` 134, `boat ← jetski` 73, `boat ← swimmer` 61
  - B öznitelikli: `jetski ← swimmer` 480, `buoy ← swimmer` 366, `boat ← swimmer` 274

  ECE: A 0.137, B kanonik 0.0616, B öznitelikli 0.127.

  Çıktılar: `outputs/error_analysis/<tag>/` — manifest.csv, fp_summary.csv,
  fn_summary.csv, calibration.csv, summary.txt, figures/ (3 grafik),
  crops/<hata_tipi>/<sınıf>/<bant>/ (A için 1597 kırpıntı).

- **Dikkat çeken:**
  1. **Baskın hata modu iki model için farklı.** Kapalı kümede lokalizasyon (%67),
     açık kelime dağarcığında sınıf karışıklığı (%47-65). Deneyin başında ilk
     şüpheli olan "background" FP'leri (köpük/yansıma/kaya) hiçbir konfigürasyonda
     baskın değil — %12-25 arasında kalıyor. Bu, tez sorusunun ön kabulünü
     doğrudan çürütüyor.
  2. **B nesneyi buluyor, adını koyamıyor.** Öznitelikli sette 2141 class_confusion
     FP'si var ve bunların büyük kısmı gerçek yüzücülerin jetski/şamandıra/tekne
     olarak etiketlenmesi. Yani B'nin başarısızlığı tespit değil, semantik.
     Kırpıntıyla görsel olarak da doğrulandı.
  3. A'daki `swimmer ← life_saving_appliances` (53) kırpıntıda net görülüyor: suda
     yüzen turuncu can simidi insan sanılıyor. CLAUDE.md'de "şamandıra insana
     benziyor" diye tarif edilen senaryonun gerçek karşılığı bu.
  4. B `<16` bandında sıfır FP üretiyor ve bandın tamamını kaçırıyor — o
     çözünürlükte hiçbir şey görmüyor, yanlış görmüyor.
  5. **B kanonik'in düşük ECE'si (0.0616) yanıltıcı.** Hem güven hem doğruluk
     sıfıra yakın olduğu için nokta bulutu köşegene yapışıyor; iyi kalibrasyon
     değil, artefakt. Kalibrasyon karşılaştırmasında bu tek sayıya bakılmamalı.
  6. A'nın güvenilirlik diyagramı sigmoid: ~0.4 altında aşırı özgüvenli, üstünde
     fazla temkinli (0.9 güvende gerçek doğruluk 1.0).

- **Sonraki soru:** A'daki lokalizasyon baskınlığı gerçek bir kutu-regresyon
  zayıflığı mı, yoksa 640 ölçeklemesinin artefaktı mı? *Tahmin (ölçülmedi):* 20px'lik
  bir nesnede birkaç piksel kayma IoU'yu 0.5'in altına düşürdüğü için oranın önemli
  kısmının artefakt olduğunu ve 1280 koşusunda localization payının düşeceğini
  bekliyorum. Düşmezse mimari/regresyon sorunu demektir. **1280 sonrası tekrar
  ölçülecek ilk sayı bu.** Ayrıca tüm sayılar `--conf 0.25`'e bağlı; eşik
  duyarlılığı taranmadı.

---

## 2026-08-14 — 1280 çözünürlük için batch ve süre ölçümü

- **Çalıştırılan:** `scripts/03_train_yolo.py --preset full --epochs {1,2} --name probe_1280*`
  (biri AutoBatch `batch=-1`, biri `--batch 8`). Probe dizinleri sonra silindi.

- **Sonuç (ölçüm):**

  | Batch | VRAM | Görüntü/sn | Epoch süresi |
  |---|---|---|---|
  | 2 (AutoBatch seçimi) | 2.5 G | 22.0 | ~6.8 dk |
  | 8 (elle) | 8.3 G | 32.8 | **295.5 sn** (val dahil) |

  100 epoch × 295.5 sn = **~8.2 saat**. GPU %80, toplam kullanım 9.0/12.2 GB.

  Ultralytics iç metriğiyle elmayla elma:

  | Koşu | Süre | mAP50 | mAP50-95 |
  |---|---|---|---|
  | 640, 10 epoch | 840 sn | 0.663 | 0.380 |
  | 1280, **1 epoch** | 296 sn | **0.724** | **0.408** |

- **Dikkat çeken:**
  1. **AutoBatch (`batch=-1`) 12 GB'ta 1280 için batch 2 seçiyor ve tahmini de
     hatalı** — 5.51 G öngörüp gerçekte 2.49 G kullanıyor. Sonuç hem %33 daha yavaş
     hem de batch 2'de gradyan gürültüsü yüksek. `PRESETS["full"]["batch"]` 8'e
     sabitlendi, koda gerekçe notu düşüldü.
  2. **1280'de tek epoch, 640'ta on epoch'u geçiyor — üçte bir sürede.** Bu veri
     setinde çözünürlük epoch sayısından baskın. Küçük hedeflerin 640'ta piksel
     düzeyinde yok olması asıl darboğaz; tez metninde tek başına savunulabilir bir
     gözlem.

- **Sonraki soru:** 1280 yeterli mi, yoksa tiling/SAHI gerekli mi? 1280'de bile
  3840 genişlikteki bir görüntüde 16px nesne ~5.3px'e iniyor. *Tahmin (ölçülmedi):*
  `<16` bandı 1280'de de zayıf kalacak; o bandı gerçekten ölçmek için tiling şart
  olabilir.

---

## 2026-08-14 — Baseline B: YOLO-World zero-shot, iki sorgu seti

- **Çalıştırılan:**
  - `scripts/06_predict_yoloworld.py --prompt-set {canonical,attributed} --tag baseline_b_{canonical,attributed} --imgsz 640`
    (model `yolov8s-worldv2.pt`, conf 0.001, NMS IoU 0.7, max_det 300)
  - `scripts/05_eval_detection.py` her iki tag için
  - `scripts/07_compare_baselines.py --tags baseline_a_pilot baseline_b_canonical baseline_b_attributed`

- **Sonuç:** mAP@50, boyut bandına göre:

  | Bant (n) | A: fine-tune | B: kanonik | B: öznitelikli |
  |---|---|---|---|
  | `<16` (182) | 0.177 | 0.000 | 0.005 |
  | `16-32` (2784) | 0.541 | 0.009 | 0.040 |
  | `32-64` (3851) | 0.705 | 0.057 | 0.103 |
  | `>64` (2813) | 0.892 | 0.289 | 0.242 |
  | tümü | **0.652** | 0.137 | 0.146 |

  Sınıf bazında mAP@50 (tümü): kanonik → öznitelikli
  - swimmer 0.068 → **0.184**
  - boat 0.571 → 0.516
  - buoy 0.040 → 0.004
  - jetski 0.006 → 0.028
  - life_saving_appliances 0.000 → 0.000

  Sınıf-bazlı NMS'in attığı mükerrer kutu: kanonik 65 (%0.3), öznitelikli 6355 (%7.8).

- **Dikkat çeken:**
  1. **Öznitelikli ifadeler yalnızca swimmer'da net kazanç veriyor (2.7×).** "person
     floating in the water" / "orange life vest", çıplak "swimmer" kelimesini açık
     ara geçiyor. Ama buoy'da 10× kötüleşiyor, boat'ta hafif düşüyor. "Öznitelikli
     sorgu daha iyidir" genellemesi bu veride tutmuyor — SAR'ın asıl hedefi olan
     insan sınıfında tutuyor, ki tez açısından işe yarayan kısım bu.
  2. **life_saving_appliances her iki sette de tam sıfır.** Görünüşten çok işleve
     dayalı (soyut) bir kategori; açık kelime dağarcığının bu tip sınıflarda
     tamamen çöktüğüne dair temiz bir örnek.
  3. **Çok-e-bir prompt eşlemesinde eşleme *sonrası* sınıf-bazlı NMS şart.** Aynı
     nesnede iki farklı ifade ateşleyip SeaDronesSee sınıfına eşlendikten sonra
     mükerrer kutuya dönüşüyor; bu adım atlanırsa B haksız yere cezalanır (%7.8
     fazladan FP). Kanonik setteki 65 kutu sadece koordinat ölçekleme yuvarlaması.
  4. jetski zero-shot'ta neredeyse sıfır — görsel olarak ayırt edici bir kavram
     olmasına rağmen. Bu, adım 4'te sınıf karışıklığı olarak doğrulandı.

- **Sonraki soru:** Sorgu metinleri bir araştırma değişkeni ve şu an tek bir el
  yazımı sete dayanıyor (`scripts/prompt_sets.json`). Sistematik prompt taraması
  yapılmalı mı, yoksa tez için "prompt duyarlılığı yüksektir" gözlemi yeterli mi?

---

## 2026-08-14 — Baseline A pilot: YOLO11s fine-tune @640

- **Çalıştırılan:**
  - `scripts/03_train_yolo.py --preset pilot --workers 4` (yolo11s.pt, 640, 10 epoch, batch 16, seed 0)
  - `scripts/04_predict_to_json.py --weights .../best.pt --imgsz 640 --tag baseline_a_pilot`
  - `scripts/05_eval_detection.py --pred ... --tag baseline_a_pilot`

- **Sonuç:** eğitim 840 sn (84 sn/epoch). Val, pycocotools ile:

  | Bant | mAP@50-95 | mAP@50 | mAP@75 | AR@50-95 |
  |---|---|---|---|---|
  | tümü | 0.378 | 0.652 | 0.375 | 0.448 |
  | `<16` | 0.084 | 0.177 | 0.060 | 0.091 |
  | `16-32` | 0.260 | 0.541 | 0.208 | 0.289 |
  | `32-64` | 0.398 | 0.705 | 0.396 | 0.471 |
  | `>64` | 0.578 | 0.892 | 0.634 | 0.684 |

  Sınıf bazında mAP@50 (tümü): boat 0.934, jetski 0.865, swimmer 0.690,
  buoy 0.581, life_saving_appliances 0.189.

- **Dikkat çeken:**
  1. Bozulma monoton ve dik: `>64` → `<16` arasında mAP@50 0.892'den 0.177'ye
     düşüyor (5×). Tezin merkezindeki olgu temiz şekilde ölçülebiliyor.
  2. life_saving_appliances fine-tune'da bile 0.189 — EDA'daki 1253 örneklik sınıf
     dengesizliği doğrudan performansa yansıyor.
  3. **`<16` bandındaki 0.177 büyük ölçüde ölçekleme artefaktı.** 640'ta 3840
     genişlikli bir görüntüde 16px nesne ağa 2.7px giriyor. Bu satır 1280 koşusundan
     önce yorumlanmamalı.
  4. Pilot 10 epoch olduğu ve `close_mosaic` varsayılanı 10 olduğu için mosaic
     augmentation hiç çalışmadı ("Closing dataloader mosaic"). 100 epoch'luk koşuda
     90 epoch aktif olacak; pilot ↔ tam koşu farkının bir kısmı buradan gelecek.

- **Sonraki soru:** `boat <16` (n=1) ve `life_saving_appliances <16` (n=5) gibi
  hücreler tek haneli örneklem — bunlar raporlanırken n ile birlikte verilmeli,
  yoksa gürültü sonuç gibi okunur.

---

## 2026-08-14 — Değerlendirme yolunun doğrulanması (iki bug)

- **Çalıştırılan:** ground truth'u skor 1.0'lık "mükemmel tahmin" olarak
  `05_eval_detection.py`'ye verip mAP = 1.0 beklendi. Ayrıca `08_error_analysis.py`
  için dört sentetik bozunum senaryosu.

- **Sonuç:** iki bug yakalandı ve düzeltildi:
  1. **`iscrowd` eksikliği:** SeaDronesSee anotasyonlarında bu alan yok, pycocotools
     zorunlu tutuyor → eval `KeyError` ile baştan çöküyordu. `load_gt()` içinde 0
     atanıyor.
  2. **`catIds` sıralama hatası:** `params.catIds` `evaluate()`'ten *sonra*
     atanmıştı; precision dizisinin sınıf ekseni boş `ignored` (id=0) sınıfını da
     içerdiği için tüm sınıf sonuçları bir kayıyordu — swimmer'ın skoru boat'a
     yazılıyordu. `evaluate()` öncesine alındı, koda uyarı notu düşüldü.

  Düzeltme sonrası sanity test tüm bantlarda 1.0 döndü ve eval'in `n_gt` sütunu
  (182/2784/3851/2813) EDA'daki val bant sayımlarıyla birebir tuttu.

  Hata analizi testleri: mükemmel tahmin → 0 FP / 0 FN; sınıf değiştirme → 200
  class_confusion; 0.3×kaydırma → 200 localization; kopyalama → 200 duplicate.
  Hepsi birebir.

- **Dikkat çeken:** `catIds` bug'ı sessizdi — hata vermiyor, sadece yanlış sayı
  üretiyordu. Sentetik sanity test olmasa tez boyunca fark edilmeyebilirdi. Yeni
  metrik kodu yazıldığında bu testin tekrarlanması gerekiyor.

- **Sonraki soru:** Yok; doğrulama tamam.

---

## 2026-08-14 — Veri hazırlığı: SeaDronesSee → YOLO formatı

- **Çalıştırılan:** `scripts/02_prepare_yolo_dataset.py` (varsayılan `--link-mode auto`)

- **Sonuç:** train 8930 görüntü / 57760 anotasyon, val 1547 / 9630. Kırpılan kutu 0,
  atılan bozuk kutu 0, eşlenemeyen kategori 0. Görüntüler **hardlink** ile
  `outputs/yolo_dataset/` altına bağlandı (`data/` değiştirilmedi, disk
  kopyalanmadı). Sınıf eşlemesi `class_mapping.json`'a yazıldı
  (SeaDronesSee id 1-5 → YOLO 0-4).

- **Dikkat çeken:**
  1. Train'de 16 birebir tekrarlanan kutu var (57760 içinde); ultralytics bunları
     otomatik atıyor. **Val'de 0** — değerlendirme seti temiz, metrikler etkilenmiyor.
  2. Ultralytics göreli `project` yolunu kendi `runs/detect/` kökü altına gömüyor
     (`runs/detect/outputs/runs/...` oluştu). `03_train_yolo.py` mutlak yola
     çevrildi, koda not düşüldü.
  3. Adlandırma: COCO burada veri seti değil, SeaDronesSee'nin dağıtım *formatı*.
     Karışıklığı önlemek için script/değişken adları `SDS_*` / `sds_to_yolo` olarak
     düzeltildi.

- **Sonraki soru:** Yok.

---

## 2026-08-14 — Veri keşfi (EDA)

- **Çalıştırılan:** `scripts/01_explore_data.py`
  (boyut metriği: `sqrt(w*h)` eşdeğer kare kenarı — bbox kare olmadığı için bu bir
  tasarım tercihi)

- **Sonuç:** train 8930 görüntü / 57760 nesne, val 1547 / 9630. 5 sınıf.

  Sınıf dağılımı (train+val): swimmer 43302, boat 15236, buoy 4949, jetski 2650,
  life_saving_appliances 1253.

  Boyut bandı (train / val): `<16` 3184/182, `16-32` 18587/2784,
  `32-64` 20311/3851, `>64` 15678/2813.

  Görüntü başına nesne: ortalama ~6.2-6.5, medyan 6, **maks 16**.

  Çözünürlük: 3840×2160 (5800), 1920×1080 (1576), 5456×3632 (860), 3632×5456 (282)
  ve ~1230×933 civarında 15+ küçük varyant.

  Çıktılar: `outputs/eda/` (6 CSV + 4 PNG).

- **Dikkat çeken:**
  1. **Sınıf dengesizliği ciddi:** swimmer 43302 vs life_saving_appliances 1253 (35×).
  2. **Nesnelerin ~%45'i ≤32px** eşdeğer kenara sahip; tezin merkezindeki bant zaten
     veri setinin çoğunluğu.
  3. Görüntülerin **~%34'ünde `meta` alanı yok** (train 3130/8930, val 517/1547),
     yani irtifa bilgisi eksik. İrtifa bazlı her analiz bu örneklem kısıtıyla
     raporlanmalı.
  4. Çözünürlük tek tip değil — birden fazla drone/kaynak karışmış. Sabit `imgsz`
     ile eğitimde efektif nesne boyutu görüntüden görüntüye değişiyor; bu, boyut
     bandı analizinde gizli bir karıştırıcı değişken.
  5. Görüntü başına maks 16 nesne → `max_det=300` fazlasıyla yeterli, kırpma riski yok.
  6. `test/` bölümünde anotasyon yok (3750 görüntü) — SeaDronesSee test seti
     değerlendirme sunucusunda tutuluyor. **Tüm değerlendirme val üzerinde yapılıyor.**

- **Sonraki soru:** Sabit `imgsz` altında çözünürlük karışıklığının boyut bandı
  sonuçlarını ne kadar kirlettiği ölçülmedi. Gerekirse bantlar piksel yerine
  *görüntüye oranlı* boyutla da tanımlanabilir.
