# Veri seti notlari — SeaDronesSee (Object Detection v2)

Kod yazarken karsilasilan pratik detaylar. Amac: ayni tuzaklara ikinci kez
dusmemek. Ilgili sayilar `outputs/tables/dataset_stats.csv` ve
`outputs/tables/size_distribution.csv` dosyalarinda.

---

## 1. Klasor yapisi

```
data/
  images/
    train/   8930 .jpg     (id.jpg, orn. 3388.jpg)
    val/     1547 .jpg
    test/    3750 .jpg     <- ANOTASYON YOK
  annotations/
    instances_train.json          COCO JSON formati
    instances_val.json
    train_images_metadata.csv     JSON'daki meta'nin duz tablo hali
    val_images_metadata.csv
```

**`test/` bolumunde anotasyon yok.** SeaDronesSee test etiketlerini degerlendirme
sunucusunda tutuyor. Bu projedeki tum degerlendirme **val** uzerinde yapiliyor;
"test sonucu" diye bir sey uretilmedi.

`data/` salt-okunur kabul ediliyor. YOLO formatindaki agac `outputs/yolo_dataset/`
altina **hardlink** ile kuruluyor (disk kopyalanmiyor, `data/` degismiyor).

---

## 2. Anotasyon formati

Dagitim COCO JSON *formatinda* — COCO veri setiyle ilgisi yok, sinif listesi
tamamen SeaDronesSee'nin.

Ust duzey anahtarlar: `info`, `licenses`, `categories`, `images`, `annotations`.
`info.version = "2"`, `date_created = 2022-08-15`. `licenses` bos/"Unknown".

### `annotations[]`
```
id, image_id, category_id, bbox, area
```
- `bbox` = `[x, y, w, h]`, **mutlak piksel**, hepsi tam sayi.
- `area` == `w * h` (dogrulandi, birebir). Yani `area` ayri bir maske alani degil.
- **`iscrowd` alani YOK** — asagida, tuzaklar bolumune bak.
- `segmentation` yok; sadece kutu.

### `images[]`
```
id, file_name, width, height, date_time, frame, source, meta
```
- `id` == dosya adinin koku (`3388.jpg` -> `id=3388`). Her iki split'te de
  birebir dogrulandi; tahmin dosyalarinda `image_id` eslemesi bu sayede kolay.
- `source` = `{drone, folder_name, video, frame_no}` — kareler videolardan
  cikarilmis. **Ayni videodan ardisik kareler var**, yani train/val icinde
  gorsel olarak neredeyse ayni goruntuler bulunabilir.
- `meta` = ucus telemetrisi (asagida) veya **`None`**.

### `meta` alanlari (varsa)
```
height_above_takeoff(meter), latitude, longitude, datetime(utc), image_name,
speed(m/s), xspeed(m/s), yspeed(m/s), zspeed(m/s),
compass_heading(degrees), gimbal_heading(degrees), gimbal_pitch(degrees)
```

---

## 3. Siniflar

| id | isim | anlami |
|----|------|--------|
| 0 | `ignored` | **Hic kullanilmamis** — 0 anotasyon (her iki split'te de) |
| 1 | `swimmer` | suda insan (yuzucu / kazazede) |
| 2 | `boat` | tekne |
| 3 | `jetski` | jetski |
| 4 | `life_saving_appliances` | can simidi / can yelegi / kurtarma sali gibi cansiz kurtarma ekipmani |
| 5 | `buoy` | samandira / duba |

`ignored` sinifi bildiriliyor ama **hicbir anotasyonda gecmiyor**. Donusumde
(`02_prepare_yolo_dataset.py`) atlaniyor; YOLO indeksleri 0-4 olarak
`swimmer, boat, jetski, life_saving_appliances, buoy` sirasiyla ataniyor.
Esleme `outputs/yolo_dataset/class_mapping.json` icinde kayitli — degerlendirme
bu dosyaya guveniyor, elle degistirilmemeli.

`life_saving_appliances` gorunusten cok **isleve** dayali bir kategori (bir
nesne "kurtarma ekipmani" cunku ne ise yaradigi belli, benzedigi sey degil).
Acik kelime dagarcikli modelde tam olarak bu yuzden coktugu dusunuluyor.

---

## 4. Sayilar (ozet)

| | train | val |
|---|---|---|
| goruntu | 8930 | 1547 |
| nesne | 57760 | 9630 |
| goruntu basina nesne (ort / medyan / **maks**) | 6.47 / 6 / **16** | 6.22 / 6 / **16** |

Sinif dagilimi ciddi dengesiz: `swimmer` 43302, `life_saving_appliances` 1253
(train+val) — **35 kat fark**.

Goruntu basina en fazla 16 nesne var; bu yuzden `max_det=300` fazlasiyla
yeterli, kirpma riski yok.

Nesnelerin **~%45'i `sqrt(w*h) <= 32 px`**. Boyut bandi tanimi olarak
`sqrt(w*h)` ("esdeger kare kenari") secildi — bbox kare olmadigi icin bu bir
tercih, `max(w,h)` veya `min(w,h)` farkli tablo uretir.

---

## 5. Drone / cozunurluk / irtifa iliskisi

Uc farkli platform karisik ve **cozunurluk ile telemetri varligi drone tipine
bagli**:

| drone | val goruntu | cozunurluk(ler) | irtifa var mi |
|---|---|---|---|
| `mavic` | 937 | 3840x2160 | **evet, hepsinde** |
| `m210` | 351 | 1920x1080 (258), 3840x2160 (93) | **hayir, hicbirinde** |
| `trinity` | 259 | 5456x3632 (149), 3632x5456 (47), ~1230x933 (20+) | **hayir** |

Irtifasiz goruntu iki ayri sebepten olabiliyor — ikisini ayirmak gerekiyor:

| | train | val |
|---|---|---|
| `meta` tamamen `None` | 2718 | 454 |
| `meta` var ama `height_above_takeoff` `None` | 412 | 63 |
| **toplam irtifasiz** | **3130 (%35)** | **517 (%33)** |

`meta` var ama irtifa `None` olanlarin **tamami `trinity`**. Yani `meta`
sozlugu varsa irtifa da vardir diye varsayilamaz — iki ayri kontrol sart.

**Sonuc:** irtifa bazli her analiz goruntulerin ~%65'iyle sinirli ve bu alt
kume **drone tipine gore yanlidir** (pratikte "sadece mavic"). Irtifa etkisi
ile platform etkisi bu veride ayristirilamiyor.

`trinity` ayrica **portre** kareler uretiyor (3632x5456). Sabit `imgsz` ile
letterbox sonrasi efektif nesne olcegi bu karelerde farkli — boyut bandi
analizinde gizli karistirici degisken.

---

## 6. Tuzaklar (hepsi bu projede basimiza geldi)

**`iscrowd` alani yok.** `pycocotools` bunu zorunlu tutuyor; eksikse `COCOeval`
`KeyError: 'iscrowd'` ile **eval'in en basinda** cokuyor. `05_eval_detection.py`
icindeki `load_gt()` eksikse 0 atiyor. Yeni bir metrik kodu yazilirsa ayni
yamanin tekrar gerekecegi unutulmamali.

**`COCOeval.params.catIds` `evaluate()`'ten ONCE atanmali.** Sonra atanirsa
precision dizisinin sinif ekseni bos `ignored` sinifini da icerir ve **tum
sinif sonuclari bir kayar** — hata vermez, sessizce yanlis sayi uretir.
Bu bug'i ground-truth'u mukemmel tahmin olarak verip mAP=1.0 bekleyen sanity
test yakaladi. **Yeni metrik kodunda bu test tekrarlanmali.**

**Tekrarlanan kutular.** Train'de 16 birebir ayni (image_id, category, bbox)
kaydi var; ultralytics egitimde bunlari otomatik atiyor ("duplicate labels
removed"). **Val'de 0** — degerlendirme seti temiz.

**Bozuk/tasan kutu yok.** Donusumde 0 kirpma, 0 dejenere kutu. Yani sinir
kontrolu gerekli degil ama kod yine de yapiyor.

**CSV metadata dosyalarinda tekrarlanan sutun adlari var** (`image_name` iki
kez, `date_time` iki kez). `pandas.read_csv` bunlari `image_name.1` diye
yeniden adlandirir. JSON'daki `meta` ayni bilgiyi temiz veriyor; CSV'lere
gerek olmadi.

**Windows: `<` ve `>` dizin adinda kullanilamaz.** Boyut bandi etiketleri
(`<16`, `>64`) klasor adi olarak kullanilinca `OSError` veriyor. Kirpinti
agacinda `lt16` / `gt64` diye sterilize ediliyor (`BAND_DIR`); CSV ve
grafiklerde orijinal etiket korunuyor.

---

## 7. Anotasyon eksikligi (onemli)

Hata analizindeki "background" yanlis pozitiflerinin gorsel incelemesi,
bunlarin buyuk kismininin **model halusinasyonu degil, anotasyonu yapilmamis
gercek nesneler** oldugunu gosterdi: acik secik yuzuculer, tekneler, jetskiler,
can simitleri — hicbiri GT'de yok.

Incelenen 81 kirpintinin **%47'si gercek ve etiketsiz** nesne, sadece **%5'i**
gercek anlamda kopuk/parilti. Ayrinti ve orneklem yanliligi uyarisi:
[error-taxonomy.md](error-taxonomy.md).

**Pratik sonucu:** bu veri setinde raporlanan precision bir **alt sinir**.
Modelin "yanlis" dedigimiz tespitlerinin onemli bir kismi aslinda dogru.
Tez metninde precision rakamlari bu cekinceyle sunulmali.

---

## 8. Ortam notu

Bu oturumdaki tum kosular **Windows 11 / RTX 5070 (12.2 GB) / torch
2.12.0.dev+cu128 / ultralytics 8.4.41** uzerinde yapildi — CLAUDE.md'de yazan
Ubuntu makinesinde degil. Mutlak sureler orada farklidir; oranlar korunur.

Ultralytics'e ozgu iki davranis:
- **Goreli `project` yolu** kendi `runs/detect/` kokunun altina gomuluyor
  (`runs/detect/outputs/runs/...` gibi). Mutlak yol vermek sart.
- **`batch=-1` (AutoBatch) 12 GB'ta imgsz=1280 icin batch 2 seciyor** ve
  tahmini de hatali (5.5 G ongorup 2.5 G kullaniyor). Elle `batch=8`
  verildiginde %49 daha hizli. `03_train_yolo.py` icinde sabitlendi.
- **`model.predict()` toplu cagride `res.path`'i korumuyor** (`image0.jpg`
  donuyor); yol bilgisi girdi listesinden eslestirilmeli.
- **Buyuk kaynak listesi RAM sisiriyor**: 1500+ goruntuluk tek listede
  `Results` nesneleri (her biri 4K `orig_img` tasiyor) birikiyor, 22 GB RAM'e
  cikip GPU'yu bos birakiyor. `predict_common.run_chunked()` parcali isliyor.

YOLO-World `set_classes()` icin **CLIP** gerekiyor; ultralytics
`git+https://github.com/ultralytics/CLIP.git` adresinden otomatik kuruyor
(ag erisimi ve git sart). CLIP ViT-B/32 agirligi (338 MB) ilk cagrida iniyor.
