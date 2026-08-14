# Hata taksonomisi — gorsel inceleme

Kapali-kume modelinin (`baseline_a_pilot`, YOLO11s @640, conf>=0.25, IoU>=0.5)
hata kirpintilarinin goze bakilarak kategorize edilmesi.

Kaynak kirpintilar: `outputs/error_analysis/baseline_a_pilot/crops/`
Kontakt sayfalari: `outputs/error_analysis/baseline_a_pilot/contact_sheets/`
Etiketlemelerim: `outputs/tables/fp_visual_taxonomy_sample.csv`,
`outputs/tables/fn_visual_taxonomy_sample.csv`
(indeks -> dosya/boyut/guven eslemesi ayni klasordeki `*_index.csv`'lerde)

---

## Orneklem ve yanliligi — once bunu oku

Sayilar **incelenen ornek icindir, tum hata kumesine genellenemez.** Iki ayri
yanlilik var:

1. **Kirpintilar zaten secilmis.** `08_error_analysis.py` klasor basina en fazla
   60 kirpinti kaydediyor ve siralama rastgele degil: **FP'de guven skoruna gore
   azalan**, **FN'de nesne boyutuna gore azalan**. Yani FP ornegi yuksek guvenli
   hatalara, FN ornegi buyuk kacirmalara kayik.
2. **Ben de hepsine bakmadim.** FP'de 225 kirpintinin 81'ine (4 kontakt sayfasi),
   FN'de 240 swimmer kirpintisinin 48'ine (2 sayfa) baktim. FN sayfalari
   kasitli olarak uctan secildi (en buyukler + en kucukler), yani orta boy
   temsil edilmiyor.

Emin olamadigim her kirpinti **`belirsiz`** sayildi; zorlama siniflandirma
yapmadim. `belirsiz` orani yuksekse bu bir basarisizlik degil, bulgunun kendisi.

Kategoriler onceden belirlenmedi; kirpintilara bakarken olustu.

---

## A. Yanlis pozitifler — `background` alt tipi

Neden sadece `background`: diger uc FP alt tipi (`localization`,
`duplicate`, `class_confusion`) tanimi geregi **gercek bir nesnenin uzerinde**.
"Neye benziyor" sorusu yalnizca hicbir GT ile ortusmeyen `background` icin
anlamli.

Incelenen ornek: **n = 81** (toplam 225 `background` FP icinden)

| Kategori | n | %  | ort. kenar (px) | ort. guven |
|---|---|---|---|---|
| `belirsiz` | 39 | 48.1 | 24.1 | 0.35 |
| `gercek_anotasyonsuz_insan` | 20 | 24.7 | 122.3 | 0.44 |
| `gercek_anotasyonsuz_deniz_araci` (jetski/yelkenli/surf) | 8 | 9.9 | 74.4 | 0.49 |
| `gercek_anotasyonsuz_tekne` | 7 | 8.6 | 62.9 | 0.72 |
| `dalga_kopugu_parilti` | 4 | 4.9 | 36.0 | 0.33 |
| `gercek_anotasyonsuz_can_simidi` | 3 | 3.7 | 32.3 | 0.67 |

### Okunacak uc sey

**1. "Yanlis pozitif"lerin en az %47'si aslinda dogru.** Dort "gercek
anotasyonsuz" kategorinin toplami 38/81. Bunlar tartismasiz gercek nesneler —
yesil/turuncu can yelegi giymis yuzuculer, motorlu tekneler, kiyidaki
jetskiler, suda yuzen turuncu can simitleri — ve hicbirinin ground truth'ta
karsiligi yok. Model dogru bulmus, veri seti etiketlememis.

**2. Deneyin baslangic hipotezi bu orneklemde dogrulanmadi.** "Kopuk, yansima,
kaya, samandira insana benziyor" beklentisinin karsiligi olan
`dalga_kopugu_parilti` sadece **%4.9** — ve o dordu de nesnesiz su dokusu
(beyaz kirilma cizgileri, gunes parlamasi), ortalama guven 0.33 ile en dusuk
guvenli kategori. Yani model bunlara zaten pek inanmiyor.

**3. `belirsiz` kucuk nesnelerde yogunlasiyor.** Ortalama kenar 24 px; en
kucuk bant (`<16`, 11-13 px) kirpintilarinin **tamami** belirsiz cikti — gri
capraz dalgali suda koyu lekeler, insan basi da olabilir dalga golgesi de.
Bu bir olcum siniri: **o boyutta hata tipi gozle ayirt edilemiyor**, dolayisiyla
`<16` bandi icin "hata modu sudur" demek mumkun degil. Ilginc olan, kacirilan
yuzuculerin 14 px'te tam da "koyu bas lekesi" gibi gorunmesi (asagi bak) —
yani bu belirsizlerin bir kismi buyuk olasilikla etiketsiz yuzucu, ama
*kanitlanamiyor*.

**Guven skoru sinyal veriyor:** en yuksek ortalama guven gercek ama etiketsiz
teknelerde (0.72) ve can simitlerinde (0.67); en dusuk kopuk/parilti (0.33) ve
belirsizlerde (0.35). Model kendi hatalarini bir olcude "biliyor" — bu, genel
ECE'nin 0.137 olmasina ragmen guven skorunun tamamen bilgisiz olmadigini
gosteriyor.

---

## B. Yanlis negatifler — kacirilan `swimmer`'lar

Incelenen ornek: **n = 48** (240 kaydedilmis swimmer kirpintisi icinden;
toplam swimmer FN sayisi cok daha yuksek). Iki uctan secildi: ~31 px'lik
en buyuk kacirmalar ve 14 px'lik en kucukler.

| Kategori | n | % | ort. kenar (px) |
|---|---|---|---|
| `sadece_bas` (sadece bas/omuz suyun ustunde) | 18 | 37.5 | 14.0 |
| `net_izole` (acik secik, tek basina, engelsiz) | 8 | 16.7 | 31.3 |
| `dusuk_kontrast` (koyu mayo/dalgic elbisesi, koyu su) | 6 | 12.5 | 31.0 |
| `dusuk_isik` (alacakaranlik/gece karesi) | 5 | 10.4 | 14.0 |
| `kopuk_icinde` (kirilan dalga / sicrama icinde) | 5 | 10.4 | 27.8 |
| `kume_halinde` (bitisik/ic ice birden fazla yuzucu) | 4 | 8.3 | 31.0 |
| `batik_golge` (govde su altinda, bulanik golge) | 2 | 4.2 | 31.0 |

### Kacirilanlarin ortak ozelligi

**Baskin desen: gorunur imza kuculuyor, nesne degil.** En kalabalik kategori
`sadece_bas` (%37.5) — yuzucunun govdesi su altinda, gorunen tek sey bir kirmizi
bone ya da koyu bir bas. Kutu 14 px olsa bile *ayirt edici piksel* bunun cok
altinda. 640'a olceklendiginde bu ~2 px'e iniyor, yani ag icin fiilen yok.
Bu, "kucuk nesne zor" ifadesinin bu veri setindeki somut mekanizmasi:
**sorun nesnenin kucuklugu degil, suyun govdeyi gizleyip geriye tek bir nokta
birakmasi.**

**Ikinci desen: kontrast, boyuttan bagimsiz olarak belirleyici.**
`dusuk_kontrast` + `dusuk_isik` + `batik_golge` toplami %27. Bunlarin bir kismi
31 px — yani `<16` bandinda degil, buna ragmen kaciriliyor. Yalnizca olcek
odakli bir aciklama bu grubu kacirirdi.

**Rahatsiz edici desen: `net_izole` %16.7.** En buyuk kacirmalar arasinda,
bos suda, turuncu can yelekli, hicbir engeli olmayan, gozle aninda secilen
yuzuculer var. Bunlarin kacirilmasi ne olcekle ne kontrastla aciklanabiliyor.
En olasi aciklama 10 epoch'luk pilotun yetersiz egitimi (`close_mosaic=10`
yuzunden mosaic hic calismadi). **1280 tam kosusundan sonra bu kategorinin
kaybolmasi bekleniyor — kaybolmazsa daha derin bir sorun var. Test edilecek
somut tahmin budur.**

**`kume_halinde` (%8.3) NMS suphesi doguruyor.** Bitisik yuzen iki-uc kisi
tek kutuya bastiriliyor olabilir. Su an olculmedi: kume icindeki FN'lerin
NMS IoU esigine duyarliligi taranmadi.

---

## C. Bu incelemenin acmadigi sorular

- Sayilar orneklem icin. `background` FP'lerin tamaminda (225) ve tum FN
  kumesinde ayni dagilim gecerli mi — **olculmedi**.
- Anotasyon eksikligi sistematik mi (belirli videolar/droneler mi)?
  **Olculmedi.** `manifest.csv`'deki `image_id`'ler `source.video` ile
  birlestirilerek bakilabilir.
- Acik kelime dagarcikli modelin (`baseline_b_*`) kirpintilari **gorsel olarak
  incelenmedi**. Onlarin baskin hata modu zaten sinif karisikligi ve o, kutu
  degil etiket sorunu oldugu icin gorsel taksonomi farkli bir yontem ister.
- `kume_halinde` FN'lerin NMS esigine duyarliligi **olculmedi**.
