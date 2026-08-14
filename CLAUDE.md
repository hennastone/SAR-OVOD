Yüksek lisans tezim için İHA tabanlı arama-kurtarma (SAR) görüntülerinde nesne 
tespiti üzerine çalışıyorum. Şu anda tez konusunu kesinleştirme aşamasındayım ve 
bir keşif deneyi yapıyorum.

Amaç: SeaDronesSee veri seti üzerinde (a) klasik kapalı-küme bir detector ile 
(b) açık kelime dağarcıklı bir detector'ün küçük hedeflerdeki davranışını 
karşılaştırmak. Özellikle merak ettiğim: baskın hata modu yanlış pozitifler mi 
(köpük, yansıma, kaya, şamandıra insana benziyor), yoksa sınıf karışıklığı mı; 
ve hedef boyutu küçüldükçe performans nasıl bozuluyor.

Ortam: Ubuntu, tek NVIDIA GPU, PyTorch, ultralytics. Conda kullanıyorum.
Deneyim: derin öğrenme ve bilgisayarlı görüde deneyimliyim (CNN, ViT, GAN, 
diffusion, multimodal modeller). Temel kavramları açıklamana gerek yok.

Nasıl çalışmanı istiyorum:
- Kodu tek dev script yerine modüler ve çalıştırılabilir parçalar halinde ver
- Her scriptin ne ürettiğini kısaca söyle, uzun açıklama yazma
- Emin olmadığın API/parametre varsa uydurma, "doğrulaman lazım" de
- Sade ve doğrudan konuş; gereksiz övgü ve dolgu cümle istemiyorum
- Bir yaklaşımın işe yaramayacağını düşünüyorsan söyle

## Dizin yapısı
- Veri seti: ./data/images
    - Annotations: ./data/annotations
- Scriptler: ./scripts/
- Çıktılar (grafik, tablo, hata kırpıntıları): ./outputs/
- Veri klasörünü asla değiştirme, sadece oku. Yeni dosyaları outputs/ altına yaz.

## Bulgu günlüğü
Her anlamlı çalıştırmadan sonra ./notes/findings.md dosyasına ekleme yap.
Format (en yeniler üstte):

## YYYY-MM-DD — <ne yapıldı>
- Çalıştırılan: <script + parametreler>
- Sonuç: <sayılar, tablolar>
- Dikkat çeken: <beklenmedik olan ne>
- Sonraki soru: <bu neyi açtı>

Yorumlarını buraya yaz; sohbette bırakma. Tahminlerini ölçümlerden ayır.