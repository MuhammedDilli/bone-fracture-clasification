ViT - Kemik görüntü sınıflandırması ve kırık görüntü tespiti 

# Giriş
Günümüzde geniş bir kullanıcı kitlesi tarafından benimsenen internet platformları ve dijital sağlık teknolojileri, tıbbi teşhis ve tedavi süreçlerinde önemli bir rol oynamaktadır. Özellikle son yıllarda yapay zeka ve derin öğrenme tekniklerinin gelişimi, tıbbi görüntü analizinde yeni yöntemlerin kullanılmasına olanak sağlamıştır. Bu projede, kemik kırığı sınıflandırılması amacıyla Vision Transformer (ViT) modeli kullanılarak yapılan çalışmanın bulguları sunulmaktadır.
Vision Transformer (ViT) modeli, görüntü işlemenin temelini oluşturan yapay sinir ağları ve katmanlarının transformer mimarisi ile birleşiminden oluşur. Bu model, kemik kırığı görüntülerini analiz ederek sınıflandırmayı hedeflemektedir. ViT modeli, kemik kırıklarının doğru bir şekilde tespit edilmesi ve sınıflandırılmasında etkili bir rol oynar. Model, eğitim süreci boyunca verilen veri setindeki kırık ve kırık olmayan kemik görüntülerini analiz ederek öğrenir. Bu öğrenme sürecinde, kemik yapısındaki ince detaylar ve kırık bölgelerinin özellikleri belirlenir. Model, bu özellikleri kullanarak kırık ve sağlam kemikler arasındaki farkları anlamayı öğrenir.

# Augmentation
Veri setinde sınıf dengesizliği olduğundan data augmentation(veri artırma işlemleri yapılmıştır.)
popüler veri artırma teknikleri kullanılmıştır.
# Dataset
orjinal veri seti:
https://figshare.com/articles/dataset/The_dataset/22363012/6#:~:text=FracAtlas%20is%20a%20musculoskeletal%20bone,freely%20available%20for%20any%20purpose.


Veri dengesizliği giderilmiş (augmentation uygulanmış) veri seti :

https://drive.google.com/drive/folders/1NE0g2E59HRR8-kuToFSlJst1KZKVH6fW?usp=sharing
