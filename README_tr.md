>Dil: Türkçe 🇹🇷<br>

İngilizce için: [English](README.md)
![Header](https://ishan-rest.vercel.app/svg/banner/creative/ANN-Siniflandirma)

# Tanım
 Bu proje, ANN (Artificial Neural Network, Yapay Sinir Ağları)  kullanan ve MNIST veri seti ile eğitilmiş bir modelin; <br>
 * Eğitim süreci detaylarını<br>
 * Çalışma mantığını
 * Model çıktılarını 
 * Performans değerlendirmesini
 * Kurulum ve kullanımını<br>
 
 İçermektedir.
 

### ANN (Artificial Neural Network) nedir?
* ANN (Yapay Sinir Ağları), insan beyninin çalışma şeklinden ve yapısından esinlenerek geliştirilmiş, bilgisayarın hatalarından öğrenerek karar vermesini sağlayan, yapay zekanın temel yapı taşıdır. Karar verirken insan beyninin nöron yapısına benzer bir sistem kullanır. Girdiler (Inputs) her bir nörona ağırlıklar (weights) ile çarpılarak bağlanır ve o nörona ait sabit değerle (bias) toplanarak nöronun sayısal bir değer üretmesi sağlanır. Katmanlar (Fully Connected Layer) boyunca ilerleyen bağlantılar (Forward Propagation), son katmanda (Output) bir çıktı üretir. Finaldeki bu sayı Sigmoid fonksiyonundan geçirilerek bir olasılık değerine dönüşür ve ait olduğu sınıfın olasılığını temsil eder. Model daha sonra olasılıklara ve gerçek sınıfa bakarak hata (loss) hesaplar (Cross-Entropy Loss) ve bu hata değerini ağ boyunca geriye doğru (chain-rule) dağıtır (Backward Propagation). Gradyanlar hesaplanır ve ağırlıklar güncellenir (learning). Böylece model en doğru ayarlamalarını yapmış ve öğrenmiş olur.

#### Yapay Sinir Ağı öğrenme adımları
- İleri yayılım (Forward Propagation)
- Sigmoid Fonksiyonu
- Hata hesabı (Loss Calculation)
- Gradyan hesaplama (Gradiant Calculation)
- Ağırlık güncelleme (Weight Update)<br>

>[!NOTE]
> Son 2 adım geri yayılım (Backward Propagation) aşamasını temsil eder.


## Eğitim süreci detayları

**Model**: `Artificial Neural Network (ANN)` <br>
**Görev**: `Rakam tespiti` <br>
**Epoch sayısı**:`10` <br>
**Veri seti**: [MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) <br>
**Gİrdi görseli boyutu**: `28x28` <br>
**Tekonloji yığını**: ![PyTorch](https://img.shields.io/badge/PyTorch-black?style=flat-square&logo=pytorch&logoColor=E33512
),![OpenCV](https://img.shields.io/badge/OpenCV-black?style=flat-square&logo=opencv&logoColor=12ABE3
),![Python](https://img.shields.io/badge/Python-black?style=flat-square&logo=python&logoColor=AAEB49
)

## Model çıktısı
- Model 10 epoch boyunca eğitildikten sonra, modele daha önce hiç görmediği bir görsel verilmiş ve modelin tahmini görsel üzerine yazdırılmıştır.

|Test Görseli   | Model Tahmini    |
| --- | ---- |
| ![Test](assets/testing_image.png)   |![Tahmin](assets/model-prediction.png)  | 
>[!IMPORTANT]
> Eğitilen model, tahmin yapabilmek için girdi görselini (Input) çok boyutlu matris halinde değil tek boyutlu vektör halinde alır. Bu yüzden eğer modele bir görsel verilecekse, görsel aşağıdaki işlemlerden geçmelidir:
```python 
#image pre-process phase
img=cv.imread("testing_image.png",cv.IMREAD_GRAYSCALE)
img_resized=cv.resize(img,(28,28))
img_normalized=img_resized/255.0
img_tensor=torch.tensor(img_normalized, dtype=torch.float32)
img_vector=img_tensor.view(1,784)
```
## Performans değerlendirmesi
- Model, 10 epoch boyunca süren eğitim sürecinde giderek daha az hata değeri (Loss value) üretmiştir. Eğitim boyunca her bir epoch aşamasında üretilen hata değerleri terminale yazdırılmıştır.

![Hata değerleri](assets/training-process.png "Modelin hata değerleri")
- Görüldüğü üzere model birinci epochta 0.39 hata değeri üretirken, son epocha gelindiğinde bu hata değerini 0.05'e kadar düşürmüş ve öğrenme sürecini tamamlamıştır.<br>

- Son satırdaki **Model Accuracy** değeri modelin test veri seti üzerindeki başarı oranını temsil etmektedir.

>[!IMPORTANT]
>Epoch sayısı model eğitiminde bir hiperparametredir. Bu parametrenin değeri belirlenirken terminal üzerindeki hata değerleri izlenmeli ve değer bir noktadan sonra azalmıyor ise eğitim orada sonlandırılmalıdır. Aksi takdirde *Overfitting* (aşırı öğrenme) gerçekleşir. <br>

- Aşağıdaki grafikte modelin hata değerinin zaman içindeki düşüşü gösterilmiştir. 
<div align="center">
  <h4>HATA DEĞERİ GRAFİĞİ</h4>
  <img src="assets/training_loss_graph.png" alt="Zaman içinde hata değerindeki düşüş">
</div>

## Kurulum ve kullanım
Modeli yerel bilgisayarınıza yüklemek ve test etmek için aşağıdaki adımları takip edin: <br>

1-
```bash
#Depoyu yerel bilgisayarınza kopyalayın.
git clone https://github.com/halileroglu711/ANN-classification-mnist.git 
```
2-
```bash
#Proje klasörüne girin.
cd ANN-classification-mnist
```
3-
```bash
#Gerekli kütüphaneleri yükleyin.
pip install -r requirements.txt
```
4-
```bash
#Eğitimi başlatın.
python main.py
```
5-
```bash
#Örnek görselle modeli test edin.
python test.py
```
## Proje dosya yapısı
```text
ANN-classification-mnist/
│
├── assets/
│   ├── model-prediction.png 
│   ├── testing_image.png
│   ├── training_loss_graph.png
|   └── training-process.png
| 
├── weights/
|    └── best.pt
│
├── .gitignore
├── classes.py
├── functions.py
├── main.py
├── README_tr.md
├── README.md
├── requirements.txt
└── test.py
```

### ©️ Lisans
Bu proje MIT lisansı ile lisanslıdır. <br>
Daha fazla detay için kontrol edin: 🔍[Lisans](LICENSE)

## 📬 İletişim
- Herhangi bir hatam varsa bana bildirin🙋. Katkılarınızı bekliyorum 🙂.Bana buradan ulaşabilirsiniz:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/halil-eroğlu-5505783a1)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/halileroglu711)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:halileroglu711@gmail.com)



