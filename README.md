# 🎭 Artificial Intelligence and Image Based Face Expression Recognition System
### (Yapay Zeka ve Görüntü Tabanlı Yüz İfade Tanıma Sistemi)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](#)

Bu proje, **Ankara Üniversitesi Bilgisayar Mühendisliği Bölümü** bitirme projesi kapsamında geliştirilmiş; derin öğrenme (CNN) mimarisi kullanarak insan yüzlerinden gerçek zamanlı duygu analizi yapan bir sistemdir.

## 📚 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [🗂 Proje Yapısı](#-proje-yapısı)
- [📊 Veri Seti Bilgisi](#-veri-seti-bilgisi)
- [🛠 Geliştirme Süreci](#-geliştirme-süreci)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)
- [📧 İletişim](#-iletişim)
- [📜 Lisans](#-lisans)

---

## 💻 Proje Hakkında
Bu çalışma, görüntü işleme ve yapay zeka tekniklerini birleştirerek bireylerin duygusal durumlarını anlık olarak tespit eder. **FER-2013** veri seti üzerinde eğitilen model, kamera görüntüsü üzerinden aldığı yüz verilerini analiz ederek 7 farklı duygu kategorisinde sınıflandırma yapar.

* **Geliştiriciler:** Haluk Can SARIÖZ & Mesut ÖZLAHLAN
* **Danışman:** Arş. Gör. İrem ÜLKÜ
* **Kurum:** Ankara Üniversitesi Mühendislik Fakültesi, Bilgisayar Mühendisliği Bölümü

---

## 🌟 Özellikler
* **Gerçek Zamanlı Tespit:** Düşük gecikme ile canlı video akışında anlık analiz.
* **7 Temel Duygu:** Kızgın, Tiksindirici, Korkmuş, Mutlu, Nötr, Üzgün ve Şaşkın.
* **Görsel Geribildirim:** Yüz tespiti için sınırlayıcı kutular (bounding box) ve canlı duygu etiketleri.
* **Akıllı Mimari:** Optimize edilmiş Convolutional Neural Networks (CNN) yapısı.

---

## 🛠 Kullanılan Teknolojiler
* **OpenCV:** Kamera erişimi ve Haar Cascade ile yüz tespiti.
* **Keras & TensorFlow:** Derin öğrenme modelinin eğitimi ve tahmini.
* **NumPy & Pandas:** Veri yönetimi ve matris operasyonları.
* **Matplotlib:** Eğitim süreci analiz grafiklerinin oluşturulması.

---

## 🚀 Kurulum ve Kullanım

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/halukcansarioz/Emotion-Detection.git
```

### 2. Proje Dizinine Gidin
```bash
cd Emotion-Detection
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Başlatın
Uygulamayı başlatmak için ana dosyayı çalıştırın:
```bash
python main.py
```
*Not: Kameranızın başka bir uygulama tarafından kullanılmadığından emin olun.*

---

## 🗂 Proje Yapısı
```text
Emotion-Detection/
├── data/
│   └── haarcascade_frontalface_default.xml  # Yüz tespiti XML dosyası
├── model/
│   └── emotion_model.h5                     # Eğitilmiş model ağırlıkları
├── src/                                     # Yardımcı fonksiyonlar
├── TrainEmotionDetector.py                  # Model eğitim betiği
├── main.py                                  # Uygulama giriş noktası
├── requirements.txt                         # Kütüphane listesi
└── README.md                                # Proje dökümantasyonu
```

---

## 📊 Veri Seti Bilgisi
Eğitim sürecinde **FER-2013** veri seti kullanılmıştır.
* **İçerik:** 48x48 piksel boyutunda gri tonlamalı yüz görüntüleri.
* **Kapsam:** Yaklaşık 35.000 veri ve 7 farklı duygu sınıfı.

---

## 🛠 Geliştirme Süreci

### 1. Forklama
Öncelikle depoyu kendi GitHub hesabınıza fork'layın.

### 2. Yeni Dal (Branch) Oluşturma
Her yeni çalışma için yeni bir dal oluşturmak düzeni korur:
```bash
git checkout -b ozellik/yeni-model-iyilestirmesi
```

### 3. Kodları Gönderme (Push)
Değişikliklerinizi yaptıktan sonra GitHub'a göndermek için:
```bash
git push origin ozellik/yeni-model-iyilestirmesi
```

---

## 🤝 Katkıda Bulunma
1. Bu depoyu **Fork**'layın.
2. Bir **Branch** oluşturun (`git checkout -b feature/YeniOzellik`).
3. Değişikliklerinizi yapın ve **Commit** edin (`git commit -m 'Ekleme: Yeni özellik'`).
4. Kodlarınızı **Push**'layın (`git push origin feature/YeniOzellik`).
5. Bir **Pull Request** açın.

---

## 📧 İletişim
**Haluk Can Sarıöz** - [GitHub Profilim](https://github.com/halukcansarioz)  
**Mesut ÖZLAHLAN** - [GitHub Profili](https://github.com/mesutozlahlan)  
**Proje Linki:** [https://github.com/halukcansarioz/Emotion-Detection](https://github.com/halukcansarioz/Emotion-Detection)

---

## 📜 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
