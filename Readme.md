# 🚗 Araç Tespiti: YOLOv8 ile Video Üzerinde Nesne Tanıma

Bu proje, Ultralytics YOLOv8 modeli ile video içeriğinde **araç tespiti** yapmayı amaçlamaktadır. Eğitim, tahmin ve çıktıların kaydedilmesi süreçleri tamamen otomatikleştirilmiştir.

---

## 🛠️ Kullanılan Teknolojiler

- Python
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Google Colab
- OpenCV (dolaylı olarak)
- Matplotlib, IPython (görselleştirme)

---

## 🧠 Model Eğitimi

Eğitim aşamasında önceden tanımlanmış bir `.yaml` dosyası kullanılır. Bu dosya eğitim ve doğrulama veri yollarını, sınıf sayısını ve sınıf isimlerini içerir.

```python
from ultralytics import YOLO

# YOLOv8 Nano modelini seçiyoruz
model = YOLO("yolov8n.pt")

# Eğitimi başlatıyoruz
model.train(data="/content/vehicle_dataset/dataset/vehicle_dataset/data.yaml", epochs=50)
```
---

## 🔍 Araç Tespiti
```python
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/detect/train/weights/best.pt")

# Araçları videoda tespit et ve kaydet
results = model.predict(source="4.mp4", save=True, conf=0.25)
```
## 💾 Çıktıyı Kaydetme ve İndirme
```python
import shutil
from google.colab import files

# Çıktıyı kaydet
shutil.move("runs/detect/predict/testvideo.mp4", "output.mp4")

# İndir
files.download("output.mp4")
```
