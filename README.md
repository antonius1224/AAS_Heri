# AAS_Heri
# OCR Plat Nomor Indonesia dengan VLM

Saya melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan Indonesia menggunakan Visual Language Model (VLM) dengan mode Gemma3-4 yang dijalankan lewat **LMStudio**, dan diintegrasikan dengan Python.

## Dataset

- Sumber: [Kaggle - Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)
- Folder yang digunakan: `images/test` (gambar) dan `labelswithLP/test` (label plat nomor)

## Cara Menjalankan Proyek

### 1. Siapkan Lingkungan
Install pustaka yang dibutuhkan:
```bash
pip install openai pillow pandas python-Levenshtein
```

### 2. Jalankan LMStudio
- Buka aplikasi **LMStudio**
- Jalankan model multimodal `Gemma 3-4` 
- Pastikan LMStudio aktif di `http://localhost:1234`

### 3. Siapkan Folder
Struktur folder yang disarankan:
```
project/
├── images/test/             ← Gambar plat nomor
├── labelswithLP/test/       ← File .txt label (isi plat nomor)
├── ground_truth.csv         ← Dihasilkan dari label
├── ocr_vlm.py               ← Script Python utama
├── hasil_prediksi.csv       ← Output hasil OCR
├── README.md
```

### 4. Generate Ground Truth
Jalankan script berikut untuk membuat `ground_truth.csv`:
```python
import os
import pandas as pd

label_folder = "labelswithLP/test/"
data = []

for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        with open(os.path.join(label_folder, file), "r") as f:
            for line in f.readlines():
                plate = line.strip().split()[-1]
                img_name = file.replace(".txt", ".jpg")
                data.append([img_name, plate])

df = pd.DataFrame(data, columns=["image", "ground_truth"])
df.to_csv("ground_truth.csv", index=False)
```

### 5. Jalankan Inference dan Evaluasi
Jalankan file utama:
```bash
python ocr_vlm.py
```
Script ini akan:
- Mengirim gambar ke LMStudio
- Mengambil hasil OCR
- Menghitung **CER (Character Error Rate)**
- Menyimpan ke `hasil_prediksi.csv`

## Evaluasi

Evaluasi dilakukan dengan menghitung CER (Character Error Rate):

```
CER = (S + D + I) / N
```

- **S**: Substitusi karakter
- **D**: Karakter dihapus
- **I**: Karakter ditambahkan
- **N**: Panjang ground truth

Semakin kecil CER, semakin baik hasil OCR.

## Hasil

| Gambar        | Ground Truth | Prediction  | CER     |
|---------------|--------------|-------------|---------|
| IMG_001.jpg   | B1234XYZ     | B1234XYZ    | 0.00    |
| IMG_014.jpg   | D5678DEF     | D5679DEE    | 0.38    |

##  Video Penjelasan

[Link YouTube Penjelasan Proyek](https://youtu.be/d6P-yrpgShA)

## link Referensi

- [LMStudio Docs](https://lmstudio.ai/docs/python/llm-prediction/image-input)
- [Kaggle Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)

##  Catatan

- Semua kode dijalankan secara **lokal tanpa internet**
- Cocok untuk sistem AI embedded, edge, atau tanpa koneksi cloud
