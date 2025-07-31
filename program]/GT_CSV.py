import os
import pandas as pd

# Ganti ini ke folder sesuai lokasi kamu
label_folder = "Indonesian License Plate Dataset/labelswithLP/test/"

# Format output CSV
data = []

# Loop semua file txt
for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        file_path = os.path.join(label_folder, file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Ambil hanya bagian terakhir dari tiap baris (teks plat nomor)
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:
                    plate = parts[-1]
                    # Ganti .txt ke .jpg sesuai nama gambar
                    img_name = file.replace(".txt", ".jpg")
                    data.append([img_name, plate])

# Simpan ke ground_truth.csv
df = pd.DataFrame(data, columns=["image", "ground_truth"])
df.to_csv("ground_truth.csv", index=False)

print(f"Berhasil generate {len(df)} data ke ground_truth.csv")
