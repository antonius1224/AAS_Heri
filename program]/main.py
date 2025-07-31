import os
import csv
import json
import base64
import difflib
import requests
from tqdm import tqdm
import lmstudio as lms

# Inisialisasi LMStudio Client 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# AAS\Indonesian License Plate Dataset\images\test
DATASET_DIR = r"C:\Users\hp\Documents\semester_6\RE604\AAS\Indonesian License Plate Dataset\images\test"
GROUND_TRUTH_CSV = "ground_truth.csv"
OUTPUT_CSV = "ocr_result.csv"
"""
Server bisa menggunakan
* 127.0.0.1
* localhost
"""
SERVER_API_HOST = "localhost:1234"
SERVER_URL = "http://localhost:1234/v1/chat/completions"
VLM_MODEL_NAME = "google/gemma-3-4b"

#Inisialisasi LMStudio Client 

lms.configure_default_client(SERVER_API_HOST)
model = lms.llm(VLM_MODEL_NAME)

#FUNGSI UNTUK MENGHITUNG CER 

def calculate_cer(ground_truth, prediction):
    matcher = difflib.SequenceMatcher(None, ground_truth, prediction)
    opcodes = matcher.get_opcodes
    S = D = I = 0
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == 'replace':
            S += max(i2 - i1, j2 - j1)
        elif opcode == 'delete':
            D += i2 - i1
        elif opcode == 'insert':
            I += j2 - j1
    N = len(ground_truth)
    CER = round((S + D + I) / N, 4)
    formula = f"CER = ({S}+{D}+{I}/{N})"
    return  f"{CER * 100:.2f}%"
 

# MASUKKAN GROUND TRUTH 

def load_ground_truth(csv_path):
    """Load ground truth from CSV into dictionary"""
    gt_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_dict[row["image"]] = row["ground_truth"]
    return gt_dict
#ENCODE GAMBAR KE BASE 64 
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# FUNGSI UNTUK MEMPREDIKSI OCR DENGAN VLM 
def ocr_image(image_path):
    try:
        image_base64 = encode_image_to_base64(image_path)
        image_url = f"data:image/jpeg;base64,{image_base64}"

        with open("prompt.json","r") as f:
            payload = json.load(f)

        payload["messages"][0]["content"][0]["image_url"]["url"] = image_url
        
        response = requests.post(SERVER_URL,json=payload)
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]
        return result.strip().replace(" ", "").upper()
    
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return "ERROR"


# Main 
# Main 

def main():
    ground_truths = load_ground_truth(GROUND_TRUTH_CSV)
    if not ground_truths:
        print("No ground truth data loaded. Exiting.")
        return
    image_files = list(ground_truths.keys())

    total_samples = 0
    correct_predictions = 0
    
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "ground_truth", "prediction", "CER_score"])

        for image_name in tqdm(image_files, desc="Processing Plates"):
            image_path = os.path.join(DATASET_DIR, image_name)
            gt_text =ground_truths[image_name]

            pred_text = ocr_image(image_path)

            cer = calculate_cer(gt_text, pred_text)

            writer.writerow([image_name, gt_text, pred_text, cer])

            print(f"\n{image_name}=>GT:{gt_text} | pred:{pred_text} | CER{cer}\n")

            # Hitung akurasi
            total_samples += 1
            if gt_text == pred_text:
                correct_predictions += 1

    # Tampilkan dan simpan akurasi
    accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n OCR selesai. Hasil disimpan di {OUTPUT_CSV}")
    print(f"Akurasi: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    # Simpan akurasi ke file
    with open("accuracy.txt", "w") as f:
        f.write(f"Akurasi: {accuracy:.2f}% ({correct_predictions}/{total_samples})\n")

if __name__ == "__main__":
    main()