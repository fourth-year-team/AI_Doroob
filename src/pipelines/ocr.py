import os
import re
import cv2
import numpy as np

# 1. MONKEY-PATCH TO FIX WINDOWS ONEDNN BUG
import paddle.inference as paddle_infer
old_config_init = paddle_infer.Config.__init__
def patched_config_init(self, *args, **kwargs):
    old_config_init(self, *args, **kwargs)
    self.switch_ir_optim(False)
    self.disable_mkldnn()
paddle_infer.Config.__init__ = patched_config_init

from paddleocr import PaddleOCR

# ---------------- CONFIGURATION ----------------
PREPROCESS_MODE = "light"  # Options: "light", "medium"

# ---------------- INITIALIZATION & PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(BASE_DIR, 'inference_model')
dict_path = os.path.join(BASE_DIR, 'plate_dict.txt')

print(f"[INFO] Loading Custom PaddleOCR Model from: {model_dir}")
print(f"[INFO] Using Character Dictionary from: {dict_path}")

ocr = PaddleOCR(
    rec_model_dir=model_dir,
    rec_char_dict_path=dict_path,
    use_angle_cls=False,  
    lang='en', 
    det=False,  
    rec=True,
    ocr_version='PP-OCRv3',
    enable_mkldnn=False,
    show_log=False
)

input_root = os.path.join(BASE_DIR, "outputs")
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------- HELPER FUNCTIONS ----------------
def extract_text_safely(ocr_result):
    """ Safely extracts text from PaddleOCR output structure """
    if not ocr_result or not isinstance(ocr_result, list):
        return ""
    first_elem = ocr_result[0]
    if isinstance(first_elem, list) and len(first_elem) > 0:
        if isinstance(first_elem[0], (list, tuple)):
            return str(first_elem[0][0])
        elif isinstance(first_elem[0], str):
            return str(first_elem[0])
    return ""

def preprocess_plate_image(img, mode="light"):
    """ Appliess light preprocessing to preserve character embeddings """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if mode == "light":
        gray = cv2.bilateralFilter(gray, 5, 40, 40)
    elif mode == "medium":
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def process_and_recognize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "No_Plate_Detected"
        
    orig_h, orig_w = img.shape[:2]
    
    # Trim 2% off borders to eliminate license plate frame noise
    trim_h = int(orig_h * 0.02)
    trim_w = int(orig_w * 0.02)
    cropped_base = img[trim_h:orig_h-trim_h, trim_w:orig_w-trim_w]
    
    # Apply balanced preprocessing
    processed_input = preprocess_plate_image(cropped_base, mode=PREPROCESS_MODE)
    
    # Save processed image for visual inspection
    folder_path = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(folder_path, "plate_processed.jpg"), processed_input)
    
    # Run OCR inference
    res = ocr.ocr(processed_input, det=False, cls=False)
    text_plate = extract_text_safely(res)
    
    # Extract only numbers and normalize Hindi digits to standard Arabic digits
    hindi_to_arabic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    digits = "".join(re.findall(r'[0-9٠-٩]', text_plate)).translate(hindi_to_arabic)
    
    # Post-OCR Rule Engine for single-row rectangular plates
    if len(digits) == 6:
        return digits  # Old plate format: XXXXXX
    elif len(digits) >= 7:
        actual_digits = digits[:7]  # Restrict to first 7 digits to fix hallucinations
        return f"{actual_digits[:2]}-{actual_digits[2:]}"  # New rectangular format: XX-XXXXX
            
    return digits if digits else "No_Plate_Detected"

# ---------------- PROCESS ALL IMAGES (LOOP) ----------------
if os.path.exists(input_root):
    for img_name in os.listdir(input_root):
        img_folder = os.path.join(input_root, img_name)
        if not os.path.isdir(img_folder):
            continue

        for car_name in os.listdir(img_folder):
            car_folder = os.path.join(img_folder, car_name)
            if not os.path.isdir(car_folder):
                continue

            car_image_path = None
            for f in os.listdir(car_folder):
                fl = f.lower()
                if fl.endswith(valid_ext) and "plate" in fl and "processed" not in fl:
                    car_image_path = os.path.join(car_folder, f)
                    break

            if car_image_path is None:
                continue

            plate_text = process_and_recognize(car_image_path)
            print(f"🎯 [OCR] {img_name}/{car_name} -> {plate_text}")

            text_file_path = os.path.join(car_folder, f"{car_name}.txt")
            with open(text_file_path, "a", encoding="utf-8") as f:
                f.write(f"Plate Text: {plate_text}\n")