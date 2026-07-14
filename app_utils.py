import sys
import os
import json
import base64

# Paths
BASE_DIR = os.getcwd()
TF_PYTHON = os.path.join(BASE_DIR, "venvs", "venv_tf", "Scripts", "python.exe")
SCRIPT_DIR = os.path.join(BASE_DIR, "src", "pipelines")

def detect_vehicles(image_path, output_dir):
    """Run vehicle detection via subprocess in venv_tf."""
    import subprocess, tempfile
    tmp = tempfile.mkdtemp(dir=output_dir)
    # Call a small script that does YOLO detection and returns plate crops
    script = f"""
import sys, os, json, base64, cv2, numpy as np
sys.path.insert(0, {repr(SCRIPT_DIR)})
from ultralytics import YOLO

img = cv2.imread({repr(image_path)})
if img is None:
    print(json.dumps({{"error": "cannot read image"}}))
    sys.exit(1)

vehicle_model = YOLO(r"{os.path.join(BASE_DIR, 'models', 'yolov8m.pt')}")
plate_model = YOLO(r"{os.path.join(BASE_DIR, 'models', 'license_plate_detector.pt')}")

vehicles = vehicle_model(img, verbose=False)
results = []
for r in vehicles:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = vehicle_model.names[cls]
        if label not in ("car", "truck", "bus"):
            continue
        vcrop = img[y1:y2, x1:x2]
        if vcrop.size == 0:
            continue
        plates = plate_model(vcrop, verbose=False)
        for pr in plates:
            for pb in pr.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                pcrop = vcrop[py1:py2, px1:px2]
                if pcrop.size == 0:
                    continue
                fname = f"plate_{{x1}}_{{y1}}.jpg"
                fpath = os.path.join({repr(tmp)}, fname)
                cv2.imwrite(fpath, pcrop)
                results.append({{
                    "vehicle_bbox": [x1, y1, x2, y2],
                    "plate_bbox_abs": [x1+px1, y1+py1, x1+px2, y1+py2],
                    "plate_crop_path": fpath,
                    "label": label
                }})
print(json.dumps(results))
"""
    code = base64.b64encode(script.encode()).decode()
    result = subprocess.run(
        [TF_PYTHON, "-c", f"exec(__import__('base64').b64decode({repr(code)}).decode())"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Detection failed: {result.stderr}")
    return json.loads(result.stdout.strip()), tmp
