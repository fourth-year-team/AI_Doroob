import subprocess
import os

os.environ["FLAGS_enable_ir_optim"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

print("Starting full pipeline...")

BASE = os.getcwd()

tf_python = os.path.join(BASE, "venvs", "venv_tf", "Scripts", "python.exe")
paddle_python = os.path.join(BASE, "venvs", "venv_paddle", "Scripts", "python.exe")

# Step 1: Vehicle detection
print("Step 1: Detection")
subprocess.run([tf_python, "src/pipelines/vehicle-detect.py"], check=True)

# Step 2: Plate detection
print("Step 2: Plate detection")
subprocess.run([tf_python, "src/pipelines/plate_crop.py"], check=True)

# Step 3: OCR
print("Step 3: OCR")
subprocess.run([paddle_python, "src/pipelines/ocr.py"], check=True)

# Step 4: Brand prediction
print("Step 4: Brand prediction")
subprocess.run([tf_python, "src/pipelines/brand-predict.py"], check=True)

# Step 5: Color prediction
print("Step 5: Color prediction")
subprocess.run([tf_python, "src/pipelines/color-predict.py"], check=True)

print("✅ Pipeline finished")