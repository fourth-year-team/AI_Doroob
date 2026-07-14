import os
import sys
import re
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="AI Doroob - Plate Recognition",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- PATH CONFIG ----------
BASE_DIR = os.getcwd()
TF_PYTHON = os.path.join(BASE_DIR, "venvs", "venv_tf", "Scripts", "python.exe")
PADDLE_PYTHON = os.path.join(BASE_DIR, "venvs", "venv_paddle", "Scripts", "python.exe")
PIPELINES_DIR = os.path.join(BASE_DIR, "src", "pipelines")

# ---------- STYLING ----------
st.markdown("""
<style>
    .main > div { padding-bottom: 2rem; }
    .stApp header { display: none; }
    .stFileUploader > div > div { padding: 2rem; border: 2px dashed #4a90d9; border-radius: 16px; background: #0a1628; }
    .stFileUploader > div > div:hover { border-color: #6ab0ff; background: #0e1e38; }
    div[data-testid="stFileUploaderDropzone"] > div > div { color: #b0c4e8 !important; }
    div[data-testid="stFileUploaderDropzone"] > div > div > small { color: #7a9ec7 !important; }
    .css-1v0mbdj { margin-top: -1rem; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(26,115,232,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e88e5, #1565c0);
        box-shadow: 0 6px 20px rgba(26,115,232,0.5);
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }
    .result-card {
        background: linear-gradient(135deg, #0d2137, #1a3a5c);
        border: 1px solid #2a5a8a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .plate-number {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 6px;
        color: #f0d060;
        text-shadow: 0 2px 8px rgba(240,208,96,0.3);
        font-family: 'Courier New', monospace;
        padding: 0.5rem 0;
    }
    .badge {
        display: inline-block;
        background: #1a5a3a;
        color: #8cf0b0;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .footer-text {
        text-align: center;
        color: #5a7a9a;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.4s ease-out; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align:center; color:#e8f0fe; font-weight:700;'>🚗 AI Doroob</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8aabca; margin-top:-0.5rem;'>License Plate Recognition</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------- MONKEY-PATCH PADDLE ----------
import paddle.inference as paddle_infer
_old_init = paddle_infer.Config.__init__
def _patched_init(self, *args, **kwargs):
    _old_init(self, *args, **kwargs)
    self.switch_ir_optim(False)
    self.disable_mkldnn()
paddle_infer.Config.__init__ = _patched_init

# ---------- LOAD PADDLEOCR ----------
@st.cache_resource
def _load_ocr():
    from paddleocr import PaddleOCR
    model_dir = os.path.join(BASE_DIR, "inference_model")
    dict_path = os.path.join(BASE_DIR, "plate_dict.txt")
    return PaddleOCR(
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

# ---------- OCR HELPERS ----------
def _extract_safe(r):
    if not r or not isinstance(r, list): return ""
    e = r[0]
    if isinstance(e, list) and e:
        if isinstance(e[0], (list, tuple)): return str(e[0][0])
        if isinstance(e[0], str): return str(e[0])
    return ""

def _preprocess(img, mode="light"):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "light":
        g = cv2.bilateralFilter(g, 5, 40, 40)
    elif mode == "medium":
        g = cv2.bilateralFilter(g, 5, 50, 50)
        g = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def _ocr_plate(ocr, plate_path):
    """Identical logic to src/pipelines/ocr.py process_and_recognize."""
    img = cv2.imread(plate_path)
    if img is None:
        return "No_Plate_Detected"

    h, w = img.shape[:2]
    th, tw = int(h * 0.02), int(w * 0.02)
    cropped = img[th:h-th, tw:w-tw]

    processed = _preprocess(cropped, mode="light")
    folder = os.path.dirname(plate_path)
    cv2.imwrite(os.path.join(folder, "plate_processed.jpg"), processed)

    hindi_map = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

    res_full = ocr.ocr(processed, det=False, cls=False)
    text = _extract_safe(res_full)
    digits = "".join(re.findall(r'[0-9٠-٩]', text)).translate(hindi_map)

    if 7 <= len(digits) < 10:
        d = digits[:7]
        return f"{d[:2]}-{d[2:]}"
    if len(digits) == 6:
        return digits
    if len(digits) >= 10:
        return digits[-6:]

    h2, w2, _ = processed.shape
    right = processed[:, int(w2 * 0.45):w2]
    res_r = ocr.ocr(right, det=False, cls=False)
    text_r = _extract_safe(res_r)
    digits_r = "".join(re.findall(r'[0-9٠-٩]', text_r)).translate(hindi_map)
    if len(digits_r) == 6:
        return digits_r

    return digits if digits else "No_Plate_Detected"

# ---------- SUBPROCESS — RUN THE EXACT PIPELINE ----------
def _run_pipeline(image_path):
    """Copies image to input/, runs the exact batch pipeline, returns plate text."""
    # Copy uploaded file to input folder as the only image
    input_dir = os.path.join(BASE_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    for f in os.listdir(input_dir):
        fp = os.path.join(input_dir, f)
        if os.path.isfile(fp) and f.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            os.remove(fp)
    dst = os.path.join(input_dir, "upload.jpg")
    shutil.copy2(image_path, dst)

    # Clean outputs
    out_root = os.path.join(BASE_DIR, "outputs")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Step 1: Vehicle detection
    r1 = subprocess.run([TF_PYTHON, os.path.join(PIPELINES_DIR, "vehicle-detect.py")],
                        capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env)
    if r1.returncode != 0:
        raise RuntimeError(f"vehicle-detect failed:\n{r1.stderr[:500]}")

    # Step 2: Plate detection
    r2 = subprocess.run([TF_PYTHON, os.path.join(PIPELINES_DIR, "plate_crop.py")],
                        capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env)
    if r2.returncode != 0:
        raise RuntimeError(f"plate_crop failed:\n{r2.stderr[:500]}")

    # Step 3: OCR using the exact same ocr.py script
    r3 = subprocess.run([PADDLE_PYTHON, os.path.join(PIPELINES_DIR, "ocr.py")],
                        capture_output=True, text=True, timeout=120, cwd=BASE_DIR, env=env)
    if r3.returncode != 0:
        raise RuntimeError(f"ocr failed:\n{r3.stderr[:500]}")

    # Parse results from output text files
    results = []
    output_root = os.path.join(BASE_DIR, "outputs")
    if os.path.exists(output_root):
        for img_name in sorted(os.listdir(output_root)):
            img_dir = os.path.join(output_root, img_name)
            if not os.path.isdir(img_dir):
                continue
            for vehicle_folder in sorted(os.listdir(img_dir)):
                vf = os.path.join(img_dir, vehicle_folder)
                if not os.path.isdir(vf):
                    continue
                txt_files = [f for f in os.listdir(vf) if f.endswith(".txt")]
                for tf in txt_files:
                    with open(os.path.join(vf, tf), "r", encoding="utf-8") as fh:
                        content = fh.read()
                    m = re.search(r'Plate Text:\s*(.+)', content)
                    if m:
                        results.append(m.group(1).strip())
    return results, r1.stdout + r2.stdout + r3.stdout

# ---------- UI ----------
uploaded_file = st.file_uploader("Drop a vehicle image here", type=["jpg", "jpeg", "png", "bmp"],
                                 label_visibility="collapsed")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Recognize Plate", type="primary"):
        with st.spinner("Processing through pipeline..."):
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()
            cv2.imwrite(tmp_path, img)

            try:
                plates, logs = _run_pipeline(tmp_path)
            except RuntimeError as e:
                st.error(f"Pipeline error: {e}")
                os.unlink(tmp_path)
                st.stop()
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out.")
                os.unlink(tmp_path)
                st.stop()

            os.unlink(tmp_path)

        if plates:
            for plate in plates:
                st.markdown(f"""
                <div class="result-card fade-in">
                    <div class="badge">LICENSE PLATE</div>
                    <div class="plate-number">{plate}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No license plate detected.")

        with st.expander("Show pipeline logs"):
            st.code(logs, language="text")

st.markdown("---")
st.markdown("<div class='footer-text'>AI Doroob &mdash; Powered by YOLOv8 + PaddleOCR</div>", unsafe_allow_html=True)
