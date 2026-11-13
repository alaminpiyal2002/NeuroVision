import os
import uuid
import shutil
import zipfile
import math
import warnings
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn

from flask import Flask, request, render_template, send_from_directory, jsonify, abort, send_file

# Visualization HTML generators
from visualization_generator import (
    generate_visualization_html,
    generate_combined_visualization_html,  # ensure available in your file
)

# MONAI model for segmentation
from monai.networks.nets import SegResNet

# Forecasting deps
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Flask app & folders
# -----------------------------------------------------------------------------
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MODELS_FOLDER'] = 'models'
app.config['GRAPHS_FOLDER'] = os.path.join(app.config['RESULTS_FOLDER'], 'graphs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['GRAPHS_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# -----------------------------------------------------------------------------
# Segmentation Model (MONAI SegResNet)
# -----------------------------------------------------------------------------
CKPT_PATH = Path("best_model_SegResNet.pt")
TARGET_SHAPE = (160, 160, 96)
THRESH = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=1,
    dropout_prob=0.2
).to(device)
seg_model.eval()

if CKPT_PATH.is_file():
    state = torch.load(CKPT_PATH, map_location=device)
    seg_model.load_state_dict(state)
else:
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

# ---------------- Helper functions for segmentation ----------------
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

def ensure_zyx(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array; got {arr.shape}")
    return np.transpose(arr, (2, 1, 0))

def center_crop_or_pad(arr: np.ndarray, target_shape=TARGET_SHAPE, pad_value=0):
    z, y, x = arr.shape
    tz, ty, tx = target_shape
    pz1 = max((tz - z) // 2, 0); pz2 = max(tz - (z + pz1), 0)
    py1 = max((ty - y) // 2, 0); py2 = max(ty - (y + py1), 0)
    px1 = max((tx - x) // 2, 0); px2 = max(tx - (x + px1), 0)
    if pz1 or pz2 or py1 or py2 or px1 or px2:
        arr = np.pad(arr, ((pz1,pz2),(py1,py2),(px1,px2)),
                     mode="constant", constant_values=pad_value)
        z, y, x = arr.shape
    sz = (z - tz) // 2; sy = (y - ty) // 2; sx = (x - tx) // 2
    return arr[sz:sz+tz, sy:sy+ty, sx:sx+tx]

def preprocess_modality(path: Path):
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    arr = np.asarray(img.dataobj, dtype=np.float32)
    arr = ensure_zyx(arr)
    mx = float(arr.max())
    if mx > 0:
        arr /= mx
    m, s = float(arr.mean()), float(arr.std())
    if s > 0:
        arr = (arr - m) / s
    return center_crop_or_pad(arr, TARGET_SHAPE, pad_value=0.0)

def get_native_and_canonical_info(ref_path: Path):
    img_native = nib.load(str(ref_path))
    arr_native = np.asarray(img_native.dataobj)
    if arr_native.ndim > 3:
        arr_native = np.squeeze(arr_native)
    if arr_native.ndim != 3:
        raise ValueError(f"Reference volume not 3D: {arr_native.shape}")
    arr_native_shape = tuple(arr_native.shape)
    orig_affine = img_native.affine
    orig_ornt = io_orientation(orig_affine)

    img_can = nib.as_closest_canonical(img_native)
    can_affine = img_can.affine
    arr_can = np.asarray(img_can.dataobj)
    can_zyx = ensure_zyx(arr_can)
    can_zyx_shape = tuple(can_zyx.shape)
    ras_ornt = axcodes2ornt(('R','A','S'))
    return arr_native_shape, orig_affine, ras_ornt, orig_ornt, can_zyx_shape, can_affine

def restore_pred_to_canonical_shape(pred_zyx: np.ndarray, target_shape_zyx: tuple) -> np.ndarray:
    tz, ty, tx = pred_zyx.shape
    z0, y0, x0 = target_shape_zyx
    if (z0 >= tz) and (y0 >= ty) and (x0 >= tx):
        out = np.zeros((z0, y0, x0), dtype=pred_zyx.dtype)
        sz = (z0 - tz) // 2; sy = (y0 - ty) // 2; sx = (x0 - tx) // 2
        out[sz:sz+tz, sy:sy+ty, sx:sx+tx] = pred_zyx
        return out
    else:
        sz = max((tz - z0) // 2, 0); sy = max((ty - y0) // 2, 0); sx = max((tx - x0) // 2, 0)
        ez = sz + min(z0, tz); ey = sy + min(y0, ty); ex = sx + min(x0, tx)
        cropped = pred_zyx[sz:ez, sy:ey, sx:ex]
        pad_z = z0 - cropped.shape[0]; pad_y = y0 - cropped.shape[1]; pad_x = x0 - cropped.shape[2]
        if pad_z or pad_y or pad_x:
            cropped = np.pad(cropped,
                             ((max(pad_z//2,0), max(pad_z - pad_z//2,0)),
                              (max(pad_y//2,0), max(pad_y - pad_y//2,0)),
                              (max(pad_x//2,0), max(pad_x - pad_x//2,0))),
                             mode="constant", constant_values=0)
        return cropped

def reorient_to_native(pred_can_zyx: np.ndarray, ras_ornt, orig_ornt, orig_affine) -> nib.Nifti1Image:
    pred_can_xyz = np.transpose(pred_can_zyx, (2, 1, 0))
    ras_to_orig = ornt_transform(ras_ornt, orig_ornt)
    pred_native_xyz = apply_orientation(pred_can_xyz, ras_to_orig)
    img = nib.Nifti1Image(pred_native_xyz.astype(np.uint8), affine=orig_affine)
    img.set_qform(orig_affine, code=1)
    img.set_sform(orig_affine, code=1)
    img.set_data_dtype(np.uint8)
    return img

def process_files(t1_path, t1ce_path, t2_path, flr_path, output_dir):
    vols = [preprocess_modality(p) for p in [t1_path, t1ce_path, t2_path, flr_path]]
    x = np.stack(vols, axis=0)[None]
    xt = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        logit = seg_model(xt)
    prob = torch.sigmoid(logit)[0,0].cpu().numpy()
    pred_zyx = (prob > THRESH).astype(np.uint8)

    native_shape, orig_affine, ras_ornt, orig_ornt, can_zyx_shape, _ = get_native_and_canonical_info(flr_path)
    pred_can_zyx = restore_pred_to_canonical_shape(pred_zyx, can_zyx_shape)
    pred_native_img = reorient_to_native(pred_can_zyx, ras_ornt, orig_ornt, orig_affine)

    mask_path = os.path.join(output_dir, "pred_mask.nii")
    nib.save(pred_native_img, mask_path)

    # RGB overlay (red tumor)
    flr_native = nib.load(str(flr_path))
    flr_data = np.asarray(flr_native.dataobj, dtype=np.float32)
    flr_norm = (flr_data - np.min(flr_data)) / (np.ptp(flr_data) + 1e-8)
    flr_gray = (flr_norm * 255).astype(np.uint8)
    rgb = np.stack([flr_gray, flr_gray, flr_gray], axis=-1)
    mask_native = pred_native_img.get_fdata().astype(bool)
    rgb[mask_native, 0] = 255
    rgb[mask_native, 1] = 0
    rgb[mask_native, 2] = 0
    overlaid_img = nib.Nifti1Image(rgb, affine=orig_affine)
    overlay_path = os.path.join(output_dir, "overlaid.nii")
    nib.save(overlaid_img, overlay_path)

    return mask_path, overlay_path

# -----------------------------------------------------------------------------
# Forecasting configuration + utilities
# -----------------------------------------------------------------------------
SEQUENCE_LEN = 10
HORIZON_DAYS = 60
QUANTILES = (0.1, 0.5, 0.9)
DEVICE_FC = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_tumor_volume(mask_path: Path):
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    voxel_volume = abs(np.linalg.det(img.affine[:3, :3]))
    tumor_voxels = np.sum(data > 0)
    return tumor_voxels * voxel_volume

def calculate_all_volumes():
    """Scan uploads/* for Patient_* and Day* with pred_mask.nii and write CSV."""
    volumes = []
    for patient_dir in Path(app.config['UPLOAD_FOLDER']).iterdir():
        if not patient_dir.is_dir():
            continue
        if not patient_dir.name.startswith("Patient"):
            continue
        patient_digits = ''.join(ch for ch in patient_dir.name if ch.isdigit())
        if not patient_digits:
            continue
        patient_id = int(patient_digits)

        day_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith("Day")]
        for day_dir in day_dirs:
            day_digits = ''.join(ch for ch in day_dir.name if ch.isdigit())
            if not day_digits:
                continue
            day = int(day_digits)
            mask_path = day_dir / "pred_mask.nii"
            if mask_path.exists():
                try:
                    volume = calculate_tumor_volume(mask_path)
                    volumes.append({
                        "Patient_ID": patient_id,
                        "Day": day,
                        "Tumor_Volume_mm3": float(volume)
                    })
                except Exception as e:
                    print(f"[volumes] Error {patient_id} {day}: {e}")

    if not volumes:
        raise ValueError("No valid mask files found in uploads/*/*/pred_mask.nii")

    df = pd.DataFrame(volumes).sort_values(["Patient_ID", "Day"]).reset_index(drop=True)
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], "tumor_volumes_unseen.csv")
    df.to_csv(csv_path, index=False)
    return df

def interpolate_data():
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], "tumor_volumes_unseen.csv")
    df = pd.read_csv(csv_path)
    df["Patient_ID"] = df["Patient_ID"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["Tumor_Volume_mm3"] = df["Tumor_Volume_mm3"].astype(float)
    df = df.sort_values(["Patient_ID", "Day"]).reset_index(drop=True)

    interpolated = []
    for pid, group in df.groupby("Patient_ID"):
        group = group.drop_duplicates(subset="Day", keep="first").set_index("Day").sort_index()
        min_day = int(group.index.min())
        max_day = int(group.index.max())
        full_index = pd.RangeIndex(start=min_day, stop=max_day + 1, step=1)

        if len(group) >= 2:
            pchip = PchipInterpolator(group.index, group["Tumor_Volume_mm3"])
            interpolated_vals = pchip(full_index)
        else:
            interpolated_vals = np.full_like(full_index, group["Tumor_Volume_mm3"].iloc[0], dtype=float)

        for day, vol in zip(full_index, interpolated_vals):
            interpolated.append({
                "Patient_ID": int(pid),
                "Day": int(day),
                "Day_relative": int(day - min_day),
                "Tumor_Volume_mm3": float(vol)
            })

    interp_df = pd.DataFrame(interpolated)
    interp_df.to_csv(os.path.join(app.config['RESULTS_FOLDER'], "interpolated_data.csv"), index=False)
    return interp_df

# ----- Transformer model for forecasting -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerQuantileDelta(nn.Module):
    def __init__(self, input_dim=1, d_model=96, nhead=4, num_layers=3, dim_ff=288, dropout=0.18, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.q = list(quantiles)
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Dropout(dropout),
            nn.Linear(96, len(self.q))
        )
    def forward(self, x):
        z = self.in_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = z[:, -1, :]
        out = self.head(z)
        return torch.sigmoid(out)

def load_forecasting_model():
    model_path = os.path.join(app.config['MODELS_FOLDER'], "best_transformer_delta_log.pt")
    scalers_path = os.path.join(app.config['MODELS_FOLDER'], "scalers_artifacts.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Scalers file not found: {scalers_path}")

    # --- Load model ---
    model = TransformerQuantileDelta(
        input_dim=1, d_model=96, nhead=4, num_layers=3, dim_ff=288,
        dropout=0.18, quantiles=QUANTILES
    ).to(DEVICE_FC)
    state = torch.load(model_path, map_location=DEVICE_FC)
    model.load_state_dict(state)
    model.eval()

    # --- Load scaler artifacts ---
    art = np.load(scalers_path, allow_pickle=True)
    print(f"[INFO] Loaded scalers_artifacts.npz keys: {list(art.keys())}")

    q_low = art["q_low"].astype(float)
    q_high = art["q_high"].astype(float)

    def get_scaler_values(entry, label="unknown"):
        """Extracts min/scale from npz entry regardless of dict format."""
        if hasattr(entry, "item"):
            entry = entry.item()
        if not isinstance(entry, dict):
            raise ValueError(f"Scaler entry {label} is not a dict: {type(entry)}")

        keys = list(entry.keys())
        print(f"[DEBUG] {label} keys found in scaler dict: {keys}")

        # handle possible naming variations
        if "min_" in entry and "scale_" in entry:
            return entry["min_"], entry["scale_"]
        elif "min" in entry and "scale" in entry:
            return entry["min"], entry["scale"]
        elif "data_min_" in entry and "data_scale_" in entry:
            return entry["data_min_"], entry["data_scale_"]
        else:
            raise KeyError(f"No valid min/scale fields found for {label}")

    global_min, global_scale = get_scaler_values(art["global_minmax"], "global_minmax")
    dlog_min, dlog_scale = get_scaler_values(art["dlog_minmax"], "dlog_minmax")

    # --- Construct scaler objects ---
    global_scaler = MinMaxScaler()
    global_scaler.min_ = np.array(global_min, dtype=float)
    global_scaler.scale_ = np.array(global_scale, dtype=float)

    dlog_scaler = MinMaxScaler()
    dlog_scaler.min_ = np.array(dlog_min, dtype=float)
    dlog_scaler.scale_ = np.array(dlog_scale, dtype=float)

    print("[INFO] Forecasting model and scalers loaded successfully.")
    return model, global_scaler, dlog_scaler, q_low, q_high


def forecast_patient(patient_data, model, global_scaler, dlog_scaler, q_low, q_high):
    patient_data = patient_data.sort_values("Day_relative").reset_index(drop=True)
    if len(patient_data) < SEQUENCE_LEN:
        raise ValueError(f"Not enough data points for patient {patient_data['Patient_ID'].iloc[0]}")

    vals_raw = patient_data[["Tumor_Volume_mm3"]].to_numpy(dtype=float)
    vals_clip = np.clip(vals_raw, q_low, q_high)
    vals_s = global_scaler.transform(vals_clip).astype(np.float32)

    X_last = vals_s[-SEQUENCE_LEN:, :].astype(np.float32)
    y_hist_raw = patient_data["Tumor_Volume_mm3"].to_numpy(dtype=float).tolist()
    last_day = int(patient_data["Day_relative"].iloc[-1])

    q10_idx = QUANTILES.index(0.1)
    q50_idx = QUANTILES.index(0.5)
    q90_idx = QUANTILES.index(0.9)

    cur = torch.from_numpy(X_last).unsqueeze(0).to(DEVICE_FC)
    pred_med, pred_p10, pred_p90 = [], [], []

    with torch.no_grad():
        for _ in range(HORIZON_DAYS):
            q_enc = model(cur).cpu().numpy()[0]

            def decode_one(enc_val):
                dlog = float(dlog_scaler.inverse_transform(np.array([[enc_val]], dtype=float))[0,0])
                y_t = float(y_hist_raw[-1])
                return math.expm1(math.log1p(y_t) + dlog)

            y50 = decode_one(q_enc[q50_idx]); pred_med.append(y50)
            y10 = decode_one(q_enc[q10_idx]); pred_p10.append(y10)
            y90 = decode_one(q_enc[q90_idx]); pred_p90.append(y90)

            y_hist_raw.append(y50)
            next_scaled = global_scaler.transform(np.array([[y50]], dtype=float)).astype(np.float32)
            cur = torch.cat([cur[:,1:,:], torch.from_numpy(next_scaled).unsqueeze(0).to(DEVICE_FC)], dim=1)

    future_days = list(range(last_day + 1, last_day + 1 + HORIZON_DAYS))
    return {
        "future_days": future_days,
        "pred_med": np.array(pred_med),
        "pred_p10": np.array(pred_p10),
        "pred_p90": np.array(pred_p90),
        "last_day": last_day,
        "history_days": patient_data["Day_relative"].tolist(),
        "history_volumes": patient_data["Tumor_Volume_mm3"].tolist()
    }

def generate_forecast_graph(patient_name, forecast_data):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(forecast_data["history_days"], forecast_data["history_volumes"],
                 linewidth=2, color='blue', label="History (mm³)")
        plt.plot(forecast_data["future_days"], forecast_data["pred_med"],
                 linestyle="--", linewidth=2, color='orange',
                 label=f"Forecast P50 ({HORIZON_DAYS}d)")
        plt.fill_between(forecast_data["future_days"],
                         forecast_data["pred_p10"],
                         forecast_data["pred_p90"],
                         alpha=0.2, color='lightblue', label="~80% interval (P10–P90)")
        plt.axvline(forecast_data["last_day"], color="gray", linestyle=":", linewidth=2, label="Forecast start")
        plt.title(f"{patient_name} — {HORIZON_DAYS}-Day Forecast (Tumor Volume, mm³)")
        plt.xlabel("Day_relative")
        plt.ylabel("Tumor Volume (mm³)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(app.config['GRAPHS_FOLDER'], exist_ok=True)
        safe_name = patient_name.replace(" ", "_")
        graph_path = os.path.join(app.config['GRAPHS_FOLDER'], f"{safe_name}_forecast.png")
        plt.savefig(graph_path, bbox_inches='tight', dpi=300)
        plt.close()
        return graph_path

    except Exception as e:
        raise RuntimeError(f"Error generating graph: {str(e)}")

# -----------------------------------------------------------------------------
# Routes (Segmentation UI + Forecasting)
# -----------------------------------------------------------------------------
@app.route('/')
def dashboard():
    """Main dashboard summary page"""
    up = Path(app.config['UPLOAD_FOLDER'])
    total_patients = total_timepoints = total_visuals = 0
    last_forecast = None

    if up.exists():
        for p in up.iterdir():
            if not p.is_dir() or p.name=="temp": continue
            total_patients+=1
            for d in p.iterdir():
                if d.is_dir() and "Day" in d.name:
                    total_timepoints+=1
                    if (d/"visualization.html").exists(): total_visuals+=1

    graphs = Path(app.config['GRAPHS_FOLDER'])
    if graphs.exists():
        latest = sorted(graphs.glob("*.png"), key=lambda f:f.stat().st_mtime, reverse=True)
        if latest: last_forecast = latest[0].name.replace("_forecast.png","")

    return render_template("dashboard.html",
                           title="Dashboard Overview",
                           total_patients=total_patients,
                           total_timepoints=total_timepoints,
                           total_visuals=total_visuals,
                           last_forecast=last_forecast)

@app.route('/upload_files', methods=['POST'])
def upload_files():
    if not all(k in request.files for k in ('t1n', 'tc1', 't2w', 't2f')) or \
       'timepoint' not in request.form or 'patientID' not in request.form:
        return jsonify(error="Missing required files, timepoint, or patientID"), 400

    patient_id = request.form['patientID'].strip().replace(" ", "_")
    timepoint = request.form['timepoint'].strip().replace(" ", "_")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', session_id)
    os.makedirs(session_dir, exist_ok=True)

    paths = {}
    for key in ('t1n', 'tc1', 't2w', 't2f'):
        file = request.files[key]
        ext = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
        save_path = os.path.join(session_dir, f"{key}{ext}")
        file.save(save_path)
        paths[key] = save_path

    try:
        patient_dir = os.path.join(app.config['UPLOAD_FOLDER'], patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        timepoint_dir = os.path.join(patient_dir, timepoint)
        os.makedirs(timepoint_dir, exist_ok=True)

        process_files(paths['t1n'], paths['tc1'], paths['t2w'], paths['t2f'], timepoint_dir)

        # Per-day visualization
        try:
            generate_visualization_html(timepoint_dir)
        except Exception as vis_err:
            print(f"[WARN] Visualization generation failed for {timepoint_dir}: {vis_err}")

        shutil.rmtree(session_dir, ignore_errors=True)

        # -----------------------------
        #     AUTO-UPDATE FORECAST CSV
        # -----------------------------
        try:
            print("[AUTO] Updating forecast CSV after new upload...")
            df = calculate_all_volumes()
            interpolate_data()
            print("[AUTO] Forecast CSV updated.")
        except Exception as e:
            print("[WARN] Forecast CSV update failed:", e)
        # -----------------------------

        return jsonify(
            success=True,
            patient=patient_id,
            timepoint=timepoint,
            mask_url=f"/download/{patient_id}/{timepoint}/pred_mask.nii",
            overlay_url=f"/download/{patient_id}/{timepoint}/overlaid.nii"
        )

    except Exception as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return jsonify(error=str(e)), 500


@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    """Extract ZIP → segment each day → create per-day HTML → combined HTML → auto-update forecasting."""
    if 'zipfile' not in request.files:
        return jsonify(error="No zip file uploaded"), 400

    zip_file = request.files['zipfile']
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        zip_path = os.path.join(temp_dir, "upload.zip")
        zip_file.save(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        processed_patients = {}

        # Find all patient folders inside extracted structure
        for root, dirs, files in os.walk(temp_dir):
            for patient in dirs:
                if not patient.lower().startswith("patient"):
                    continue
                patient_path = os.path.join(root, patient)

                uploads_patient_dir = os.path.join(app.config['UPLOAD_FOLDER'], patient)
                os.makedirs(uploads_patient_dir, exist_ok=True)
                processed_patients[patient] = []

                for day in os.listdir(patient_path):
                    day_path = os.path.join(patient_path, day)
                    if not os.path.isdir(day_path) or not day.lower().startswith("day"):
                        continue

                    def find_file(base, name):
                        for ext in [".nii.gz", ".nii"]:
                            f = os.path.join(base, f"{name}{ext}")
                            if os.path.exists(f):
                                return f
                        return None

                    t1n = find_file(day_path, "t1n")
                    t2f = find_file(day_path, "t2f")
                    t2w = find_file(day_path, "t2w")
                    tc1 = find_file(day_path, "tc1")

                    if all([t1n, t2f, t2w, tc1]):
                        output_day = os.path.join(uploads_patient_dir, day)
                        os.makedirs(output_day, exist_ok=True)
                        try:
                            process_files(t1n, tc1, t2w, t2f, output_day)
                            processed_patients[patient].append(day)

                            try:
                                generate_visualization_html(output_day)
                                print(f"[INFO] Visualization generated at {output_day}/visualization.html")
                            except Exception as vis_err:
                                print(f"[WARN] Visualization generation failed for {output_day}: {vis_err}")

                        except Exception as e:
                            print(f"[zip] Error processing {patient}/{day}: {e}")

                # Combined visualization
                try:
                    generate_combined_visualization_html(uploads_patient_dir)
                except Exception as comb_err:
                    print(f"[WARN] Combined visualization failed for {patient}: {comb_err}")

        shutil.rmtree(temp_dir, ignore_errors=True)

        if not processed_patients:
            return jsonify(error="No valid patients found in zip"), 400

        # -----------------------------
        #     AUTO-UPDATE FORECAST CSV
        # -----------------------------
        try:
            print("[AUTO] Updating forecast CSV after ZIP upload...")
            df = calculate_all_volumes()
            interpolate_data()
            print("[AUTO] Forecast CSV updated after ZIP.")
        except Exception as e:
            print("[WARN] Forecast CSV update after ZIP failed:", e)
        # -----------------------------

        return jsonify(success=True, processed=processed_patients)

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(error=str(e)), 500


@app.route("/download/<path:patient>/<path:timepoint>/<path:filename>")
def download_file(patient, timepoint, filename):
    patient = unquote(patient)
    timepoint = unquote(timepoint)
    filename = unquote(filename)

    file_dir = os.path.join(app.config["UPLOAD_FOLDER"], patient, timepoint)
    file_path = os.path.join(file_dir, filename)

    if not os.path.exists(file_path):
        abort(404, description=f"File not found: {file_path}")

    return send_from_directory(file_dir, filename, as_attachment=False)

@app.route('/view_timepoints')
def view_timepoints():
    patients = []
    for patient in os.listdir(app.config['UPLOAD_FOLDER']):
        patient_path = os.path.join(app.config['UPLOAD_FOLDER'], patient)
        if not os.path.isdir(patient_path) or patient == "temp":
            continue
        timepoints = []
        for day in os.listdir(patient_path):
            day_path = os.path.join(patient_path, day)
            if os.path.isdir(day_path):
                mask_exists = os.path.exists(os.path.join(day_path, "pred_mask.nii"))
                overlay_exists = os.path.exists(os.path.join(day_path, "overlaid.nii"))
                timepoints.append({
                    'name': day,
                    'mask_exists': mask_exists,
                    'overlay_exists': overlay_exists
                })
        patients.append({'name': patient, 'timepoints': timepoints})
    return render_template('timepoints.html', patients=patients)

# ---------------- Forecasting Pages & APIs ----------------
@app.route('/forecast_page')
def forecast_page():
    """Render the forecasting selection page with full folder names."""
    patients = []
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    if upload_dir.exists():
        for p in upload_dir.iterdir():
            if p.is_dir() and p.name.lower().startswith("patient"):
                patients.append({"id": p.name, "name": p.name})
    patients = sorted(patients, key=lambda x: x["name"].lower())
    return render_template('forecast.html', patients=patients)

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate 60-day forecast graph for a selected patient (with debug logging)."""
    try:
        patient_ref = request.form.get("patient_id", "").strip()
        print(f"\n[DEBUG] Forecast request for: {patient_ref}")
        if not patient_ref:
            return jsonify({"error": "Missing patient ID"}), 400

        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        ref_digits = ''.join(ch for ch in patient_ref if ch.isdigit())
        possible_dirs = [
            d for d in upload_dir.iterdir()
            if d.is_dir() and (
                d.name.lower() == patient_ref.lower() or
                ''.join(ch for ch in d.name if ch.isdigit()) == ref_digits
            )
        ]
        if not possible_dirs:
            print(f"[ERROR] Patient folder not found for {patient_ref}")
            return jsonify({"error": f"Patient {patient_ref} not found"}), 404

        patient_dir = possible_dirs[0]
        patient_name = patient_dir.name
        print(f"[DEBUG] Matched folder: {patient_name}")

        volumes_csv = os.path.join(app.config["RESULTS_FOLDER"], "tumor_volumes_unseen.csv")
        interp_csv = os.path.join(app.config["RESULTS_FOLDER"], "interpolated_data.csv")

        if not os.path.exists(volumes_csv):
            print("[INFO] Calculating volumes...")
            calculate_all_volumes()
        if not os.path.exists(interp_csv):
            print("[INFO] Interpolating data...")
            interpolate_data()

        df = pd.read_csv(interp_csv)
        print(f"[DEBUG] Loaded interpolated CSV ({len(df)} rows)")
        df["Patient_ID_str"] = df["Patient_ID"].astype(str)

        patient_data = df[
            (df["Patient_ID_str"] == ref_digits) |
            (df["Patient_ID_str"] == str(int(ref_digits))) |
            (df["Patient_ID_str"].str.contains(ref_digits))
        ]

        if patient_data.empty:
            print(f"[ERROR] No data found for {patient_name}")
            return jsonify({"error": f"No data found for {patient_name}"}), 404

        print(f"[DEBUG] Found {len(patient_data)} records for forecast")
        model, global_scaler, dlog_scaler, q_low, q_high = load_forecasting_model()
        print("[DEBUG] Forecasting model loaded")

        forecast_data = forecast_patient(patient_data, model, global_scaler, dlog_scaler, q_low, q_high)
        print("[DEBUG] Forecast computed")

        graph_path = generate_forecast_graph(patient_name, forecast_data)
        relative_path = os.path.relpath(graph_path, os.getcwd()).replace("\\", "/")
        print(f"[DEBUG] Graph generated at: {relative_path}")

        return jsonify({"graph_path": relative_path})

    except Exception as e:
        import traceback
        print("[ERROR in /forecast]", traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
@app.route('/forecast_visualization/<patient_ref>')
def forecast_visualization(patient_ref):
    """Render the forecast visualization HTML for a given patient inside a new tab."""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        ref_digits = ''.join(ch for ch in str(patient_ref) if ch.isdigit())

        # --- Find matching patient folder ---
        possible_dirs = [
            d for d in upload_dir.iterdir()
            if d.is_dir() and (
                d.name.lower() == patient_ref.lower() or
                ''.join(ch for ch in d.name if ch.isdigit()) == ref_digits
            )
        ]

        if not possible_dirs:
            return f"<h2>❌ Patient {patient_ref} not found.</h2>", 404

        patient_dir = possible_dirs[0]
        patient_name = patient_dir.name

        # --- Ensure CSVs exist ---
        volumes_path = os.path.join(app.config['RESULTS_FOLDER'], "tumor_volumes_unseen.csv")
        interp_path = os.path.join(app.config['RESULTS_FOLDER'], "interpolated_data.csv")

        if not os.path.exists(volumes_path):
            calculate_all_volumes()
        if not os.path.exists(interp_path):
            interpolate_data()

        # --- Load data ---
        df = pd.read_csv(interp_path)
        df["Patient_ID_str"] = df["Patient_ID"].astype(str)

        patient_data = df[
            (df["Patient_ID_str"] == ref_digits) |
            (df["Patient_ID_str"] == str(int(ref_digits))) |
            (df["Patient_ID_str"].str.contains(ref_digits))
        ]

        if patient_data.empty:
            return f"<h2>⚠️ No forecast data available for {patient_name}.</h2>", 404

        # --- Run forecasting model ---
        model, global_scaler, dlog_scaler, q_low, q_high = load_forecasting_model()
        forecast_data = forecast_patient(patient_data, model, global_scaler, dlog_scaler, q_low, q_high)

        # --- Helper to safely convert numpy arrays to Python lists ---
        def to_list_safe(x):
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x) if isinstance(x, (tuple, set)) else x

        # --- Convert forecast arrays to lists for Jinja ---
        forecast_data["pred_med"] = to_list_safe(forecast_data.get("pred_med", []))
        forecast_data["pred_p10"] = to_list_safe(forecast_data.get("pred_p10", []))
        forecast_data["pred_p90"] = to_list_safe(forecast_data.get("pred_p90", []))
        forecast_data["history_volumes"] = to_list_safe(forecast_data.get("history_volumes", []))
        forecast_data["history_days"] = to_list_safe(forecast_data.get("history_days", []))
        forecast_data["future_days"] = to_list_safe(forecast_data.get("future_days", []))

        # --- Calculate metrics ---
        history_vols = forecast_data["history_volumes"]
        history_days = forecast_data["history_days"]
        initial_vol = history_vols[0] if history_vols else 0
        current_vol = history_vols[-1] if history_vols else 0
        change_pct = ((current_vol - initial_vol) / initial_vol * 100) if initial_vol else 0

        # --- Detect phases ---
        min_vol = min(history_vols) if history_vols else 0
        min_day = history_days[history_vols.index(min_vol)] if history_vols else 0
        has_shrinking = min_vol < initial_vol * 0.95 if initial_vol else False
        has_regrowth = current_vol > min_vol * 1.1 if min_vol else False
        has_phases = has_shrinking or has_regrowth

        # --- Define main messages ---
        if current_vol < initial_vol * 0.7:
            emotion_class = "good"
            main_message = "Your tumor is shrinking! 💚"
            subtitle = "Treatment is showing positive results."
        elif current_vol > initial_vol * 1.1:
            emotion_class = "warning"
            main_message = "Tumor regrowth detected ⚠️"
            subtitle = "Please consult your oncologist soon."
        else:
            emotion_class = "stable"
            main_message = "Tumor remains stable 💙"
            subtitle = "No significant changes detected."

        # --- Build tumor journey story ---
        story_html = ""
        if has_phases:
            shrink_pct = ((initial_vol - min_vol) / initial_vol * 100) if initial_vol else 0
            regrowth_pct = ((current_vol - min_vol) / min_vol * 100) if min_vol else 0
            story_html += f"""
            <div class='milestone good'>
                <b>Phase 1:</b> Tumor shrank by {shrink_pct:.1f}% — from {initial_vol:.0f} → {min_vol:.0f} mm³ 💚
            </div>
            <div class='milestone warning'>
                <b>Phase 2:</b> Regrowth detected — increased by {regrowth_pct:.1f}% up to {current_vol:.0f} mm³ ⚠️
            </div>
            <div class='milestone' style="background:#fff8e1;">
                <b>Next 60 Days:</b> Forecast predicts continued trend under observation 🔮
            </div>
            """
        else:
            story_html = """
            <div class='milestone stable'>
                The tumor volume remained stable over time — no significant growth or shrinkage detected 💙
            </div>
            """

        # --- Render page with full dataset ---
        return render_template(
            'forecast_visualization.html',
            patient_name=patient_name,
            main_message=main_message,
            subtitle=subtitle,
            emotion_class=emotion_class,
            initial_vol=initial_vol,
            current_vol=current_vol,
            change_pct=change_pct,
            change_arrow="↓" if change_pct < 0 else ("↑" if change_pct > 0 else "→"),
            total_days=len(history_days),
            last_day=forecast_data.get("last_day", 0),
            story_html=story_html,
            has_shrinking=has_shrinking,
            has_regrowth=has_regrowth,
            has_phases=has_phases,

            # ✅ Data for Plotly chart
            history_days=forecast_data["history_days"],
            history_volumes=forecast_data["history_volumes"],
            future_days=forecast_data["future_days"],
            future_volumes=forecast_data["pred_med"],
            p10=forecast_data["pred_p10"],
            p90=forecast_data["pred_p90"],
            min_day=min_day
        )

    except Exception as e:
        import traceback
        print("[ERROR in /forecast_visualization]", traceback.format_exc())
        return f"<h1>Error generating forecast: {e}</h1>", 500








@app.route('/get_days/<patient_ref>')
def get_days(patient_ref):
    """Get available days for a patient; JSON only (no HTML errors)."""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        possible_dirs = []

        for d in upload_dir.iterdir():
            if not d.is_dir():
                continue
            digits = ''.join(ch for ch in d.name if ch.isdigit())
            if d.name == patient_ref or digits == patient_ref or digits == ''.join(ch for ch in str(patient_ref) if ch.isdigit()):
                possible_dirs.append(d)

        if not possible_dirs:
            return jsonify({"error": f"Patient {patient_ref} not found"}), 404

        patient_dir = possible_dirs[0]
        days = []
        for day_dir in patient_dir.iterdir():
            if day_dir.is_dir() and "Day" in day_dir.name:
                digits = ''.join(ch for ch in day_dir.name if ch.isdigit())
                if digits:
                    mask_path = day_dir / "pred_mask.nii"
                    if mask_path.exists():
                        days.append(int(digits))

        return jsonify({"days": sorted(days)})

    except Exception as e:
        import traceback
        print("[ERROR in /get_days]", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/results/graphs/<filename>')
def serve_graph(filename):
    return send_from_directory(app.config['GRAPHS_FOLDER'], filename)

@app.route('/debug')
def debug():
    volumes_path = os.path.join(app.config['RESULTS_FOLDER'], "tumor_volumes_unseen.csv")
    interp_path = os.path.join(app.config['RESULTS_FOLDER'], "interpolated_data.csv")

    volumes = "Not found"
    if os.path.exists(volumes_path):
        with open(volumes_path, 'r') as f:
            volumes = f.read()

    interp = "Not found"
    if os.path.exists(interp_path):
        with open(interp_path, 'r') as f:
            interp = f.read()

    return f"""
    <h1>Tumor Volumes CSV</h1>
    <pre>{volumes}</pre>
    <h1>Interpolated Data CSV</h1>
    <pre>{interp}</pre>
    """

@app.route('/force_forecast_data')
def force_forecast_data():
    try:
        print("🔁 Recomputing all tumor volumes...")
        df = calculate_all_volumes()
        interp_df = interpolate_data()
        msg = f"""
        ✅ Forecast data regenerated.<br>
        - {len(df)} raw volume records<br>
        - {len(interp_df)} interpolated records<br>
        - CSV files saved in /results/
        """
        return msg
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"<h3>❌ Error during recomputation:</h3><pre>{e}</pre>", 500



# -----------------------------------------------------------------------------
# View visualization (single-day or combined)
# -----------------------------------------------------------------------------
@app.route('/view_visualization/<patient_ref>/<day>')
def view_visualization(patient_ref, day):
    """Serve pre-generated 3D visualization HTML for a given patient and day (or combined view)."""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        ref_digits = ''.join(ch for ch in str(patient_ref) if ch.isdigit())
        patient_dir = None

        for d in os.listdir(upload_dir):
            full_path = os.path.join(upload_dir, d)
            if not os.path.isdir(full_path):
                continue
            digits = ''.join(ch for ch in d if ch.isdigit())
            if d == patient_ref or digits == ref_digits:
                patient_dir = full_path
                break

        if not patient_dir:
            return f"<h2>❌ Patient folder not found for '{patient_ref}'</h2>", 404

        if str(day).lower() in ["all", "combined"]:
            html_path = os.path.join(patient_dir, "combined_visualization.html")
            if not os.path.exists(html_path):
                try:
                    generate_combined_visualization_html(patient_dir)
                except Exception as gen_err:
                    return f"<h2>⚠️ Failed to generate combined visualization: {gen_err}</h2>", 500
            return send_from_directory(os.path.dirname(html_path), os.path.basename(html_path))

        timepoint_dir = os.path.join(patient_dir, f"Day {day}")
        if not os.path.exists(timepoint_dir):
            timepoint_dir = os.path.join(patient_dir, f"Day{day}")
        if not os.path.exists(timepoint_dir):
            return f"<h2>Day folder for '{day}' not found in {patient_dir}.</h2>", 404

        html_path = os.path.join(timepoint_dir, "visualization.html")
        if not os.path.exists(html_path):
            generate_visualization_html(timepoint_dir)

        return send_from_directory(os.path.dirname(html_path), os.path.basename(html_path))

    except Exception as e:
        import traceback
        print("[ERROR in /view_visualization]", traceback.format_exc())
        return f"<h2>Error loading visualization: {e}</h2>", 500

    
@app.route('/view_visualization')
def view_visualization_page():
    up=Path(app.config['UPLOAD_FOLDER'])
    patients=[{"name":p.name} for p in up.iterdir() if p.is_dir() and p.name.lower().startswith("patient")]
    return render_template('visualization.html',patients=patients,title="3D Visualization")

# -----------------------------------------------------------------------------
# Forecast PDF (unchanged behavior)
# -----------------------------------------------------------------------------




@app.route('/upload_page')
def upload_page():
    """Render the upload dashboard page (modern UI from index.html)."""
    return render_template('index.html', title="Upload Scans", active="upload")

@app.route('/about')
def about_page():
    return render_template('about.html', title="About Us", active="about")
# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
