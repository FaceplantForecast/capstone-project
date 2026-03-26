#!/usr/bin/env python3
"""
Optimized fall-detection training / inference script for AWR2944EVM radar .npy data.

Key upgrades:
1) Uses snr_adj if present (matches your background-subtracted collector output).
2) Supports labels.csv columns:
      - filename,is_fall
      - file,fall
3) Groups by sample_frame if available, else frame_id.
4) Uses live-like person cluster selection with centroid tracking.
5) No data leakage: scaler fit on TRAIN only.
6) No window leakage: train/val split BY FILE.
7) Stronger temporal model: residual dilated 1D CNN.
8) Better callbacks and robust saved artifacts.
9) TFLite export included.
"""

import os
import argparse
import math
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./data"
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

WINDOW_SIZE = 48
WINDOW_STEP = 1

MODEL_PATH = "model.h5"
BEST_MODEL_PATH = "best_model.h5"
SCALER_PATH = "scaler.npz"
TFLITE_PATH = "model.tflite"
TRAIN_LOG_PATH = "train_log.csv"
VAL_REPORT_PATH = "val_report.json"

FEATURE_KEYS = [
    "cx", "cy", "cz", "height", "spread_xy",
    "mean_doppler", "num_points", "vz", "speed",
]

# ============================================================
# DATA / PREPROCESSING KNOBS
# ============================================================

SEED = 99
VAL_FRACTION_BY_FILE = 0.15

# Prefer snr_adj from your collector. If missing, fallback to snr.
PREFER_SNR_ADJ = True

MIN_RANGE_M = 0.25
MAX_RANGE_FOR_PERSON_M = 20.0

# Thresholding on adjusted SNR-like signal
SNR_THRESHOLD = 3.5
SNR_RELAX_DB = 3.0
MIN_POINTS_AFTER_SNR = 8
MIN_POINTS_FALLBACK_ALL = 25

# Person-cluster selection / geometry guards
MIN_HEIGHT_M = 0.01
MAX_SPREAD_XY_M = 3.5

TRACK_MAX_DIST_M = 0.75
TRACK_BONUS = 1.0
CENTROID_EMA_ALPHA = 0.35

# DBSCAN
DBSCAN_EPS_CANDIDATES = [0.25, 0.30, 0.36, 0.42, 0.50, 0.60]
DBSCAN_MIN_SAMPLES_FRAC = 0.06
DBSCAN_MIN_SAMPLES_MIN = 4
DBSCAN_MIN_SAMPLES_MAX = 12

# ============================================================
# TRAINING KNOBS
# ============================================================

BATCH_SIZE = 64
MAX_EPOCHS = 100
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
USE_MIXED_PRECISION = False

# use class weights OR focal, not both piled on aggressively
USE_CLASS_WEIGHTS = True

# ============================================================
# REPRO
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def get_group_col(df: pd.DataFrame) -> str:
    if "sample_frame" in df.columns:
        return "sample_frame"
    if "frame_id" in df.columns:
        return "frame_id"
    raise ValueError("Input .npy must contain either 'sample_frame' or 'frame_id'.")

def get_label_columns(labels_df: pd.DataFrame):
    if {"filename", "is_fall"}.issubset(labels_df.columns):
        return "filename", "is_fall"
    if {"file", "fall"}.issubset(labels_df.columns):
        return "file", "fall"
    raise ValueError(
        "labels.csv must have either columns ['filename','is_fall'] or ['file','fall']"
    )

def pick_signal_col(df: pd.DataFrame) -> str:
    if PREFER_SNR_ADJ and "snr_adj" in df.columns:
        return "snr_adj"
    if "snr" in df.columns:
        return "snr"
    raise ValueError("Input .npy must contain 'snr' or 'snr_adj'.")

def load_structured_npy_as_df(npy_path: str) -> pd.DataFrame:
    raw = np.load(npy_path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
        df = pd.DataFrame(raw)
    elif isinstance(raw, np.ndarray) and raw.ndim == 2:
        # fallback only if plain matrix provided
        cols = ["frame_id", "timestamp", "x", "y", "z", "doppler", "snr"]
        if raw.shape[1] < len(cols):
            raise ValueError(f"{npy_path}: unsupported ndarray shape {raw.shape}")
        df = pd.DataFrame(raw[:, :len(cols)], columns=cols)
    else:
        raise ValueError(f"{npy_path}: unsupported npy format")

    required_base = {"timestamp", "x", "y", "z", "doppler"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"{npy_path}: missing required columns {missing}")

    if "frame_id" not in df.columns and "sample_frame" not in df.columns:
        raise ValueError(f"{npy_path}: needs either frame_id or sample_frame")

    return df

def find_fall_start_for_file(basename_no_ext):
    candidates = [
        os.path.join(DATA_DIR, f"{basename_no_ext}_fall.csv"),
        os.path.join(DATA_DIR, f"{basename_no_ext}.csv"),
    ]

    for path in candidates:
        if not os.path.isfile(path):
            continue

        try:
            df = pd.read_csv(path)

            if "fall_frame" in df.columns and not df["fall_frame"].empty:
                val = df["fall_frame"].dropna()
                if not val.empty:
                    return int(val.iloc[0])

            if df.shape[1] == 1 and not df.empty:
                val = df.iloc[:, 0].dropna()
                if not val.empty:
                    return int(val.iloc[0])
        except Exception:
            pass

        try:
            df = pd.read_csv(path, header=None)
            if not df.empty:
                return int(df.iloc[0, 0])
        except Exception:
            pass

    return None


# ============================================================
# CLUSTERING HELPERS
# ============================================================

def _adaptive_min_samples(n_pts: int) -> int:
    ms = int(np.ceil(DBSCAN_MIN_SAMPLES_FRAC * n_pts))
    ms = max(DBSCAN_MIN_SAMPLES_MIN, min(DBSCAN_MIN_SAMPLES_MAX, ms))
    return ms

def _dbscan_sweep(pts, eps_list, min_samples):
    for eps in eps_list:
        labels = DBSCAN(eps=float(eps), min_samples=min_samples).fit_predict(pts)
        if np.any(labels != -1):
            return labels
    return np.full(len(pts), -1, dtype=int)

def cluster_frame_points(points_xyz):
    n = len(points_xyz)
    if n == 0:
        return np.array([], dtype=int)

    pts = np.asarray(points_xyz, dtype=np.float32)
    ms = _adaptive_min_samples(n)

    labels = _dbscan_sweep(pts, DBSCAN_EPS_CANDIDATES, ms)
    if np.any(labels != -1):
        return labels

    pts_xy = pts[:, :2]
    labels = _dbscan_sweep(pts_xy, [e * 1.15 for e in DBSCAN_EPS_CANDIDATES], ms)
    return labels

def _cluster_stats(pts):
    cx, cy, cz = pts.mean(axis=0)
    height = float(pts[:, 2].max() - pts[:, 2].min())
    spread_xy = float(max(
        pts[:, 0].max() - pts[:, 0].min(),
        pts[:, 1].max() - pts[:, 1].min()
    ))
    r = float(np.sqrt(cx * cx + cy * cy + cz * cz))
    return cx, cy, cz, height, spread_xy, r

def pick_person_cluster(points_xyz, labels, prev_centroid=None):
    n = len(points_xyz)
    if n == 0:
        return np.zeros(0, dtype=bool)

    labels = np.asarray(labels)
    valid_labels = np.unique(labels[labels != -1])

    if valid_labels.size == 0:
        if n >= MIN_POINTS_FALLBACK_ALL:
            return np.ones(n, dtype=bool)
        return np.zeros(n, dtype=bool)

    pts_all = np.asarray(points_xyz, dtype=np.float32)

    best_label = None
    best_score = -1e9

    for lab in valid_labels:
        mask = labels == lab
        pts = pts_all[mask]
        m = pts.shape[0]
        if m < 3:
            continue

        cx, cy, cz, height, spread_xy, r = _cluster_stats(pts)
        score = np.sqrt(m)

        if height >= MIN_HEIGHT_M:
            score += 1.0
        else:
            score -= 1.5

        if spread_xy > MAX_SPREAD_XY_M:
            score -= 2.0 * (spread_xy / max(MAX_SPREAD_XY_M, 1e-6))

        if MAX_RANGE_FOR_PERSON_M is not None and r > MAX_RANGE_FOR_PERSON_M:
            score -= 2.5 * (r / MAX_RANGE_FOR_PERSON_M)

        if prev_centroid is not None:
            px, py, pz = prev_centroid
            d = float(np.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2))
            if d <= TRACK_MAX_DIST_M:
                score += TRACK_BONUS * (1.0 - d / TRACK_MAX_DIST_M)
            else:
                score -= 0.5 * (d / TRACK_MAX_DIST_M)

        if score > best_score:
            best_score = score
            best_label = lab

    if best_label is None:
        if n >= MIN_POINTS_FALLBACK_ALL:
            return np.ones(n, dtype=bool)
        return np.zeros(n, dtype=bool)

    return labels == best_label


# ============================================================
# TRACK STATE
# ============================================================

class TrackState:
    def __init__(self):
        self.prev_feat = None
        self.prev_centroid_ema = None

    def update_centroid_ema(self, cx, cy, cz):
        if self.prev_centroid_ema is None:
            self.prev_centroid_ema = (cx, cy, cz)
        else:
            px, py, pz = self.prev_centroid_ema
            a = CENTROID_EMA_ALPHA
            self.prev_centroid_ema = (
                a * cx + (1 - a) * px,
                a * cy + (1 - a) * py,
                a * cz + (1 - a) * pz,
            )
        return self.prev_centroid_ema


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_frame_features_df(df_frame: pd.DataFrame, track: TrackState, signal_col: str):
    t = float(df_frame["timestamp"].iloc[0])

    default_feat = dict(
        cx=0.0,
        cy=0.0,
        cz=0.0,
        height=0.0,
        spread_xy=0.0,
        mean_doppler=0.0,
        num_points=0,
        timestamp=t,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        speed=0.0,
    )

    x = df_frame["x"].to_numpy(dtype=np.float32)
    y = df_frame["y"].to_numpy(dtype=np.float32)
    z = df_frame["z"].to_numpy(dtype=np.float32)
    doppler = df_frame["doppler"].to_numpy(dtype=np.float32)
    signal = df_frame[signal_col].to_numpy(dtype=np.float32)

    if x.size == 0:
        track.prev_feat = default_feat
        return default_feat

    r = np.sqrt(x * x + y * y + z * z)
    mask_r = r >= MIN_RANGE_M
    if not np.any(mask_r):
        track.prev_feat = default_feat
        return default_feat

    x = x[mask_r]
    y = y[mask_r]
    z = z[mask_r]
    doppler = doppler[mask_r]
    signal = signal[mask_r]

    def _apply_thr(thr):
        mask = signal > thr
        if not np.any(mask):
            return None
        pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
        dop = doppler[mask]
        return pts, dop

    out = _apply_thr(SNR_THRESHOLD)
    if out is None:
        track.prev_feat = default_feat
        return default_feat

    points_xyz, dop_f = out

    if len(points_xyz) < MIN_POINTS_AFTER_SNR and SNR_RELAX_DB > 0:
        out2 = _apply_thr(max(0.0, SNR_THRESHOLD - SNR_RELAX_DB))
        if out2 is not None:
            points_xyz, dop_f = out2

    if len(points_xyz) == 0:
        track.prev_feat = default_feat
        return default_feat

    labels = cluster_frame_points(points_xyz)
    prev_centroid = track.prev_centroid_ema if track.prev_centroid_ema is not None else None
    person_mask = pick_person_cluster(points_xyz, labels, prev_centroid=prev_centroid)

    if not np.any(person_mask):
        track.prev_feat = default_feat
        return default_feat

    pts = points_xyz[person_mask]
    dop = dop_f[person_mask]

    cx, cy, cz, height, spread_xy, _ = _cluster_stats(pts)

    if height < MIN_HEIGHT_M or spread_xy > MAX_SPREAD_XY_M:
        track.prev_feat = default_feat
        return default_feat

    cx_s, cy_s, cz_s = track.update_centroid_ema(float(cx), float(cy), float(cz))

    feat = dict(
        cx=float(cx_s),
        cy=float(cy_s),
        cz=float(cz_s),
        height=float(height),
        spread_xy=float(spread_xy),
        mean_doppler=float(np.mean(dop)) if dop.size else 0.0,
        num_points=int(pts.shape[0]),
        timestamp=t,
    )

    if track.prev_feat is not None:
        dt = max(float(t) - float(track.prev_feat["timestamp"]), 1e-3)
        vx = (feat["cx"] - track.prev_feat["cx"]) / dt
        vy = (feat["cy"] - track.prev_feat["cy"]) / dt
        vz = (feat["cz"] - track.prev_feat["cz"]) / dt
    else:
        vx = vy = vz = 0.0

    speed = float(np.sqrt(vx**2 + vy**2 + vz**2))
    feat.update(vx=float(vx), vy=float(vy), vz=float(vz), speed=speed)

    track.prev_feat = feat
    return feat

def extract_frame_features_for_file(npy_path):
    df_points = load_structured_npy_as_df(npy_path)
    signal_col = pick_signal_col(df_points)
    group_col = get_group_col(df_points)

    sort_cols = [group_col]
    if "timestamp" in df_points.columns:
        sort_cols.append("timestamp")
    df_points = df_points.sort_values(sort_cols).reset_index(drop=True)

    feats = []
    track = TrackState()

    for gid, df_frame in df_points.groupby(group_col, sort=True):
        f = compute_frame_features_df(df_frame, track, signal_col=signal_col)
        f[group_col] = int(gid)
        feats.append(f)

    if not feats:
        raise RuntimeError(f"{npy_path}: no frames found after feature extraction")

    df_feat = pd.DataFrame(feats).sort_values(group_col).reset_index(drop=True)

    # Keep frame_id-like canonical column for downstream logic
    if group_col != "frame_id":
        df_feat["frame_id"] = df_feat[group_col].astype(np.int32)

    return df_feat


# ============================================================
# WINDOW BUILDING
# ============================================================

def build_windows_and_labels(
    df_feat,
    fall_flag,
    fall_start_frame=None,
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP
):
    feature_array = df_feat[FEATURE_KEYS].to_numpy(dtype=np.float32)
    frame_ids = df_feat["frame_id"].to_numpy()

    num_frames = feature_array.shape[0]
    X_list, y_list = [], []

    force_all_fall = (fall_flag == 1 and fall_start_frame is None)

    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size
        window_feat = feature_array[start:end]
        window_frames = frame_ids[start:end]

        if fall_flag == 0:
            label = 0
        elif force_all_fall:
            label = 1
        else:
            label = 1 if (window_frames.min() <= fall_start_frame <= window_frames.max()) else 0

        X_list.append(window_feat)
        y_list.append(label)

    if not X_list:
        return None, None

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)

def build_windows_for_inference(df_feat, window_size=WINDOW_SIZE, step=WINDOW_STEP):
    feature_array = df_feat[FEATURE_KEYS].to_numpy(dtype=np.float32)
    frame_ids = df_feat["frame_id"].to_numpy()

    num_frames = feature_array.shape[0]
    X_list, frame_ranges = [], []

    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size
        X_list.append(feature_array[start:end])
        frame_ranges.append((int(frame_ids[start]), int(frame_ids[end - 1])))

    if not X_list:
        raise RuntimeError("Not enough frames to build any window for inference.")

    return np.stack(X_list, axis=0), frame_ranges


# ============================================================
# DATASET BUILDING (BY FILE)
# ============================================================

def load_dataset_by_file():
    labels_df = pd.read_csv(LABELS_CSV)
    file_col, label_col = get_label_columns(labels_df)

    file_records = []
    for _, row in labels_df.iterrows():
        fname = str(row[file_col])
        fall_flag = int(row[label_col])
        npy_path = os.path.join(DATA_DIR, fname)

        if not os.path.isfile(npy_path):
            print(f"[WARN] Missing data file {npy_path}, skipping.")
            continue

        print(f"[INFO] Processing {fname}, fall={fall_flag}")
        df_feat = extract_frame_features_for_file(npy_path)

        fall_start_frame = None
        if fall_flag == 1:
            fall_start_frame = find_fall_start_for_file(os.path.splitext(fname)[0])
            if fall_start_frame is None:
                print(f"[WARN] No fall-frame CSV for {fname}, labeling all windows as fall=1")

        X, y = build_windows_and_labels(df_feat, fall_flag, fall_start_frame)
        if X is None:
            print(f"[WARN] No windows for {fname}, skipping.")
            continue

        file_records.append((fname, X, y))

    if not file_records:
        raise RuntimeError("No usable data found for training.")

    return file_records

def split_train_val_by_file(file_records, val_fraction=VAL_FRACTION_BY_FILE):
    fnames = np.array([r[0] for r in file_records])
    groups = fnames.copy()
    idx = np.arange(len(file_records))

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=SEED)
    train_idx, val_idx = next(gss.split(idx, groups=groups))

    train_recs = [file_records[i] for i in train_idx]
    val_recs = [file_records[i] for i in val_idx]

    print(f"[INFO] File split: train_files={len(train_recs)}, val_files={len(val_recs)}")
    return train_recs, val_recs

def concat_records(records):
    X_all = np.concatenate([r[1] for r in records], axis=0)
    y_all = np.concatenate([r[2] for r in records], axis=0)
    return X_all, y_all


# ============================================================
# MODEL
# ============================================================

def residual_tcn_block(x, filters, kernel_size, dilation, dropout=DROPOUT):
    shortcut = x

    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("swish")(y)
    y = layers.SpatialDropout1D(dropout)(y)

    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(y)
    y = layers.BatchNormalization()(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([shortcut, y])
    out = layers.Activation("swish")(out)
    return out

def build_model(window_size, num_features):
    inputs = keras.Input(shape=(window_size, num_features), name="radar_window")

    x = layers.Conv1D(
        64, 3, padding="same", use_bias=False,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = residual_tcn_block(x, 64, 3, dilation=1, dropout=0.10)
    x = residual_tcn_block(x, 96, 3, dilation=2, dropout=0.12)
    x = residual_tcn_block(x, 128, 3, dilation=4, dropout=0.15)
    x = residual_tcn_block(x, 128, 3, dilation=8, dropout=0.15)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dense(
        128,
        activation="swish",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        64,
        activation="swish",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY)
    )(x)
    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=BASE_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ============================================================
# TF.DATA
# ============================================================

def make_tfdata(X, y, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if training:
        ds = ds.shuffle(min(len(X), 20000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    if thresholds.size == 0:
        return 0.5, 0.0

    # precision/recall arrays are one longer than thresholds
    f1s = []
    for p, r in zip(precision[:-1], recall[:-1]):
        denom = p + r
        f1 = 0.0 if denom <= 0 else (2.0 * p * r / denom)
        f1s.append(f1)

    f1s = np.asarray(f1s)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


# ============================================================
# TRAIN
# ============================================================

def train_main():
    if USE_MIXED_PRECISION:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[INFO] Mixed precision enabled.")
        except Exception as e:
            print("[WARN] Mixed precision not enabled:", e)

    file_records = load_dataset_by_file()
    train_recs, val_recs = split_train_val_by_file(file_records)

    X_train_raw, y_train = concat_records(train_recs)
    X_val_raw, y_val = concat_records(val_recs)

    num_train, T, F = X_train_raw.shape
    num_val = X_val_raw.shape[0]

    print(f"[INFO] Train windows={num_train}, Val windows={num_val}, window_size={T}, features={F}")

    scaler = StandardScaler()
    scaler.fit(X_train_raw.reshape(-1, F))

    X_train = scaler.transform(X_train_raw.reshape(-1, F)).reshape(num_train, T, F)
    X_val = scaler.transform(X_val_raw.reshape(-1, F)).reshape(num_val, T, F)

    np.savez(
        SCALER_PATH,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        feature_keys=np.array(FEATURE_KEYS),
        window_size=np.array([WINDOW_SIZE], dtype=np.int32),
    )
    print(f"[INFO] Saved scaler to {SCALER_PATH}")

    class_weights = None
    present_classes = np.unique(y_train)
    if USE_CLASS_WEIGHTS and len(present_classes) > 1:
        cw_arr = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=y_train
        )
        class_weights = {int(c): float(w) for c, w in zip(present_classes, cw_arr)}
    print("[INFO] Class weights:", class_weights)

    ds_train = make_tfdata(X_train, y_train, BATCH_SIZE, training=True)
    ds_val = make_tfdata(X_val, y_val, BATCH_SIZE, training=False)

    model = build_model(T, F)
    model.summary()

    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(TRAIN_LOG_PATH, append=False),
    ]

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    best_model = keras.models.load_model(BEST_MODEL_PATH, compile=False)
    best_model.save(MODEL_PATH)
    print(f"[INFO] Saved best model to {MODEL_PATH}")

    # Validation report with best threshold
    val_probs = best_model.predict(ds_val, verbose=0).ravel()
    best_thr, best_f1 = find_best_threshold(y_val, val_probs)
    val_pred = (val_probs >= best_thr).astype(np.int32)
    final_f1 = float(f1_score(y_val, val_pred)) if len(np.unique(y_val)) > 1 else 0.0

    report = {
        "best_threshold_by_f1": best_thr,
        "best_f1": best_f1,
        "f1_at_best_threshold": final_f1,
        "num_val_windows": int(len(y_val)),
        "val_positive_rate": float(np.mean(y_val)),
    }
    with open(VAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved validation report to {VAL_REPORT_PATH}")
    print(f"[INFO] Suggested threshold: {best_thr:.4f} | F1={best_f1:.4f}")

    # TFLite export
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Saved TFLite model to {TFLITE_PATH}")


# ============================================================
# INFERENCE HELPERS
# ============================================================

def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = keras.models.load_model(model_path, compile=False)

    data = np.load(scaler_path, allow_pickle=True)
    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2
    scaler.n_features_in_ = mean.shape[0]

    return model, scaler

def run_inference_on_npy(npy_path):
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"Input .npy file not found: {npy_path}")

    print(f"[INFO] Running inference on {npy_path}")
    model, scaler = load_model_and_scaler()

    threshold = 0.5
    if os.path.isfile(VAL_REPORT_PATH):
        try:
            with open(VAL_REPORT_PATH, "r", encoding="utf-8") as f:
                rep = json.load(f)
            threshold = float(rep.get("best_threshold_by_f1", 0.5))
        except Exception:
            pass

    df_feat = extract_frame_features_for_file(npy_path)
    X, frame_ranges = build_windows_for_inference(df_feat, WINDOW_SIZE, WINDOW_STEP)

    num_windows, window_size, num_features = X.shape
    print(f"[INFO] Built {num_windows} window(s) (window_size={window_size}, num_features={num_features})")

    X_scaled = scaler.transform(X.reshape(-1, num_features)).reshape(num_windows, window_size, num_features)
    probs = model.predict(X_scaled, verbose=0).ravel()

    for i, (p, fr) in enumerate(zip(probs, frame_ranges)):
        pred = int(p >= threshold)
        print(
            f"Window {i:03d} | frames [{fr[0]}, {fr[1]}] | "
            f"fall_prob = {p:.4f} | pred={pred}"
        )

    max_idx = int(np.argmax(probs))
    print("-" * 60)
    print(
        f"[SUMMARY] Max fall probability: {probs[max_idx]:.4f} "
        f"at window {max_idx} frames {frame_ranges[max_idx]}"
    )
    print(f"[SUMMARY] Suggested threshold used: {threshold:.4f}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train optimized fall-detection model or run inference on a .npy file."
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--npy", type=str, help="Path to .npy file for inference (required if --mode infer).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train_main()
    else:
        if not args.npy:
            raise SystemExit("ERROR: --npy path is required when --mode infer")
        run_inference_on_npy(args.npy)