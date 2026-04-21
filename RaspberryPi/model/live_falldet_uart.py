#!/usr/bin/env python3

import time
import collections
import numpy as np
from sklearn.cluster import DBSCAN
import serial
import tensorflow as tf
import os
import asyncio
import websockets
import json

from mmwave_parser import (
    parser_helper,
    parser_one_mmw_demo_output_packet,
    getUint32,
)

CFG_PORT = "COM5"
CFG_BAUD = 115200
CFG_FILE = "radar_profile.cfg"

UART_PORT = "COM6"
UART_BAUD = 3125000

WINDOW_SIZE = 48
FEATURE_KEYS = [
    "cx", "cy", "cz", "height", "spread_xy",
    "mean_doppler", "num_points", "vz", "speed",
]

TFLITE_MODEL_PATH = "model.tflite"
SCALER_PATH = "scaler.npz"

FALL_THRESHOLD = 0.90
RESET_THRESHOLD = 0.6

ALERT_COOLDOWN_SEC = 6.0
REARM_LOW_SEC = 2.0

TREND_ENABLE = False
TREND_WINDOW = 7
TREND_MIN_RISE = 0.03
TREND_MIN_SLOPE = 0.009
TREND_MIN_END = 0.06
TREND_ALLOW_DROPS = 2
TREND_MIN_UP_STEPS = 3

MIN_RANGE_M = 0.25

SNR_THRESHOLD_ADJ = 3.5
SNR_RELAX_DB = 3.0
MIN_POINTS_AFTER_SNR = 8
SNR_CLIP_MAX = None   # match collector behavior

DBSCAN_EPS_CANDIDATES = [0.25, 0.30, 0.36, 0.42, 0.50, 0.60]
DBSCAN_MIN_SAMPLES_FRAC = 0.06
DBSCAN_MIN_SAMPLES_MIN = 4
DBSCAN_MIN_SAMPLES_MAX = 12
MIN_POINTS_FALLBACK_ALL = 25

MIN_HEIGHT_M = 0.01
MAX_SPREAD_XY_M = 3.5
MAX_RANGE_FOR_PERSON_M = 20

TRACK_MAX_DIST_M = 0.75
TRACK_BONUS = 1.0

CENTROID_EMA_ALPHA = 0.35

MIN_POINTS_FOR_WINDOW = 5
REQUIRE_HEIGHT_FOR_WINDOW = False

DEBUG = False
MISS_PRINT_EVERY = 20
HEALTH_PRINT_EVERY_SEC = 5.0

BACKGROUND_WAIT_SEC = 10
BACKGROUND_CAPTURE_SEC = 10
BACKGROUND_MIN_FRAMES = 80
BG_BIN_SIZE_M = 0.15
BG_MAX_RANGE_M = 20.0

# match collector
TARGET_MEDIAN_SNR = 10.0
GAIN_ALPHA = 0.03
MAX_GAIN = 5.0

FALL_DETECTED = 0

BASE_GCP_URL = "gcr-ws-482782751069.us-central1.run.app/ws"
AUTH_TOKEN = "M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE"
GCP_WSS_URL = f"wss://{BASE_GCP_URL}?role=pi&token={AUTH_TOKEN}"


def is_rising_trend(ps,
                    window=TREND_WINDOW,
                    min_rise=TREND_MIN_RISE,
                    min_slope=TREND_MIN_SLOPE,
                    min_end=TREND_MIN_END,
                    allow_drops=TREND_ALLOW_DROPS,
                    min_up_steps=TREND_MIN_UP_STEPS):
    if len(ps) < window:
        return False

    w = np.array(list(ps)[-window:], dtype=np.float32)

    rise = float(w[-1] - w[0])
    if rise < min_rise:
        return False

    slope = rise / max(window - 1, 1)
    if slope < min_slope:
        return False

    if float(w[-1]) < min_end:
        return False

    diffs = np.diff(w)
    up_steps = int(np.sum(diffs > 0))
    down_steps = int(np.sum(diffs < 0))

    if up_steps < min_up_steps:
        return False
    if down_steps > allow_drops:
        return False

    return True


class BackgroundSnapshotCompensator:
    def __init__(
        self,
        bin_size_m=BG_BIN_SIZE_M,
        max_range_m=BG_MAX_RANGE_M,
        target_median_snr=TARGET_MEDIAN_SNR,
        gain_alpha=GAIN_ALPHA,
        max_gain=MAX_GAIN,
    ):
        self.bin_size_m = float(bin_size_m)
        self.max_range_m = float(max_range_m)
        self.target_median_snr = float(target_median_snr)
        self.gain_alpha = float(gain_alpha)
        self.max_gain = float(max_gain)

        self.n_bins = int(np.ceil(self.max_range_m / self.bin_size_m))
        self.bg_snr = np.zeros(self.n_bins, dtype=np.float32)
        self.gain_ema = 1.0

        self._cap = [list() for _ in range(self.n_bins)]
        self._is_built = False

    def _range_bins(self, x, y, z):
        r = np.sqrt(x * x + y * y + z * z)
        idx = np.floor(r / self.bin_size_m).astype(np.int32)
        idx = np.clip(idx, 0, self.n_bins - 1)
        return idx

    def capture_add(self, x, y, z, snr):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        snr = np.asarray(snr, dtype=np.float32)

        if snr.size == 0:
            return

        bins = self._range_bins(x, y, z)
        for b in np.unique(bins):
            vals = snr[bins == b]
            if vals.size:
                self._cap[b].append(float(np.median(vals)))

    def build_background(self):
        for b in range(self.n_bins):
            if len(self._cap[b]) > 0:
                self.bg_snr[b] = float(np.median(self._cap[b]))
            else:
                self.bg_snr[b] = 0.0
        self._is_built = True

    def clean_snr(self, x, y, z, snr):
        if snr.size == 0:
            return snr.astype(np.float32)
        if not self._is_built:
            return snr.astype(np.float32)

        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        z = z.astype(np.float32, copy=False)
        snr = snr.astype(np.float32, copy=False)

        bins = self._range_bins(x, y, z)

        snr_clean = snr - self.bg_snr[bins]
        snr_clean = np.maximum(0.0, snr_clean)

        pos = snr_clean[snr_clean > 0]
        if pos.size:
            med = float(np.median(pos))
            raw_gain = self.target_median_snr / (med + 1e-6)
            raw_gain = float(np.clip(raw_gain, 1.0 / self.max_gain, self.max_gain))
            self.gain_ema = (1.0 - self.gain_alpha) * self.gain_ema + self.gain_alpha * raw_gain

        return (snr_clean * self.gain_ema).astype(np.float32)


BG_COMP = BackgroundSnapshotCompensator()


def load_tflite_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    in_info = interpreter.get_input_details()[0]
    out_info = interpreter.get_output_details()[0]
    return interpreter, in_info["index"], out_info["index"]


def load_scaler(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    data = np.load(path, allow_pickle=True)
    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    return mean, scale


interpreter, input_index, output_index = load_tflite_model(TFLITE_MODEL_PATH)
SCALER_MEAN, SCALER_SCALE = load_scaler(SCALER_PATH)


def run_inference(window_array: np.ndarray) -> float:
    arr = window_array.astype(np.float32)
    arr = (arr - SCALER_MEAN) / SCALER_SCALE
    X = arr[np.newaxis, ...]
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_index)[0][0])


def send_cfg(cfg_port, cfg_baud, cfg_file):
    print(f"[CFG] Opening {cfg_port} @ {cfg_baud} ...")
    ser = serial.Serial(cfg_port, baudrate=cfg_baud, timeout=1.0)

    if not os.path.exists(cfg_file):
        raise FileNotFoundError(cfg_file)

    with open(cfg_file, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('%')]

    print(f"[CFG] Sending {len(lines)} commands from {cfg_file}...")
    for line in lines:
        ser.write((line + "\n").encode())
        ser.flush()
        time.sleep(0.05)
        resp = ser.read(ser.in_waiting or 1)
        if resp and DEBUG:
            try:
                txt = resp.decode(errors="ignore").strip()
                if txt:
                    print(f"[CFG RESP] {txt}")
            except Exception:
                pass
    ser.close()
    print("[CFG] Radar started.")


def send_sensor_stop(cfg_port, cfg_baud):
    try:
        ser = serial.Serial(cfg_port, baudrate=cfg_baud, timeout=1.0)
        ser.write(b"sensorStop\n")
        ser.flush()
        time.sleep(0.1)
        print("[CFG] Sent sensorStop.")
        ser.close()
    except Exception as e:
        print("[CFG] Stop failed:", e)


def frame_to_dataset_like_rows(frame, sample_frame_idx=1, use_background_subtraction=True):
    """
    Build the exact same per-point row structure as collect_radar_data.py:
      sample_frame, frame_id, timestamp, x, y, z, doppler, snr, snr_adj
    """
    frame_id = frame["frame_id"]
    ts = frame["timestamp"]
    x = frame["x"]
    y = frame["y"]
    z = frame["z"]
    doppler = frame["doppler"]
    snr = frame["snr"]

    rows = []

    if snr.size == 0:
        return rows

    r = np.sqrt(x * x + y * y + z * z)
    mask_r = r >= MIN_RANGE_M

    x_f = x[mask_r]
    y_f = y[mask_r]
    z_f = z[mask_r]
    doppler_f = doppler[mask_r]
    snr_f = snr[mask_r]

    if use_background_subtraction:
        snr_adj = BG_COMP.clean_snr(x_f, y_f, z_f, snr_f)
    else:
        snr_adj = snr_f.astype(np.float32)

    # keep identical to collector unless you intentionally want clipping
    if SNR_CLIP_MAX is not None:
        snr_adj = np.clip(snr_adj, 0.0, float(SNR_CLIP_MAX))

    for xi, yi, zi, vi, snri, snr_adji in zip(
        x_f, y_f, z_f, doppler_f, snr_f, snr_adj
    ):
        rows.append({
            "sample_frame": int(sample_frame_idx),
            "frame_id": int(frame_id),
            "timestamp": float(ts),
            "x": float(xi),
            "y": float(yi),
            "z": float(zi),
            "doppler": float(vi),
            "snr": float(snri),
            "snr_adj": float(snr_adji),
        })

    return rows


def preprocess_frame_like_dataset(frame, use_background_subtraction=True):
    """
    Convert a live frame into the same point-level representation used in the
    dataset collector, then return arrays for the live clustering/features code.
    """
    rows = frame_to_dataset_like_rows(
        frame,
        sample_frame_idx=1,
        use_background_subtraction=use_background_subtraction,
    )

    if not rows:
        return None

    x_f = np.array([r["x"] for r in rows], dtype=np.float32)
    y_f = np.array([r["y"] for r in rows], dtype=np.float32)
    z_f = np.array([r["z"] for r in rows], dtype=np.float32)
    doppler_f = np.array([r["doppler"] for r in rows], dtype=np.float32)
    snr_f = np.array([r["snr"] for r in rows], dtype=np.float32)
    snr_adj = np.array([r["snr_adj"] for r in rows], dtype=np.float32)

    return {
        "x": x_f,
        "y": y_f,
        "z": z_f,
        "doppler": doppler_f,
        "snr": snr_f,
        "snr_adj": snr_adj,
        "rows": rows,
    }


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
            score -= 2.0 * (spread_xy / MAX_SPREAD_XY_M)

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
                a * cz + (1 - a) * pz
            )
        return self.prev_centroid_ema


def compute_frame_features(frame, track: TrackState):
    t = frame["timestamp"]

    default_feat = dict(
        cx=0.0,
        cy=0.0,
        cz=0.0,
        height=0.0,
        spread_xy=0.0,
        mean_doppler=0.0,
        num_points=0,
        timestamp=float(t),
        vx=0.0,
        vy=0.0,
        vz=0.0,
        speed=0.0,
    )

    pre = preprocess_frame_like_dataset(frame, use_background_subtraction=True)
    if pre is None:
        track.prev_feat = default_feat
        return default_feat

    x = pre["x"]
    y = pre["y"]
    z = pre["z"]
    doppler = pre["doppler"]
    snr_adj = pre["snr_adj"]

    def _apply_thr(thr: float):
        mask = snr_adj > thr
        if not np.any(mask):
            return None
        xx, yy, zz = x[mask], y[mask], z[mask]
        dd = doppler[mask]
        pts = np.stack([xx, yy, zz], axis=1)
        return pts, dd

    out = _apply_thr(SNR_THRESHOLD_ADJ)

    if out is None:
        track.prev_feat = default_feat
        return default_feat

    points_xyz, dop_f = out

    if len(points_xyz) < MIN_POINTS_AFTER_SNR and SNR_RELAX_DB > 0:
        out2 = _apply_thr(max(0.0, SNR_THRESHOLD_ADJ - SNR_RELAX_DB))
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
        timestamp=float(t)
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


def uart_frame_stream(port, baud):
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
    print(f"[UART] Listening on {port} @ {baud}")

    buf = bytearray()

    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)

            while True:
                if len(buf) < 40:
                    break

                try:
                    headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber = \
                        parser_helper(buf, len(buf), debug=False)
                except Exception as e:
                    if DEBUG:
                        print(f"[PARSE WARN] parser_helper failed: {e}")

                    if len(buf) > 8192:
                        buf = buf[-8192:]
                    elif len(buf) > 1:
                        buf = buf[1:]
                    else:
                        buf = bytearray()
                    break

                if headerStartIndex == -1 or totalPacketNumBytes <= 0:
                    if len(buf) > 8192:
                        buf = buf[-8192:]
                    break

                if len(buf) < headerStartIndex + totalPacketNumBytes:
                    break

                frame_bytes = buf[headerStartIndex: headerStartIndex + totalPacketNumBytes]
                buf = buf[headerStartIndex + totalPacketNumBytes:]

                try:
                    (result,
                     hdrIdx2,
                     totalBytes2,
                     numDetObj2,
                     numTlv2,
                     subFrameNumber2,
                     detectedX_array,
                     detectedY_array,
                     detectedZ_array,
                     detectedV_array,
                     detectedRange_array,
                     detectedAzimuth_array,
                     detectedElevAngle_array,
                     detectedSNR_array,
                     detectedNoise_array) = parser_one_mmw_demo_output_packet(
                        frame_bytes,
                        len(frame_bytes),
                        debug=False,
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"[PARSE WARN] packet parse failed: {e}")
                    continue

                if result != 0:
                    continue

                try:
                    frame_id = getUint32(frame_bytes[20:24])
                except Exception:
                    frame_id = -1

                ts = time.time()

                x = np.asarray(detectedX_array, dtype=np.float32)
                y = np.asarray(detectedY_array, dtype=np.float32)
                z = np.asarray(detectedZ_array, dtype=np.float32)
                doppler = np.asarray(detectedV_array, dtype=np.float32)
                snr = np.asarray(detectedSNR_array, dtype=np.float32)

                yield {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "x": x,
                    "y": y,
                    "z": z,
                    "doppler": doppler,
                    "snr": snr,
                }
    finally:
        ser.close()
        print("[UART] Closed.")


def build_background_from_stream(stream):
    print(f"\n[BG] Please clear the room. Waiting {BACKGROUND_WAIT_SEC} seconds...")
    for i in range(BACKGROUND_WAIT_SEC, 0, -1):
        print(f"[BG] Starting background capture in {i:02d}s", end="\r", flush=True)
        time.sleep(1.0)
    print("\n[BG] Capturing background... (empty room)")

    start = time.time()
    frames = 0
    pts_total = 0

    while time.time() - start < BACKGROUND_CAPTURE_SEC:
        frame = next(stream)

        x = frame["x"]
        y = frame["y"]
        z = frame["z"]
        snr = frame["snr"]

        if snr.size == 0:
            frames += 1
            continue

        r = np.sqrt(x * x + y * y + z * z)
        mask_r = r >= MIN_RANGE_M
        x, y, z, snr = x[mask_r], y[mask_r], z[mask_r], snr[mask_r]

        BG_COMP.capture_add(x, y, z, snr)

        frames += 1
        pts_total += int(snr.size)

    if frames < BACKGROUND_MIN_FRAMES:
        print(
            f"[BG WARN] Only captured {frames} frames (<{BACKGROUND_MIN_FRAMES}). "
            f"Background may be weak. Consider increasing BACKGROUND_CAPTURE_SEC."
        )
    else:
        print(f"[BG] Captured {frames} frames, pts_total={pts_total}")

    BG_COMP.build_background()
    print("[BG] Background baseline built. Beginning live inference.\n")


async def _send_fall_flag_ws(probability: float, frame_id: int, ts: float):
    msg = {
        "msg_type" : "fall_event",
        "fall_detected": 1,
        "probability": float(probability),
        "frame_id": int(frame_id),
        "ts": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime(ts)),
        "device_id": "Main_PI"
    }

    try:
        async with websockets.connect(GCP_WSS_URL) as ws:
            await ws.send(json.dumps(msg))
            if DEBUG:
                print("[GCP] Sent fall flag payload")
    except Exception as e:
        print(f"[GCP] Failed to send fall flag via WSS: {e}")


def send_fall_flag(probability: float, frame_id: int, ts: float):
    try:
        asyncio.run(_send_fall_flag_ws(probability, frame_id, ts))
    except RuntimeError as e:
        print(f"[GCP] asyncio runtime error: {e}")


def live_loop(stream):
    global FALL_DETECTED

    window = collections.deque(maxlen=WINDOW_SIZE)
    track = TrackState()

    miss_count = 0
    total_frames = 0
    valid_frames = 0
    last_health = time.time()

    last_alert_ts = -1e9
    low_start_ts = None

    p_hist = collections.deque(maxlen=50)

    print("[LIVE] Starting fall detection...")

    for frame in stream:
        total_frames += 1
        feat = compute_frame_features(frame, track)

        if DEBUG:
            pre = preprocess_frame_like_dataset(frame, use_background_subtraction=True)
            if pre is not None and pre["rows"]:
                print("[DEBUG] First 5 dataset-like rows:")
                for row in pre["rows"][:5]:
                    print(row)

        if feat["num_points"] <= 0:
            miss_count += 1
            if DEBUG or (miss_count % MISS_PRINT_EVERY == 0):
                print(f"        (no valid person cluster) misses={miss_count}")
        else:
            valid_frames += 1
            miss_count = 0

        vec = np.array([feat[k] for k in FEATURE_KEYS], np.float32)
        window.append(vec)

        now = time.time()
        if now - last_health >= HEALTH_PRINT_EVERY_SEC:
            vr = 100.0 * (valid_frames / max(total_frames, 1))
            print(f"[HEALTH] frames={total_frames} valid={valid_frames} ({vr:.1f}%) window={len(window)}/{WINDOW_SIZE}")
            last_health = now

        if len(window) < WINDOW_SIZE:
            continue

        arr = np.stack(window)
        p = run_inference(arr)

        ts_now = frame["timestamp"]
        p_hist.append(p)

        if p <= RESET_THRESHOLD:
            if low_start_ts is None:
                low_start_ts = ts_now
            if (ts_now - low_start_ts) >= REARM_LOW_SEC:
                FALL_DETECTED = 0
        else:
            low_start_ts = None

        abs_trigger = (p > FALL_THRESHOLD)

        trend_trigger = False
        if TREND_ENABLE and (not abs_trigger):
            trend_trigger = is_rising_trend(
                p_hist,
                window=TREND_WINDOW,
                min_rise=TREND_MIN_RISE,
                min_slope=TREND_MIN_SLOPE,
                min_end=TREND_MIN_END,
                allow_drops=TREND_ALLOW_DROPS,
                min_up_steps=TREND_MIN_UP_STEPS,
            )

        should_alert = abs_trigger or trend_trigger

        if should_alert:
            reason = "ABS" if abs_trigger else "TREND"
            if (FALL_DETECTED == 0) and ((ts_now - last_alert_ts) >= ALERT_COOLDOWN_SEC):
                print(f"[ALERT] Fall detected! reason={reason} p={p:.3f}")
                send_fall_flag(probability=p, frame_id=frame["frame_id"], ts=ts_now)
                last_alert_ts = ts_now
                FALL_DETECTED = 1
            else:
                print(f"[ALERT] (latched/cooldown) reason={reason} p={p:.3f}")
        else:
            print(f"[INFO] p_fall={p:.3f}")


if __name__ == "__main__":
    try:
        send_cfg(CFG_PORT, CFG_BAUD, CFG_FILE)

        stream = uart_frame_stream(UART_PORT, UART_BAUD)

        build_background_from_stream(stream)

        live_loop(stream)

    except KeyboardInterrupt:
        print("\n[EXIT] User stopped.")
        print(f"[REPORT] Fall Detected={FALL_DETECTED}")
    finally:
        send_sensor_stop(CFG_PORT, CFG_BAUD)
        print("[DONE] Radar stopped cleanly.")