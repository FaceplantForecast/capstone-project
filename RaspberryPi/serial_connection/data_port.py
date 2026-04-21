# ==================================================================================
# Faceplant Forecast, 2025-2026
# This code handles collecting data from the data port and parsing it.
# It is also responsible for running the ML model and determining if a fall has
# occured or not.
# ==================================================================================

from serial_connection.parser_mmw_demo import parser_one_mmw_demo_output_packet as parse_guy
import sys
import os
import serial
import time
import numpy as np
import multiprocessing.shared_memory as sm
from sklearn.cluster import DBSCAN
import tensorflow as tf
import collections
import struct

#set parent directory so enums can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enums import PACKET_DATA, DEBUG_LEVEL as DEBUG, BUFF_SIZES, CMD_INDEX, DAT_PORT_STATUS, BOOT_MODE, RADAR_DATA, PLATFORM, APP_CMD

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

# ===============================================================
# WITHIN THESE EQUALS SIGNS, THE AI MODEL IS BORN! FUS RO DAH!
# Also it was all written by Charles "They Called Him Mr. AI Back In College" Marks

#region Config

#AI config
WINDOW_SIZE = 48
FEATURE_KEYS = [
    "cx", "cy", "cz", "height", "spread_xy",
    "mean_doppler", "num_points", "vz", "speed",
]

TFLITE_MODEL_PATH = "./model/model.tflite"
SCALER_PATH = "./model/scaler.npz"

FALL_THRESHOLD = 0.9
RESET_THRESHOLD = 0.6

# --- alert spam control ---
ALERT_COOLDOWN_SEC = 6.0   # minimum time between notifications
REARM_LOW_SEC = 2.0        # must stay <= RESET_THRESHOLD for this long to rearm

# --- Trend/ramp fall detection (for small spikes like 0.07) ---
TREND_ENABLE = False
TREND_WINDOW = 9        # History window for trend check (number of consecutive p values)
TREND_MIN_RISE = 0.2    # Require total rise across the window (0.02 -> 0.07 is +0.05)
TREND_MIN_SLOPE = 0.025 # Require average slope (rise/(window-1)); set ~ TREND_MIN_RISE/(TREND_WINDOW-1)
TREND_MIN_END = 0.38    # Require final p to reach at least this (prevents tiny drifts triggering)
TREND_ALLOW_DROPS = 1   # Allow some noise while still being "mostly increasing"
TREND_MIN_UP_STEPS = 7

# Near-range point removal (radome/self-reflection)
MIN_RANGE_M = 0.25

# --- SNR filtering (on snr_adj) ---
SNR_THRESHOLD_ADJ = 3.5
SNR_RELAX_DB = 3.0
MIN_POINTS_AFTER_SNR = 8
SNR_CLIP_MAX = 200.0

# --- Clustering knobs ---
DBSCAN_EPS_CANDIDATES = [0.25, 0.30, 0.36, 0.42, 0.50, 0.60]
DBSCAN_MIN_SAMPLES_FRAC = 0.06
DBSCAN_MIN_SAMPLES_MIN = 4
DBSCAN_MIN_SAMPLES_MAX = 12
MIN_POINTS_FALLBACK_ALL = 25

# --- Human-ish gating ---
MIN_HEIGHT_M = 0.15
MAX_SPREAD_XY_M = 3.5
MAX_RANGE_FOR_PERSON_M = 10.0  # set None to disable

# --- Temporal tracking ---
TRACK_MAX_DIST_M = 0.75
TRACK_BONUS = 1.0

# --- Centroid smoothing ---
CENTROID_EMA_ALPHA = 0.35

# --- Window gating ---
MIN_POINTS_FOR_WINDOW = 5
REQUIRE_HEIGHT_FOR_WINDOW = False

# --- Debug / logging ---
_DEBUG = False
MISS_PRINT_EVERY = 20
HEALTH_PRINT_EVERY_SEC = 5.0

# --- Background capture ---
BACKGROUND_WAIT_SEC = 10           # give people time to leave
BACKGROUND_CAPTURE_SEC = 10        # capture empty-room data for this long
BACKGROUND_MIN_FRAMES = 80         # minimum frames to accept baseline
BG_BIN_SIZE_M = 0.15
BG_MAX_RANGE_M = 20.0

# Gain correction (still useful after background subtraction)
TARGET_MEDIAN_SNR = 10.0
GAIN_ALPHA = 0.03
MAX_GAIN = 5.0

FALL_DETECTED = 0
#endregion

# ----------------- Trend detection helper -----------------
#region Trend Detection
def is_rising_trend(ps,
                    window=TREND_WINDOW,
                    min_rise=TREND_MIN_RISE,
                    min_slope=TREND_MIN_SLOPE,
                    min_end=TREND_MIN_END,
                    allow_drops=TREND_ALLOW_DROPS,
                    min_up_steps=TREND_MIN_UP_STEPS):
    """
    Detect a non-random rising ramp in probability history.

    ps: list/deque of recent p values (oldest -> newest)
    True if it looks like a real ramp instead of jitter.
    """
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
#endregion

# ----------------- Background Snapshot Compensator -----------------
#region Background Snapshot Compensator
class BackgroundSnapshotCompensator:
    """
    Two-phase compensator:
      Phase A (capture): accumulate per-bin SNR samples and build fixed bg_snr[bin] = median.
      Phase B (run): subtract fixed bg_snr[bin] from incoming SNR and apply gain correction.

    This ensures "background" doesn't learn the person.
    """
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
#endregion

# ----------------- TFLite + scaler -----------------
#region TFLite + Scaler
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
#endregion

# ----------------- Clustering Helpers -----------------
#region Clustering Helpers
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
#endregion

# ----------------- Feature Computation -----------------
#region Feature Computation
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
            self.prev_centroid_ema = (a * cx + (1 - a) * px,
                                      a * cy + (1 - a) * py,
                                      a * cz + (1 - a) * pz)
        return self.prev_centroid_ema

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

#endregion

# ----------------- Background Capture Phase -----------------
#region Background Capture
def build_background_from_stream(stream):
    global cmd_data

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
        x, y, z, snr = frame["x"], frame["y"], frame["z"], frame["snr"]

        r = np.sqrt(x * x + y * y + z * z)
        mask_r = r >= MIN_RANGE_M
        x, y, z, snr = x[mask_r], y[mask_r], z[mask_r], snr[mask_r]

        BG_COMP.capture_add(x, y, z, snr)

        frames += 1
        pts_total += int(snr.size)

    if frames < BACKGROUND_MIN_FRAMES:
        print(f"[BG WARN] Only captured {frames} frames (<{BACKGROUND_MIN_FRAMES}). "
              f"Background may be weak. Consider increasing BACKGROUND_CAPTURE_SEC.")
    else:
        print(f"[BG] Captured {frames} frames, pts_total={pts_total}")

    BG_COMP.build_background()
    print("[BG] Background baseline built. Beginning live inference.\n")
    cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.RUNNING
#endregion
# ===============================================================
#region Helpers
def bootstrapper():
    """
    This function handles the startup sequence for the process
    """
    global cmd_buffer
    global cmd_data
    global radar_buffer
    global radar_data

    #create the buffer, give it a name, set create to False, and give the size in bytes
    cmd_buffer = sm.SharedMemory("cmd_buffer", create=False)
    # Create the data, which is the array that is accessed by this script.
    # By setting the buffer, it can be accessed by other scripts as well
    cmd_data = np.ndarray(  shape=(BUFF_SIZES.CMD_BUFF,),
                            dtype=np.int8,
                            buffer=cmd_buffer.buf)
    
    radar_buffer = sm.SharedMemory("radar_buffer", create=False)
    radar_data = np.ndarray(shape=(BUFF_SIZES.RADAR_LEN,),
                            dtype=np.int64,
                            buffer=radar_buffer.buf)
    
def live_visualizer():
    from test_process import live_visualizer as demo_vis

    #debugging on desktop
    #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

    #debugging on laptop
    data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

    demo_vis(data_port)

def check_dropped_frames():
    #debugging on desktop
    #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

    #debugging on laptop
    data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

    stream_frames(data_port, mode=BOOT_MODE.DEMO_DROPPED_FRAMES)
#endregion

#====================================STREAMING CODE====================================

# ----------------- UART parsing (BUFFERED) -----------------
#region UART Parsing
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_LEN = 40

def get_uint16(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)

class UARTFrameParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        if data:
            self.buf.extend(data)

    def _discard_until_magic(self) -> bool:
        idx = self.buf.find(MAGIC_WORD)
        if idx == -1:
            keep = len(MAGIC_WORD) - 1
            if len(self.buf) > keep:
                self.buf = self.buf[-keep:]
            return False
        if idx > 0:
            del self.buf[:idx]
        return True

    def next_frame(self):
        if not self._discard_until_magic():
            return None
        if len(self.buf) < HEADER_LEN:
            return None

        totalPacketLen = int.from_bytes(self.buf[12:16], "little", signed=False)
        if totalPacketLen < HEADER_LEN or totalPacketLen > 65535:
            del self.buf[:1]
            return None
        if len(self.buf) < totalPacketLen:
            return None

        pkt = bytes(self.buf[:totalPacketLen])
        del self.buf[:totalPacketLen]

        header = pkt[:HEADER_LEN]
        payload = pkt[HEADER_LEN:]

        (
            version,
            totalPacketLen2,
            platform,
            frameNumber,
            timeCpuCycles,
            numDetectedObj,
            numTLVs,
            subFrameNumber
        ) = struct.unpack('<IIIIIIII', header[8:8 + 32])

        xs, ys, zs, vs = [], [], [], []
        snrs = []

        offset = 0
        for _ in range(numTLVs):
            if offset + 8 > len(payload):
                break
            tlv_type, tlv_len = struct.unpack('<II', payload[offset:offset + 8])
            offset += 8

            tlv_payload = payload[offset:offset + tlv_len]
            offset += tlv_len

            if tlv_type == 1 and numDetectedObj > 0:
                point_size = 16
                expected = numDetectedObj * point_size
                if len(tlv_payload) < expected:
                    return None
                for i in range(numDetectedObj):
                    start = i * point_size
                    x, y, z, v = struct.unpack('<ffff', tlv_payload[start:start + point_size])
                    xs.append(float(x))
                    ys.append(float(y))
                    zs.append(float(z))
                    vs.append(float(v))

            elif tlv_type == 7 and numDetectedObj > 0:
                point_size = 4
                expected = numDetectedObj * point_size
                usable = min(len(tlv_payload), expected)
                for i in range(usable // point_size):
                    start = i * point_size
                    snr = get_uint16(tlv_payload[start:start + 2])
                    snrs.append(float(snr))

        n = len(xs)
        if len(snrs) < n:
            snrs.extend([0.0] * (n - len(snrs)))

        detections = []
        for i in range(n):
            detections.append({
                "x": xs[i],
                "y": ys[i],
                "z": zs[i],
                "doppler": vs[i],
                "snr": snrs[i],
            })

        ts = time.time()
        return frameNumber, ts, detections

def uart_frame_stream(ser):

    ser.reset_input_buffer()
    print(f"[UART] Listening...")

    parser = UARTFrameParser()

    read_size = 16384
    last_tune = time.time()

    while True:
        chunk = ser.read(read_size)
        if chunk:
            parser.feed(chunk)

        now = time.time()
        if now - last_tune > 2.0:
            read_size = 8192 if read_size == 16384 else 16384
            last_tune = now

        out = parser.next_frame()
        if out is None:
            continue

        fid, ts, dets = out

        x = np.array([d["x"] for d in dets], dtype=np.float32)
        y = np.array([d["y"] for d in dets], dtype=np.float32)
        z = np.array([d["z"] for d in dets], dtype=np.float32)
        dop = np.array([d["doppler"] for d in dets], dtype=np.float32)
        snr = np.array([d["snr"] for d in dets], dtype=np.float32)

        yield dict(frame_id=fid, timestamp=ts, x=x, y=y, z=z, doppler=dop, snr=snr)
#endregion

# ----------------- Live Loop -----------------
#region Live Loop
def handle_runtime_commands(stream, window, track, p_hist):
    """Handle in-band APP commands while streaming frames.

    REDO_BACKGROUND_SCAN pauses inference, rebuilds the background baseline, and
    resets runtime model state before normal frame processing resumes.
    """
    global cmd_data
    global FALL_DETECTED

    if cmd_data[CMD_INDEX.APP_CMD] != APP_CMD.REDO_BACKGROUND_SCAN:
        return

    print("[CMD] REDO_BACKGROUND_SCAN received. Pausing inference and rebuilding baseline...")
    cmd_data[CMD_INDEX.APP_CMD] = APP_CMD.NONE

    # Clear latched outputs/state so no stale fall output leaks across recalibration.
    radar_data[RADAR_DATA.FALL_DETECTED] = 0
    radar_data[RADAR_DATA.PROBABILITY] = 0
    FALL_DETECTED = 0

    window.clear()
    p_hist.clear()
    track.prev_feat = None
    track.prev_centroid_ema = None

    build_background_from_stream(stream)

def live_loop(stream):
    """
    Main loop for data collection and processing.
    """
    global FALL_DETECTED

    window = collections.deque(maxlen=WINDOW_SIZE)
    track = TrackState()

    miss_count = 0
    total_frames = 0
    valid_frames = 0
    last_health = time.time()

    # anti-spam state
    last_alert_ts = -1e9
    low_start_ts = None

    # --- NEW: probability history for trend detection ---
    p_hist = collections.deque(maxlen=50)

    print("[LIVE] Starting fall detection...")

    for frame in stream:
        handle_runtime_commands(stream, window, track, p_hist)

        total_frames += 1
        feat = compute_frame_features(frame, track)

        """
        DEPRECATED
        if feat is None:
            miss_count += 1
            if _DEBUG or (miss_count % MISS_PRINT_EVERY == 0):
                print(f"        (no valid person cluster) misses={miss_count}")
        else:
            valid_frames += 1
<<<<<<< HEAD
            miss_count = 0
=======
            miss_count = 0"""
>>>>>>> d6b9de695a86fc0ad279df769f8425a76fe737de

        vec = np.array([feat[k] for k in FEATURE_KEYS], np.float32)
        window.append(vec)

        now = time.time()
        if now - last_health >= HEALTH_PRINT_EVERY_SEC:
            vr = 100.0 * (valid_frames / max(total_frames, 1))
            #UNCOMMENT BELOW FOR DEBUGGING
            #print(f"[HEALTH] frames={total_frames} valid={valid_frames} ({vr:.1f}%) window={len(window)}/{WINDOW_SIZE}")
            last_health = now

        if len(window) < WINDOW_SIZE:
            continue

        arr = np.stack(window)
        p = run_inference(arr)

        ts_now = frame["timestamp"]
        p_hist.append(p)

        # rearm only after sustained low
        if p <= RESET_THRESHOLD:
            if low_start_ts is None:
                low_start_ts = ts_now
            if (ts_now - low_start_ts) >= REARM_LOW_SEC:
                FALL_DETECTED = 0
        else:
            low_start_ts = None

        # --- triggers ---
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

                #send fall flag to buffer for server
                radar_data[RADAR_DATA.FALL_DETECTED] = 1
                radar_data[RADAR_DATA.PROBABILITY] = int(p*100)

                last_alert_ts = ts_now
                FALL_DETECTED = 1
            #UNCOMMENT BELOW FOR DEBUGGING
            #else:
                #print(f"[ALERT] (latched/cooldown) reason={reason} p={p:.3f}")
        #UNCOMMENT BELOW FOR DEBUGGING
        #else:
            #print(f"[INFO] p_fall={p:.3f}")
#endregion
#========================================================================
#deprecated; for debug modes
def stream_frames(con, debug=DEBUG.NONE, mode=BOOT_MODE.STANDARD):
    local_frame_buffer = bytearray()

    #create profiling variables
    run_time_sec = 7200
    last_frame = 0
    dropped_frames = 0
    frame_count = 0
    start_time = time.time()
    
    #setting up for inferencing
    output = None
    window = collections.deque(maxlen=WINDOW_SIZE)
    prev = None

    while True:
        #read any available data
        data = con.read(4096)
        if not data:
            continue

        #append to buffer
        local_frame_buffer.extend(data)

        #parse frame
        while True:
            dropped_frame = False
            parsed_data = parse_guy(local_frame_buffer, len(local_frame_buffer))
            result = parsed_data[PACKET_DATA.RESULT] #TC_PASS or TC_FAIL
            num_bytes = parsed_data[PACKET_DATA.NUM_BYTES]

            if result == 0 and num_bytes > 0:
                #frame parse was successful
                num_det_obj = parsed_data[PACKET_DATA.NUM_DET_OBJ]
                det_x = parsed_data[PACKET_DATA.DET_X]
                det_y = parsed_data[PACKET_DATA.DET_Y]
                det_z = parsed_data[PACKET_DATA.DET_Z]
                det_v = parsed_data[PACKET_DATA.DET_V]
                det_range = parsed_data[PACKET_DATA.RANGE]
                frame_num = parsed_data[PACKET_DATA.FRAME_NUM]
                ts = time.time()
                snr = parsed_data[PACKET_DATA.SNR]
                dop = parsed_data[PACKET_DATA.RANGE]

                #place data into shared buffer
                radar_data[RADAR_DATA.FRAME_ID] = frame_num
                radar_data[RADAR_DATA.TIMESTAMP] = ts

                #check if frame was dropped
                if last_frame == 0 or last_frame > frame_num:
                    last_frame = frame_num
                elif last_frame == frame_num - 1:
                    last_frame += 1
                else:
                    dropped_frames += 1
                    last_frame = frame_num
                    dropped_frame = True

                #add relevant data to a dictionary
                output = dict(frame_id=frame_num, timestamp=ts, x=det_x, y=det_y, z=det_z,
                              doppler=dop, snr=snr)
                
                #remove frame from local buffer
                local_frame_buffer = local_frame_buffer[num_bytes:]

                #print info to console if debug mode is set
                if debug != DEBUG.NONE:
                    print(f"received frame number {frame_num} with {num_det_obj} objects and length {num_bytes} bytes ({dropped_frames} dropped frames)")
                    if debug == DEBUG.VERBOSE:
                        for guy in range(num_det_obj):
                            print(f"    Obj {guy+1}: x={det_x[guy]:.2f}, y={det_y[guy]:.2f}, z={det_z[guy]:.2f}, v={det_v[guy]:.2f}, range={det_range[guy]:.2f}")

                #process frame
                feat = compute_frame_features(output, prev)
                if feat is None:
                    miss_count += 1
                    if DEBUG or (miss_count % MISS_PRINT_EVERY == 0):
                        print(f"        (no valid person cluster) misses={miss_count}")
                    continue

                prev = feat
                vec = np.array([feat[k] for k in FEATURE_KEYS], np.float32)
                window.append(vec)
                if len(window) < WINDOW_SIZE:
                    print(f"        (filling window {len(window)}/{WINDOW_SIZE})")
                    continue

                arr = np.stack(window)  # shape (WINDOW_SIZE, num_features)
                p = run_inference(arr)

                # reset fall flag if model outputs 0.00 ---
                if p <= 0.2:
                    radar_data[RADAR_DATA.FALL_DETECTED] = 0

                if p > FALL_THRESHOLD:
                    print(f"[ALERT] Fall detected! p={p:.2f}")
                    radar_data[RADAR_DATA.FALL_DETECTED] = 1
                    radar_data[RADAR_DATA.PROBABILITY] = int(p*100)
                else:
                    print(f"[INFO] p_fall={p:.2f}")
            
                #increase frame count
                frame_count += 1
            
            else:
                #not a full frame
                break

        #check if enough time has passed to end profiling
        elapsed_time = time.time() - start_time
        if mode == BOOT_MODE.DEMO_DROPPED_FRAMES:
            if (elapsed_time >= run_time_sec):
                print(f"ELAPSED TIME (SEC): {elapsed_time}\nFRAMES PROCESSED: {frame_count}\nDROPPED FRAMES: {dropped_frames}")
                print(f"DROPPED FRAMES PER MIN: {dropped_frames / (elapsed_time/60)}")
                print(f"DROPPED FRAMES PER HOUR: {dropped_frames / (elapsed_time/3600)}")
                print(f"PERCENTAGE OF FRAMES DROPPED: {(dropped_frames / (dropped_frames + frame_count)) * 100}%")
                break
            elif dropped_frame == True:
                print(f"DROPPED FRAME AT RUN TIME (SEC): {elapsed_time}\n")

        

        #delay to not consume more resources than necessary
        #time.sleep(0.1)

def main():
    global cmd_data
    
    #initiate bootstrapper
    bootstrapper()

    cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.CONNECTING

    #launch in demo mode if needed
    if cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_VISUALIZER:
        live_visualizer()
    elif cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_DROPPED_FRAMES:
        check_dropped_frames()
    elif cmd_data[CMD_INDEX.BOOT_MODE] == BOOT_MODE.DEMO_CONNECTION_TEST:
        print("Not connecting to radar")
    else:
        try:
            match cmd_data[CMD_INDEX.PLATFORM]:
                case PLATFORM.RASPBERRY_PI:
                    #Adjust device names and baud rates (deployment on Raspberry Pi)
                    try:
                        data_port = serial.Serial('/dev/ttyACM1', 3125000, timeout=0.1)   # for data streaming
                    except:
                        data_port = serial.Serial('/dev/ttyUSB1', 3125000, timeout=0.1)   # sometimes switches, I don't know why
                case PLATFORM.FRITZ_LAPTOP:
                    print("Running on laptop")
                    #debugging on laptop
                    data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming
                case PLATFORM.FRITZ_DESKTOP:
                    #debugging on desktop
                    data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming
                case _:
                    #default to raspberry pi
                    try:
                        data_port = serial.Serial('/dev/ttyACM1', 3125000, timeout=0.1)   # for data streaming
                    except:
                        data_port = serial.Serial('/dev/ttyUSB1', 3125000, timeout=0.1)   # sometimes switches, I don't know why

            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.SETTING_UP
        except:
            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.ERROR
            print("uh oh")
            sys.exit()

        #stream the frames
        stream = uart_frame_stream(data_port)

        build_background_from_stream(stream)

        live_loop(stream)

if __name__ == "__main__":
    sys.exit(main())