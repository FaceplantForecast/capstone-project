#!/usr/bin/env python3
"""
Live fall detection over UART from AWR2944EVM.

Now also starts radar automatically using .cfg on CONFIG UART.

Pipeline:
    Config UART  -> send cfg (starts radar)
    Data UART    -> parse_frame_from_uart()
                  -> compute features -> scale features -> AI inference -> fall alert
                  -> send flag to GCP endpoint on fall (via WSS)

Written by Charles Marks
"""
#NOTE: The inferencing code in data_port.py was based on or fully copied from this file

import struct
import time
import collections
import numpy as np
from sklearn.cluster import DBSCAN
import serial
import tensorflow as tf
import os
import asyncio
import websockets  # pip install websockets

# CONFIG

CFG_PORT = "COM4"              # CLI UART (Change for PI)
CFG_BAUD = 115200
CFG_FILE = "radar_profile.cfg"

UART_PORT = "COM3"             # Data UART (Change for PI)
UART_BAUD = 3125000            

WINDOW_SIZE = 48
FEATURE_KEYS = [
    "cx", "cy", "cz", "height", "spread_xy",
    "mean_doppler", "num_points", "vz", "speed",
]

TFLITE_MODEL_PATH = "model.tflite"
SCALER_PATH = "scaler.npz"     # from training script
FALL_THRESHOLD = 0.95

FALL_DETECTED = 0

# GCP endpoint + auth token (WebSocket)
BASE_GCP_URL = "gcr-ws-482782751069.us-central1.run.app/ws"
# Original token you gave earlier (without the stray dash)
AUTH_TOKEN = "M8b4eFJHq2pI3V9nW5r0dE-PLZpQyX7uB1cTa9kN4mE"

# Full WSS URL with auth_token as query param
GCP_WSS_URL = f"wss://{BASE_GCP_URL}?role=pi&token={AUTH_TOKEN}"


#AI model + scaler setup (TFLite)

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
    mean = data["mean"].astype(np.float32)   # shape (num_features,)
    scale = data["scale"].astype(np.float32) # shape (num_features,)
    return mean, scale

interpreter, input_index, output_index = load_tflite_model(TFLITE_MODEL_PATH)
SCALER_MEAN, SCALER_SCALE = load_scaler(SCALER_PATH)


def run_inference(window_array: np.ndarray) -> float:
    """
    window_array: shape (WINDOW_SIZE, num_features), raw features

    We apply the same scaling as in training:
      X_scaled = (X - mean) / scale
    then run TFLite.
    """
    # Ensure float32
    arr = window_array.astype(np.float32)

    # Broadcast mean/scale: (T, F) - (F,) -> (T, F)
    arr = (arr - SCALER_MEAN) / SCALER_SCALE

    X = arr[np.newaxis, ...]  # (1, T, F)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_index)[0][0])


# Radar control helpers

def send_cfg(cfg_port, cfg_baud, cfg_file):
    """Send TI mmWave .cfg script to radar over CONFIG UART."""
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
        if resp:
            try:
                txt = resp.decode(errors="ignore").strip()
                if txt:
                    print(f"[CFG RESP] {txt}")
            except Exception:
                pass
    ser.close()
    print("[CFG] Radar started.")


def send_sensor_stop(cfg_port, cfg_baud):
    """Send 'sensorStop' command."""
    try:
        ser = serial.Serial(cfg_port, baudrate=cfg_baud, timeout=1.0)
        ser.write(b"sensorStop\n")
        ser.flush()
        time.sleep(0.1)
        print("[CFG] Sent sensorStop.")
        ser.close()
    except Exception as e:
        print("[CFG] Stop failed:", e)


# Feature computation

def cluster_frame_points(points_xyz, eps=0.30, min_samples=8):
    if len(points_xyz) == 0:
        return np.array([], dtype=int)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(points_xyz)


def pick_person_cluster(points_xyz, labels):
    if len(points_xyz) == 0:
        return np.zeros(len(points_xyz), dtype=bool)
    unique_labels, counts = np.unique(labels, return_counts=True)
    mask_valid = unique_labels != -1
    unique_labels = unique_labels[mask_valid]
    counts = counts[mask_valid]
    if len(unique_labels) == 0:
        return np.zeros(len(points_xyz), dtype=bool)
    best_label = unique_labels[np.argmax(counts)]
    return labels == best_label


def compute_frame_features(frame, prev_feat=None):
    """
    Compute the same features as in training:

      cx, cy, cz, height, spread_xy, mean_doppler, num_points, vz, speed

    We compute vx, vy, vz from centroid differences between *valid* frames,
    just like the training pipeline that uses df_feat rows only where a
    person cluster was found.
    """
    x, y, z, doppler, snr, t = (
        frame["x"], frame["y"], frame["z"],
        frame["doppler"], frame["snr"], frame["timestamp"]
    )

    # SNR filter: same numeric threshold as training (SNR_THRESHOLD = 5.0)
    mask = snr > 5.0
    if not np.any(mask):
        return None

    x, y, z, doppler = x[mask], y[mask], z[mask], doppler[mask]
    points_xyz = np.stack([x, y, z], axis=1)
    labels = cluster_frame_points(points_xyz)
    person_mask = pick_person_cluster(points_xyz, labels)
    if not np.any(person_mask):
        return None

    pts = points_xyz[person_mask]
    dop = doppler[person_mask]
    cx, cy, cz = pts.mean(axis=0)
    height = pts[:, 2].max() - pts[:, 2].min()
    spread_xy = max(pts[:, 0].max() - pts[:, 0].min(),
                    pts[:, 1].max() - pts[:, 1].min())
    mean_doppler = np.mean(dop)
    num_points = pts.shape[0]

    feat = dict(cx=cx, cy=cy, cz=cz, height=height,
                spread_xy=spread_xy, mean_doppler=mean_doppler,
                num_points=num_points, timestamp=t)

    if prev_feat:
        dt = max(t - prev_feat["timestamp"], 1e-3)
        vx = (feat["cx"] - prev_feat["cx"]) / dt
        vy = (feat["cy"] - prev_feat["cy"]) / dt
        vz = (feat["cz"] - prev_feat["cz"]) / dt
    else:
        vx = vy = vz = 0.0

    speed = float(np.sqrt(vx**2 + vy**2 + vz**2))

    feat.update(vx=vx, vy=vy, vz=vz, speed=speed)
    return feat

# UART parsing

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'  # TI mmWave magic

def get_uint16(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def find_magic(ser):
    """Block until we see the magic word in the UART stream."""
    sync = b''
    while True:
        b = ser.read(1)
        if not b:
            continue
        sync += b
        if len(sync) > len(MAGIC_WORD):
            sync = sync[-len(MAGIC_WORD):]
        if sync == MAGIC_WORD:
            return


def parse_frame_from_uart(ser):
    """
    Parse ONE frame from UART (mmwDemo-style).

    Header (40 bytes):
      0..7   : magic
      8..11  : version
      12..15 : totalPacketLen
      16..19 : platform
      20..23 : frameNumber
      24..27 : timeCpuCycles
      28..31 : numDetectedObj
      32..35 : numTLVs
      36..39 : subFrameNumber

    TLVs:
      type 1: detected points (x, y, z, v) as float32 -> 16 bytes per object
      type 7: snr, noise as uint16 -> 4 bytes per object
    """
    # 1) Sync to magic
    find_magic(ser)

    # 2) Header
    header_rest = ser.read(40 - len(MAGIC_WORD))
    if len(header_rest) < (40 - len(MAGIC_WORD)):
        raise IOError("Incomplete frame header")

    header = MAGIC_WORD + header_rest

    (
        version,
        totalPacketLen,
        platform,
        frameNumber,
        timeCpuCycles,
        numDetectedObj,
        numTLVs,
        subFrameNumber
    ) = struct.unpack('<IIIIIIII', header[8:8+32])

    # 3) Payload
    remaining = totalPacketLen - 40
    payload = ser.read(remaining)
    if len(payload) < remaining:
        raise IOError("Incomplete frame payload")

    xs, ys, zs, vs = [], [], [], []
    snrs, noises = [], []

    offset = 0
    for _ in range(numTLVs):
        if offset + 8 > len(payload):
            break
        tlv_type, tlv_len = struct.unpack('<II', payload[offset:offset+8])
        offset += 8
        tlv_payload = payload[offset:offset + tlv_len]
        offset += tlv_len

        if tlv_type == 1 and numDetectedObj > 0:
            # Detected points: x, y, z, v (4 floats) -> 16 bytes each
            point_size = 16
            expected = numDetectedObj * point_size
            if len(tlv_payload) < expected:
                raise ValueError(
                    f"Detected points TLV too short: {len(tlv_payload)} vs {expected}"
                )
            for i in range(numDetectedObj):
                start = i * point_size
                x, y, z, v = struct.unpack(
                    '<ffff',
                    tlv_payload[start:start + point_size]
                )
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
                vs.append(float(v))

        elif tlv_type == 7 and numDetectedObj > 0:
            # SNR + noise: 2x uint16 -> 4 bytes each
            point_size = 4
            expected = numDetectedObj * point_size
            usable = min(len(tlv_payload), expected)
            for i in range(usable // point_size):
                start = i * point_size
                snr = get_uint16(tlv_payload[start:start+2])
                noise = get_uint16(tlv_payload[start+2:start+4])
                snrs.append(float(snr))
                noises.append(float(noise))

        else:
            # ignore other TLVs
            continue

    # 4) Build detections
    detections = []
    n = len(xs)
    if len(snrs) < n:
        snrs.extend([0.0] * (n - len(snrs)))

    for i in range(n):
        detections.append({
            "x": xs[i],
            "y": ys[i],
            "z": zs[i],
            "doppler": vs[i],
            "snr": snrs[i],
        })

    ts = time.time()
    print(f"[UART] Frame {frameNumber} parsed, numDetObj={numDetectedObj}, "
          f"got_points={len(detections)}, TLVs={numTLVs}")
    return frameNumber, ts, detections


def uart_frame_stream(port, baud):
    ser = serial.Serial(port, baudrate=baud, timeout=0.5)
    print(f"[UART] Listening on {port} @ {baud}")
    while True:
        try:
            fid, ts, dets = parse_frame_from_uart(ser)
        except Exception as e:
            print("[UART] Parse error:", e)
            continue
        if not dets:
            print(f"[UART] Frame {fid} had 0 detections")
            continue
        x = np.array([d["x"] for d in dets], dtype=np.float32)
        y = np.array([d["y"] for d in dets], dtype=np.float32)
        z = np.array([d["z"] for d in dets], dtype=np.float32)
        dop = np.array([d["doppler"] for d in dets], dtype=np.float32)
        snr = np.array([d["snr"] for d in dets], dtype=np.float32)

        print(f"[FRAME] id={fid:<6d} points={len(dets):<3d} time={ts:.3f}")

        yield dict(frame_id=fid, timestamp=ts, x=x, y=y, z=z,
                   doppler=dop, snr=snr)


# GCP flag sender (WSS)

async def _send_fall_flag_ws(probability: float, frame_id: int, ts: float):
    """
    Internal coroutine: open WSS, send one JSON message, close.
    Includes auth_token both in query string and in the JSON body.
    """
    msg = {"fall_detected": 1, "ts": time.strftime("%m-%d-%Y %H:%M:%S", time.localtime())}

    print(f"[GCP] Connecting to {GCP_WSS_URL} ...")
    try:
        async with websockets.connect(GCP_WSS_URL) as ws:
            # Send payload
            await ws.send(json.dumps(msg))
            print("[GCP] Sent fall flag payload")

            # Optional: try to read a response (if server replies)
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print(f"[GCP] Received reply: {reply}")
            except asyncio.TimeoutError:
                print("[GCP] No reply from server (timeout), assuming fire-and-forget.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[GCP] WebSocket closed: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"[GCP] Failed to send fall flag via WSS: {e}")


# We still expose a synchronous function to call from live_loop
import json  # needed for dumps above

def send_fall_flag(probability: float, frame_id: int, ts: float):
    """
    Public function used in live_loop.
    Wraps the async WebSocket sender in asyncio.run.
    """
    try:
        asyncio.run(_send_fall_flag_ws(probability, frame_id, ts))
    except RuntimeError as e:
        # In case there's already a running event loop (unlikely in this script),
        # you could adapt this to use create_task instead.
        print(f"[GCP] asyncio runtime error: {e}")

# Live loop
def live_loop(stream):
    global FALL_DETECTED   # we modify this
    window = collections.deque(maxlen=WINDOW_SIZE)
    prev = None
    print("[LIVE] Starting fall detection...")
    for frame in stream:
        feat = compute_frame_features(frame, prev)
        if feat is None:
            print("        (no valid person cluster in this frame)")
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
            FALL_DETECTED = 0

        if p > FALL_THRESHOLD:
            print(f"[ALERT] Fall detected! p={p:.2f}")
            if FALL_DETECTED == 0:
                # Only send once (first detection)
                send_fall_flag(probability=p,
                               frame_id=frame["frame_id"],
                               ts=frame["timestamp"])
            FALL_DETECTED = 1
        else:
            print(f"[INFO] p_fall={p:.2f}")


# Main

if __name__ == "__main__":
    try:
        send_cfg(CFG_PORT, CFG_BAUD, CFG_FILE)
        live_loop(uart_frame_stream(UART_PORT, UART_BAUD))
    except KeyboardInterrupt:
        print("\n[EXIT] User stopped.")
        print(f"[REPORT] Fall Detected={FALL_DETECTED}")
    finally:
        send_sensor_stop(CFG_PORT, CFG_BAUD)
        print("[DONE] Radar stopped cleanly.")
