"""
Live radar point cloud visualizer for AWR2944EVM using YOUR UART parser.

- Sends .cfg over CONFIG UART
- Uses your parse_frame_from_uart() to decode frames
- Builds Nx7 array: [frame_id, timestamp, x, y, z, doppler, snr]
- Visualizes 3D point cloud live, with SNR filter (400 < snr < 450)
- Also renders a doppler-range plot
"""

import struct
import time
import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------
# CONFIG
# -------------------------
CFG_PORT = "COM6"          # Config / CLI UART
CFG_BAUD = 115200
CFG_FILE = "radar_profile.cfg"

DATA_PORT = "COM3"         # Data UART
DATA_BAUD = 3125000

# SNR filter for visualization
SNR_MIN = 400.0
SNR_MAX = 10000.0
Y_MIN = 1

# Axis limits (tweak for your setup)
X_LIM = (-2.0, 2.0)
Y_LIM = (0.0, 4.0)
Z_LIM = (-2.0, 2.0)


# -------------------------
# YOUR PARSER (adapted)
# -------------------------
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

    Returns:
      frameNumber (int), ts (float), detections (list of dicts with x,y,z,doppler,snr)
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
    ) = struct.unpack('<IIIIIIII', header[8:8 + 32])

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
        tlv_type, tlv_len = struct.unpack('<II', payload[offset:offset + 8])
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
                snr = get_uint16(tlv_payload[start:start + 2])
                noise = get_uint16(tlv_payload[start + 2:start + 4])
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
    print(f"[UART] Frame {frameNumber} parsed, numDetObj={numDetectedObj}, got_points={len(detections)}, TLVs={numTLVs}")
    return frameNumber, ts, detections


# -------------------------
# Wrapper: make Nx7 array for visualizer
# -------------------------
def get_next_frame(ser):
    """
    Uses parse_frame_from_uart(ser) and converts detections into
    a NumPy array of shape (N, 7):

      [frame_id, timestamp, x, y, z, doppler, snr]
    """
    try:
        frame_num, ts, detections = parse_frame_from_uart(ser)
    except Exception as e:
        print(f"[WARN] parse_frame_from_uart error: {e}")
        return None

    if not detections:
        return None

    xs = [d["x"] for d in detections]
    ys = [d["y"] for d in detections]
    zs = [d["z"] for d in detections]
    vs = [d["doppler"] for d in detections]
    snrs = [d["snr"] for d in detections]
    rs = [np.sqrt((d["x"] ** 2) + (d["y"] ** 2) + (d["z"] ** 2)) for d in detections]

    frame_ids = [float(frame_num)] * len(detections)
    tss = [float(ts)] * len(detections)

    arr = np.column_stack([frame_ids, tss, xs, ys, zs, vs, snrs, rs]).astype(np.float32)
    return arr


# -------------------------
# Live visualizer
# -------------------------
def live_visualizer():
    # Open data UART
    ser_data = serial.Serial(DATA_PORT, DATA_BAUD, timeout=0.01)
    print(f"[INFO] Opening DATA port {DATA_PORT} @ {DATA_BAUD}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_zlim(Z_LIM)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Live Radar Point Cloud (400 < SNR < {int(SNR_MAX)})")

    last_pts = None  # last non-empty filtered frame

    def update(_):
        nonlocal last_pts

        pts = get_next_frame(ser_data)
        if pts is not None and pts.shape[0] > 0:
            # pts: [frame_id, ts, x, y, z, doppler, snr]
            snr = pts[:, 6]
            y = pts[:, 3]
            mask = (snr > SNR_MIN) & (snr < SNR_MAX) & (y > Y_MIN)
            filt = pts[mask]
            if filt.shape[0] > 0:
                last_pts = filt

        ax.cla()
        ax.set_xlim(X_LIM)
        ax.set_ylim(Y_LIM)
        ax.set_zlim(Z_LIM)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        if last_pts is None:
            ax.set_title("Waiting for data...")
            scat = ax.scatter([], [], [], s=1)
            return scat,

        frame_id = int(last_pts[0, 0])
        ts = float(last_pts[0, 1])
        x = last_pts[:, 2]
        y = last_pts[:, 3]
        z = last_pts[:, 4]
        snr = last_pts[:, 6]

        ax.set_title(
            f"Frame {frame_id} | t={ts:.2f}s | "
            f"{last_pts.shape[0]} pts (400 < SNR < {int(SNR_MAX)})"
        )

        # Color by SNR within this frame
        smin = snr.min()
        smax = snr.max()
        srange = smax - smin if smax != smin else 1.0
        colors = (snr - smin) / srange

        scat = ax.scatter(x, y, z, c=colors, cmap="viridis", s=8)
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        interval=100,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()
    ser_data.close()
    print("[INFO] Closed DATA UART")


if __name__ == "__main__":
    live_visualizer()