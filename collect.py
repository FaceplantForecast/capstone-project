"""
collect_radar_data.py

Raspberry Pi script that:

  1) Sends a TI mmWave .cfg file over the CONFIG UART to start the AWR2944EVM
  2) Listens on the DATA UART
  3) Uses TI's UART parser (parser_one_mmw_demo_output_packet) to decode each frame
  4) Logs per-point data as:
       frame_id, timestamp, x, y, z, doppler, snr
  5) Saves CSV + NPY for offline training

"""

import os
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import serial

# *** ADJUST THIS IMPORT TO MATCH YOUR PROJECT STRUCTURE ***
# e.g. if your file is mmwave_parser.py in the same dir, use:
#   from mmwave_parser import parser_helper, parser_one_mmw_demo_output_packet, getUint32
from serial_connection.parser_mmw_demo import (
    parser_helper,
    parser_one_mmw_demo_output_packet,
    getUint32,
)


# -------------------------------
# Default ports / baudrates
# -------------------------------

CFG_PORT_DEFAULT = "COM6"   # Config / CLI UART
CFG_BAUD_DEFAULT = 115200

DATA_PORT_DEFAULT = "COM3"  # Data UART
DATA_BAUD_DEFAULT = 3125000

OUTPUT_DIR_DEFAULT = "./recordings"


# -------------------------------
# 1. Send .cfg to config UART
# -------------------------------

def send_cfg(cfg_port, cfg_baud, cfg_file):
    """
    Send a TI mmWave .cfg script to the radar over the CONFIG UART.

    - Skips empty lines and lines starting with '%'
    - Sends each line + '\n'
    - Expects cfg to contain 'sensorStart' near the end
    """
    print(f"[CFG] Opening {cfg_port} @ {cfg_baud} for config...")
    ser = serial.Serial(cfg_port, baudrate=cfg_baud, timeout=1.0)

    with open(cfg_file, "r") as f:
        lines = f.readlines()

    cmd_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('%'):
            continue
        cmd_lines.append(line)

    print(f"[CFG] Sending {len(cmd_lines)} cfg lines from {cfg_file}...")
    for line in cmd_lines:
        ser.write((line + "\n").encode("utf-8"))
        ser.flush()
        time.sleep(0.05)

        # Optional: read any response the radar prints
        resp = ser.read(ser.in_waiting or 1)
        if resp:
            try:
                s = resp.decode(errors="ignore").strip()
                if s:
                    print(f"[CFG RESP] {s}")
            except Exception:
                pass

    print("[CFG] Finished sending cfg. Radar should now be streaming.")
    ser.close()


# -------------------------------
# 2. Optional: send sensorStop
# -------------------------------

def send_sensor_stop(cfg_port, cfg_baud):
    """
    Send 'sensorStop' to the config port to stop the radar.
    """
    try:
        ser = serial.Serial(cfg_port, baudrate=cfg_baud, timeout=1.0)
        ser.write(b"sensorStop\n")
        ser.flush()
        time.sleep(0.1)
        resp = ser.read(ser.in_waiting or 1)
        if resp:
            try:
                s = resp.decode(errors="ignore").strip()
                if s:
                    print(f"[CFG RESP] {s}")
            except Exception:
                pass
        ser.close()
        print("[CFG] Sent sensorStop.")
    except Exception as e:
        print(f"[CFG] Failed to send sensorStop: {e}")


# -------------------------------
# 3. Data collection using TI parser
# -------------------------------

def collect_data_with_ti_parser(
    data_port,
    data_baud,
    max_frames=None,
    max_seconds=None,
):
    """
    Read the DATA UART, parse frames using TI's parser, and build per-point rows.

    Returns:
        rows: list of dicts with keys:
          frame_id, timestamp, x, y, z, doppler, snr
    """
    ser = serial.Serial(data_port, baudrate=data_baud, timeout=0.1)
    print(f"[DATA] Opened {data_port} @ {data_baud} baud")

    buf = bytearray()
    rows = []
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # Read some bytes from UART
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)

            # Try to parse as many full packets as are in the buffer
            while True:
                if len(buf) < 40:
                    # Not enough data for a header
                    break

                # Use TI helper to find header and packet length
                headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, frameNumber = \
                    parser_helper(buf, len(buf), debug=False)

                if headerStartIndex == -1 or totalPacketNumBytes <= 0:
                    # No magic word / invalid header found yet.
                    # To avoid unbounded growth, keep only the last few KB.
                    if len(buf) > 8192:
                        buf = buf[-8192:]
                    break

                # If we don't yet have the entire packet, wait for more data
                if len(buf) < headerStartIndex + totalPacketNumBytes:
                    # Incomplete frame, need more bytes
                    break

                # Extract one full packet
                frame_bytes = buf[headerStartIndex: headerStartIndex + totalPacketNumBytes]

                # Drop consumed bytes from buffer
                buf = buf[headerStartIndex + totalPacketNumBytes:]

                # Parse this packet using TI's full parser
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
                 detectedNoise_array,
                 frameNumber_array) = parser_one_mmw_demo_output_packet(
                    frame_bytes,
                    len(frame_bytes),
                    debug=False,
                )

                if result != 0 or numDetObj2 <= 0:
                    # Failed frame or no detections, skip to next
                    continue

                # Extract frameNumber from header (bytes 20..24 relative to header start)
                frame_id = getUint32(frame_bytes[20:24])
                ts = time.time()
                frame_count += 1

                # Append one row per detected object
                for x, y, z, v, snr in zip(
                    detectedX_array,
                    detectedY_array,
                    detectedZ_array,
                    detectedV_array,
                    detectedSNR_array,
                ):
                    rows.append({
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "doppler": float(v),  # TI parser calls this 'v'
                        "snr": float(snr),    # in 0.1 dB units
                    })

                # Progress print every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"\r[DATA] Frames: {frame_count}, Points: {len(rows)}, "
                        f"Elapsed: {elapsed:.1f}s",
                        end="",
                        flush=True,
                    )

                # Check stop-by-frames
                if max_frames is not None and frame_count >= max_frames:
                    print("\n[DATA] Reached max_frames, stopping.")
                    raise KeyboardInterrupt  # break outer loop cleanly

            # Outer stop-by-time
            if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                print("\n[DATA] Reached max_seconds, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[DATA] Stopped by user or by frame limit.")

    finally:
        ser.close()
        print("[DATA] UART closed.")

    print(f"[DATA] Collected {frame_count} frames, {len(rows)} points.")
    return rows


# -------------------------------
# 4. Save recording (CSV + NPY)
# -------------------------------

def save_recording(rows, output_dir):
    """
    Save collected rows as CSV and NPY in output_dir.
    Returns (csv_path, npy_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not rows:
        print("[WARN] No data to save.")
        return None, None
    
    base_name = "leg_swing_10"
    csv_path = os.path.join(output_dir, base_name + ".csv")
    npy_path = os.path.join(output_dir, base_name + ".npy")

    df = pd.DataFrame(rows)
    df = df[["frame_id", "timestamp", "x", "y", "z", "doppler", "snr"]]

    df.to_csv(csv_path, index=False)
    print(f"[SAVE] CSV: {csv_path}")

    rec = df.to_records(index=False)
    np.save(npy_path, rec)
    print(f"[SAVE] NPY: {npy_path}")

    return csv_path, npy_path


# -------------------------------
# 5. Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Start AWR2944EVM via .cfg, collect radar data using TI parser, save to .npy"
    )
    parser.add_argument(
        "--cfg-port", default=CFG_PORT_DEFAULT,
        help=f"Config UART port (default: {CFG_PORT_DEFAULT})"
    )
    parser.add_argument(
        "--cfg-baud", type=int, default=CFG_BAUD_DEFAULT,
        help=f"Config UART baud (default: {CFG_BAUD_DEFAULT})"
    )
    parser.add_argument(
        "--data-port", default=DATA_PORT_DEFAULT,
        help=f"Data UART port (default: {DATA_PORT_DEFAULT})"
    )
    parser.add_argument(
        "--data-baud", type=int, default=DATA_BAUD_DEFAULT,
        help=f"Data UART baud (default: {DATA_BAUD_DEFAULT})"
    )
    parser.add_argument(
        "--cfg-file", required=True,
        help="Path to TI mmWave .cfg file to send to radar"
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="Max frames to collect (default: unlimited)"
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="Max seconds to collect (default: unlimited)"
    )
    parser.add_argument(
        "--outdir", default=OUTPUT_DIR_DEFAULT,
        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})"
    )
    parser.add_argument(
        "--no-stop", action="store_true",
        help="Do NOT send sensorStop at the end"
    )

    args = parser.parse_args()

    # 1) Start radar via cfg
    send_cfg(args.cfg_port, args.cfg_baud, args.cfg_file)

    # 2) Collect data from data UART using TI parser
    rows = collect_data_with_ti_parser(
        data_port=args.data_port,
        data_baud=args.data_baud,
        max_frames=args.frames,
        max_seconds=args.seconds,
    )

    # 3) Stop radar (optional)
    if not args.no_stop:
        send_sensor_stop(args.cfg_port, args.cfg_baud)

    # 4) Save recording
    save_recording(rows, output_dir=args.outdir)


if __name__ == "__main__":
    main()