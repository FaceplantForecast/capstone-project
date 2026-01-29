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

#set parent directory so enums can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from enums import PACKET_DATA, DEBUG_LEVEL as DEBUG, BUFF_SIZES, CMD_INDEX, DAT_PORT_STATUS, BOOT_MODE, RADAR_DATA

#global variables so that all functions modify the same instances
global cmd_buffer
global cmd_data
global radar_buffer
global radar_data

# ===============================================================
# WITHIN THESE EQUALS SIGNS, THE AI MODEL IS BORN! FUS RO DAH!
# Also it was all written by Charles "They Called Him Mr. AI Back In College" Marks

#AI config
WINDOW_SIZE = 48
FEATURE_KEYS = [
    "cx", "cy", "cz", "height", "spread_xy",
    "mean_doppler", "num_points", "vz", "speed",
]

TFLITE_MODEL_PATH = "./model/model.tflite"
SCALER_PATH = "./model/scaler.npz"     # from training script
FALL_THRESHOLD = 0.95

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
        np.array(frame["x"]), np.array(frame["y"]), np.array(frame["z"]),
        np.array(frame["doppler"]), np.array(frame["snr"]), np.array(frame["timestamp"])
    )

    # SNR filter: same numeric threshold as training (SNR_THRESHOLD = 5.0)
    mask = np.array(snr) > 5.0
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
# ===============================================================

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
            #Adjust device names and baud rates (deployment on Raspberry Pi)
            #data_port = serial.Serial('/dev/ttyUSB1', 3125000, timeout=0.1)   # for data streaming

            #debugging on laptop
            data_port = serial.Serial('COM3', 3125000, timeout=0.1)   # for data streaming

            #debugging on desktop
            #data_port = serial.Serial('COM4', 3125000, timeout=0.1)   # for data streaming

            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.RUNNING
        except:
            cmd_data[CMD_INDEX.DAT_PORT_STATUS] = DAT_PORT_STATUS.ERROR
            print("uh oh")

        #stream the frames
        stream_frames(data_port, DEBUG.VERBOSE)

if __name__ == "__main__":
    sys.exit(main())