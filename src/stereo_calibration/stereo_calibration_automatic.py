import cv2
import numpy as np
import json
import time
import sys
import os
import argparse
import threading
from queue import Queue, Empty, Full
from metavision_core.event_io import EventsIterator

sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from datetime import datetime

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6       
CHECKERBOARD_COLS = 9       
SQUARE_SIZE_M = 0.0165      
COOLDOWN_SECONDS = 2.0      

DELTA_T_VAL = 30000         
MIN_EVENTS_THRESHOLD = 10 
MAX_SYNC_DIFF_US = 200  
BIAS_INCREMENT = 10     

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

FLAGS_SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

def camera_worker(serial, side, queue, stop_event, logger):
    """
    Thread dedicated to fetching events from a single camera as fast as possible.
    """
    try:
        mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T_VAL)
        device = mv_it.reader.device if hasattr(mv_it.reader, 'device') else mv_it.reader.get_device()
        
        # Configure Hardware Sync (Left=Slave, Right=Master)
        sync = device.get_i_camera_synchronization()
        if side == "LEFT":
            sync.set_mode_slave()
            logger.info(f"[{side}] Hardware set to SLAVE mode")
        else:
            sync.set_mode_master()
            logger.info(f"[{side}] Hardware set to MASTER mode")

        # Bias configuration
        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", biases.get("bias_diff_on") + BIAS_INCREMENT)
            biases.set("bias_diff_off", biases.get("bias_diff_off") + BIAS_INCREMENT)
            logger.info(f"[{side}] Biases increased (+{BIAS_INCREMENT})")

        for evs in mv_it:
            if stop_event.is_set():
                break
            
            # Filter Polarity 1 and check threshold
            evs_filt = evs[evs['p'] == 1]
            if evs_filt.size >= MIN_EVENTS_THRESHOLD:
                # Policy: Drop old data to keep latency at minimum
                if queue.full():
                    try: queue.get_nowait()
                    except Empty: pass
                queue.put((evs_filt, mv_it.get_current_time()))
                
    except Exception as e:
        logger.error(f"Worker {side} critical error: {e}")

def generate_point_heatmap(accumulated_mask):
    """ Generate a jet-colored heatmap based on accumulated corner detections """
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def load_prophesee_json(filename, logger):
    """ Load intrinsic parameters from config folder """
    filepath = os.path.join("config", filename)
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found!")
        sys.exit(1)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Support both Metavision 'K' and analyzed 'camera_matrix' formats
    k_raw = data.get("K", data.get("camera_matrix"))
    mtx = np.array(k_raw).reshape(3, 3).astype(np.float32)
    
    d_raw = data.get("D", data.get("dist_coeffs"))
    dist = np.array(d_raw).astype(np.float32)
    
    return mtx, dist, data["width"], data["height"]

def save_points_json(objpoints, imgpoints_L, imgpoints_R, width, height, logger):
    """ Save extracted points for offline RMS analysis """
    os.makedirs("data_analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("data_analysis", f"points_stereo_{timestamp}.json")
    
    data = {
        "width": width, "height": height,
        "objpoints": [op.tolist() for op in objpoints],
        "imgpoints_L": [ip.tolist() for ip in imgpoints_L],
        "imgpoints_R": [ip.tolist() for ip in imgpoints_R]
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)
    logger.info(f"Stereo points saved for analysis: {filepath}")

def save_stereo_params(filename, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F, width, height, logger):
    """ Save final stereo parameters to config folder """
    os.makedirs("config", exist_ok=True)
    filepath = os.path.join("config", filename)
    data = {
        "width": width, "height": height,
        "camera_left": {"K": mtx_L.tolist(), "D": dist_L.tolist()},
        "camera_right": {"K": mtx_R.tolist(), "D": dist_R.tolist()},
        "stereo": {"R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist()}
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Stereo calibration parameters saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Single-threaded Stereo Calibration")
    parser.add_argument("--mode", choices=["calibrate", "analyze"], required=True,
                        help="'calibrate' updates config, 'analyze' only saves points.")
    parser.add_argument("--headless", action="store_true", help="Run without UI windows")
    args = parser.parse_args()

    # Initialize logger
    logger = setup_logging("stereo_calibration")
    logger.info(f"Starting process in {args.mode.upper()} mode")

    # Load intrinsics
    mtx_L, dist_L, width_L, height_L = load_prophesee_json("camera_left.json", logger)
    mtx_R, dist_R, width_R, height_R = load_prophesee_json("camera_right.json", logger)
    width, height = width_L, height_L

    # Initialize Queues and Threads
    q_L, q_R = Queue(maxsize=2), Queue(maxsize=2)
    stop_event = threading.Event()
    
    t_L = threading.Thread(target=camera_worker, args=(SERIAL_LEFT, "LEFT", q_L, stop_event, logger))
    t_R = threading.Thread(target=camera_worker, args=(SERIAL_RIGHT, "RIGHT", q_R, stop_event, logger))
    
    # Start threads
    t_L.start()
    time.sleep(0.5)
    t_R.start()

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2) * SQUARE_SIZE_M

    objpoints, imgpoints_L, imgpoints_R = [], [], []
    valid_snaps, last_cap = 0, time.time()
    point_mask = np.zeros((height, width), dtype=np.float32)

    logger.info("Stereo acquisition started. Press 'q' to stop.")
    
    try:
        while not stop_event.is_set():
            # Mandatory GUI refresher for OpenCV
            if not args.headless:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            try:
                # Fetch synchronized packets from queues
                data_L = q_L.get(timeout=0.01)
                data_R = q_R.get(timeout=0.01)
            except Empty:
                continue

            ts_L = data_L[1]
            ts_R = data_R[1]

            # --- ACTIVE ALIGNMENT ---
            while abs(ts_L - ts_R) > MAX_SYNC_DIFF_US:
                logger.info(f"Syncing... Offset: {ts_L - ts_R}us")
                if ts_L < ts_R:
                    logger.debug(f"Flushing Left... diff: {ts_L - ts_R}")
                    try:
                        data_L = q_L.get(timeout=0.01)
                        ts_L = data_L[1]
                    except Empty: break 
                else:
                    logger.debug(f"Flushing Right... diff: {ts_L - ts_R}")
                    try:
                        data_R = q_R.get(timeout=0.01)
                        ts_R = data_R[1]
                    except Empty: break

            if abs(ts_L - ts_R) <= MAX_SYNC_DIFF_US:
                # Frame generation and processing
                im_L, im_R = np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8)
                im_L[data_L[0]['y'], data_L[0]['x']] = 255
                im_R[data_R[0]['y'], data_R[0]['x']] = 255
                
                if time.time() - last_cap >= COOLDOWN_SECONDS:
                    proc_L = cv2.bitwise_not(cv2.dilate(im_L, np.ones((3,3))))
                    proc_R = cv2.bitwise_not(cv2.dilate(im_R, np.ones((3,3))))

                    found_L = cv2.checkChessboard(proc_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS))
                    found_R = cv2.checkChessboard(proc_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS))

                    found_L = found_R = True

                    if found_L and found_R:
                        ret_L, corners_L = cv2.findChessboardCornersSB(proc_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)
                        ret_R, corners_R = cv2.findChessboardCornersSB(proc_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)

                        if ret_L and ret_R:
                            if time.time() - last_cap > COOLDOWN_SECONDS:
                                objpoints.append(objp)
                                imgpoints_L.append(corners_L)
                                imgpoints_R.append(corners_R)
                                valid_snaps += 1
                                last_cap = time.time()
                                for c in corners_L:
                                    cv2.circle(point_mask, (int(c[0][0]), int(c[0][1])), 2, 1.0, -1)
                                logger.info(f"Snap {valid_snaps} captured (Sync diff: {abs(ts_L - ts_R)}us)")

                if not args.headless:
                    # Prepare the combined frame (L | R)
                    combined = np.hstack((im_L, im_R))
                    # Define text parameters
                    text = f"Snaps: {valid_snaps}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    color = 255
                    position = (10, 20) # Top-left corner
                    # Put the text on the image
                    cv2.putText(combined, text, position, font, font_scale, color, thickness)
                    cv2.imshow("Stereo Stream", combined)
                    cv2.imshow("Heatmap", generate_point_heatmap(point_mask))
                    if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        stop_event.set()
        t_L.join()
        t_R.join()
        cv2.destroyAllWindows()

    # Final calibration logic
    if valid_snaps >= 15:
        save_points_json(objpoints, imgpoints_L, imgpoints_R, width, height, logger)
        if args.mode == "calibrate":
            logger.info("Starting stereo calibration with outlier rejection...")
            
            # Step 1: Extended calibration to identify residues
            res = cv2.stereoCalibrateExtended(
                objpoints, imgpoints_L, imgpoints_R,
                mtx_L, dist_L, mtx_R, dist_R,
                (width, height), flags=cv2.CALIB_FIX_INTRINSIC
            )
            ret, M1, D1, M2, D2, R, T, E, F, perViewErrors = res

            # Step 2: Error analysis (Mean + 1 StdDev)
            errors = np.mean(perViewErrors, axis=1).flatten()
            threshold = np.mean(errors) + np.std(errors)
            
            f_obj, f_L, f_R = [], [], []
            for i, err in enumerate(errors):
                if err < threshold:
                    f_obj.append(objpoints[i])
                    f_L.append(imgpoints_L[i])
                    f_R.append(imgpoints_R[i])
                else:
                    logger.info(f"Snap {i} discarded: RMS {err:.4f} > {threshold:.4f}")

            # Step 3: Final refined calibration
            logger.info(f"Refining with {len(f_obj)} valid snapshots...")
            ret_f, M1f, D1f, M2f, D2f, Rf, Tf, Ef, Ff = cv2.stereoCalibrate(
                f_obj, f_L, f_R, mtx_L, dist_L, mtx_R, dist_R,
                (width, height), flags=cv2.CALIB_FIX_INTRINSIC
            )

            logger.info(f"Final Stereo RMS: {ret_f:.4f} pixels (Original: {ret:.4f})")
            save_stereo_params("stereo_params.json", M1f, D1f, M2f, D2f, Rf, Tf, Ef, Ff, width, height, logger)
        else:
            logger.info("Mode 'analyze' complete. Parameters were not updated.")
    else:
        logger.error(f"Not enough snaps captured ({valid_snaps}/15).")

if __name__ == "__main__":
    main()
