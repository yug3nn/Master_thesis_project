import cv2
import numpy as np
import json
import time
import os
import argparse
from metavision_core.event_io import EventsIterator
from datetime import datetime

# Adds the current working directory (project root) to sys.path
sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6     
CHECKERBOARD_COLS = 9     
SQUARE_SIZE_M = 0.0165    
COOLDOWN_SECONDS = 3.0    
BIAS_INCREMENT = 0       
DELTA_T = 20000
MAX_SINGLE_SNAP_RMS = 1.0  
MIN_REQUIRED_SNAPS = 15    

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

def configure_biases(iterator, logger):
    try:
        device = iterator.reader.device
        biases = device.get_i_ll_biases()
        if biases:
            current_on = biases.get("bias_diff_on")
            current_off = biases.get("bias_diff_off")
            biases.set("bias_diff_on", current_on + BIAS_INCREMENT)
            biases.set("bias_diff_off", current_off + BIAS_INCREMENT)
            logger.info(f"HW Biases set: ON={current_on+BIAS_INCREMENT}, OFF={current_off+BIAS_INCREMENT}")
    except Exception as e:
        logger.warning(f"Could not configure hardware biases: {e}")

def save_prophesee_json(filename, width, height, mtx, dist, logger):
    os.makedirs("config", exist_ok=True)
    full_path = os.path.join("config", filename)
    data = {
        "type": "pinhole", "width": width, "height": height,
        "K": mtx.flatten().tolist(),
        "D": dist.flatten().tolist()
    }
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Calibration saved to: {full_path}")

def save_points_json(side, objpoints, imgpoints, width, height, logger):
    os.makedirs("data_analysis", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join("data_analysis", f"points_{side}_{timestamp}.json")
    data = {
        "side": side, "width": width, "height": height,
        "objpoints": [op.tolist() for op in objpoints],
        "imgpoints": [ip.tolist() for ip in imgpoints]
    }
    with open(full_path, 'w') as f:
        json.dump(data, f)
    logger.info(f"Points saved for offline analysis: {full_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["left", "right"], required=True)
    # Mode is now mandatory as requested
    parser.add_argument("--mode", choices=["calibrate", "analyze"], required=True)
    args = parser.parse_args()

    logger = setup_logging("heatmap_auto", args.side)
    serial = SERIAL_LEFT if args.side == "left" else SERIAL_RIGHT
    
    mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T) 
    configure_biases(mv_it, logger)
    height, width = mv_it.get_size()
    
    # Chessboard params
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints, imgpoints = [], []
    point_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps, last_cap = 0, time.time()

    logger.info(f"Streaming from {args.side} camera. Press 'q' to stop.")

    for evs in mv_it:
        if evs.size == 0: continue
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs[evs['p']==1]['y'], evs[evs['p']==1]['x']] = 255
        im_vis = cv2.bitwise_not(cv2.dilate(im, np.ones((2,2), np.uint8), iterations=1))
        
        ret, corners = cv2.findChessboardCornersSB(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_sb)
        
        if ret and (time.time() - last_cap > COOLDOWN_SECONDS):
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_snaps += 1
            last_cap = time.time()
            for c in corners: cv2.circle(point_mask, (int(c[0][0]), int(c[0][1])), 2, 1.0, -1)
            logger.info(f"Snap {valid_snaps} captured.")

        cv2.imshow("Calibration", im_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

    if valid_snaps >= 20:
        save_points_json(args.side, objpoints, imgpoints, width, height, logger)
        if args.mode == "calibrate":
            res = cv2.calibrateCameraExtended(objpoints, imgpoints, (width, height), None, None)
            ret, mtx, dist, rvecs, tvecs, stdInt, stdExt, viewErr = res
            
            thresh = np.mean(viewErr) + np.std(viewErr)
            f_obj, f_img = [], []
            for i, err in enumerate(viewErr.flatten()):
                if err < thresh and err < MAX_SINGLE_SNAP_RMS:
                    f_obj.append(objpoints[i]); f_img.append(imgpoints[i])
            
            if len(f_obj) >= MIN_REQUIRED_SNAPS:
                ret_f, mtx_f, dist_f, _, _ = cv2.calibrateCamera(f_obj, f_img, (width, height), None, None)
                logger.info(f"Final Refined RMS: {ret_f:.4f}")
                save_prophesee_json(f"camera_{args.side}.json", width, height, mtx_f, dist_f, logger)
    else:
        logger.error("Insufficient snapshots for calibration.")

if __name__ == "__main__":
    main()
