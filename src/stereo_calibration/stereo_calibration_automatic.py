import cv2
import numpy as np
import json
import time
import sys
import os
import argparse # Added for command line arguments
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6       
CHECKERBOARD_COLS = 9       
SQUARE_SIZE_M = 0.0165      
COOLDOWN_SECONDS = 3.0      

DELTA_T_VAL = 30000         
MIN_EVENTS_THRESHOLD = 5000 
MAX_SYNC_DIFF_US = 5000  

# Serial numbers
SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

FLAGS_SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def load_prophesee_json(filename):
    """ Loads intrinsic parameters from the config folder """
    filepath = os.path.join("config", filename)
    if not os.path.exists(filepath):
        print(f"{RED}[ERROR] File {filepath} not found!{RESET}")
        sys.exit(1)

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Ensure keys match your single camera calibration script output
    k_list = data["K"]
    mtx = np.array([[k_list[0], k_list[1], k_list[2]],
                    [k_list[3], k_list[4], k_list[5]],
                    [k_list[6], k_list[7], k_list[8]]], dtype=np.float32)
    dist = np.array(data["D"], dtype=np.float32)
    return mtx, dist, data["width"], data["height"]

def save_stereo_params(filename, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F, width, height):
    """ Saves stereo parameters using your original JSON structure inside 'config' """
    os.makedirs("config", exist_ok=True)
    filepath = os.path.join("config", filename)
    
    data = {
        "width": width,
        "height": height,
        "camera_left": {
            "K": mtx_L.tolist(),
            "D": dist_L.tolist()
        },
        "camera_right": {
            "K": mtx_R.tolist(),
            "D": dist_R.tolist()
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist()
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n{GREEN}[SUCCESS] Stereo calibration saved to: {filepath}{RESET}")

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Stereo Calibration for GenX320")
    parser.add_argument("--headless", action="store_true", help="Run without showing video windows")
    args = parser.parse_args()

    print(f"{YELLOW}--- STEREO CALIBRATION PROCESS (SYNC ENABLED) ---{RESET}")
    if args.headless:
        print(f"{YELLOW}[INFO] Running in HEADLESS mode. No windows will be opened.{RESET}")

    # Step 1: Automatic loading from config/
    mtx_L, dist_L, width_L, height_L = load_prophesee_json("camera_left.json")
    mtx_R, dist_R, width_R, height_R = load_prophesee_json("camera_right.json")
    width, height = width_L, height_L

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints, imgpoints_L, imgpoints_R = [], [], []

    print(f"Connecting to cameras...")
    it_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=DELTA_T_VAL)
    it_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=DELTA_T_VAL)

    valid_snaps = 0
    last_capture_time = time.time()
    
    try:
        # Step 2: Synchronized loop
        for evs_L, evs_R in zip(it_L, it_R):
            ts_L = it_L.get_current_time()
            ts_R = it_R.get_current_time()
            
            if abs(ts_L - ts_R) > MAX_SYNC_DIFF_US:
                continue

            if evs_L.size < MIN_EVENTS_THRESHOLD or evs_R.size < MIN_EVENTS_THRESHOLD:
                continue

            # Frame generation (needed for corner detection even in headless)
            im_L = np.zeros((height, width), dtype=np.uint8)
            im_R = np.zeros((height, width), dtype=np.uint8)
            im_L[evs_L['y'], evs_L['x']] = 255
            im_R[evs_R['y'], evs_R['x']] = 255
            
            im_L_proc = cv2.bitwise_not(cv2.dilate(im_L, np.ones((2,2))))
            im_R_proc = cv2.bitwise_not(cv2.dilate(im_R, np.ones((2,2))))

            ret_L, corners_L = cv2.findChessboardCornersSB(im_L_proc, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)
            ret_R, corners_R = cv2.findChessboardCornersSB(im_R_proc, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)

            if ret_L and ret_R:
                if time.time() - last_capture_time > COOLDOWN_SECONDS:
                    objpoints.append(objp)
                    imgpoints_L.append(corners_L)
                    imgpoints_R.append(corners_R)
                    valid_snaps += 1
                    last_capture_time = time.time()
                    print(f"{GREEN}[SNAP {valid_snaps}] Captured! Sync Diff: {abs(ts_L - ts_R)} us{RESET}")

            # Visualization only if NOT headless
            if not args.headless:
                combined = np.hstack((im_L, im_R))
                cv2.imshow("Stereo Calibration - Live View", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass

    if not args.headless:
        cv2.destroyAllWindows()

    if valid_snaps >= 15:
        print(f"\n{YELLOW}[PROCESSING] Computing Stereo Geometry...{RESET}")
        ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_L, imgpoints_R,
            mtx_L, dist_L, mtx_R, dist_R,
            (width, height), flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"{GREEN}Done! RMS: {ret:.4f}{RESET}")
        save_stereo_params("stereo_calibration.json", M1, D1, M2, D2, R, T, E, F, width, height)
    else:
        print(f"{RED}[ERROR] Not enough snaps ({valid_snaps}). Minimum 15 required.{RESET}")

if __name__ == "__main__":
    main()