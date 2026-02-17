import cv2
import numpy as np
import json
import time
import os
import argparse
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6     
CHECKERBOARD_COLS = 9     
SQUARE_SIZE_M = 0.0165    
COOLDOWN_SECONDS = 3.0    
BIAS_INCREMENT = 0       # The value to add to current biases
DELTA_T = 10000

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

def configure_biases(iterator):
    """ Access the hardware device and increase bias_diff_on/off by a fixed offset """
    try:
        # Access the underlying HAL device
        device = iterator.reader.device
        biases = device.get_i_ll_biases()
        
        if biases:
            # Get current values
            current_on = biases.get("bias_diff_on")
            current_off = biases.get("bias_diff_off")
            
            # Set new values
            biases.set("bias_diff_on", current_on + BIAS_INCREMENT)
            biases.set("bias_diff_off", current_off + BIAS_INCREMENT)
            
            print(f"[HARDWARE] Biases increased by {BIAS_INCREMENT}:")
            print(f"           ON: {current_on} -> {current_on + BIAS_INCREMENT}")
            print(f"           OFF: {current_off} -> {current_off + BIAS_INCREMENT}")
        else:
            print("[WARNING] Biases interface not available.")
    except Exception as e:
        print(f"[WARNING] Could not configure hardware biases: {e}")

def save_prophesee_json(filename, width, height, mtx, dist):
    """ Save calibration data in Metavision SDK compatible format inside 'config' folder """
    os.makedirs("config", exist_ok=True)
    full_path = os.path.join("config", filename)
    
    data = {
        "type": "pinhole",
        "width": width,
        "height": height,
        "K": [mtx[0,0], mtx[0,1], mtx[0,2], 
              mtx[1,0], mtx[1,1], mtx[1,2], 
              mtx[2,0], mtx[2,1], mtx[2,2]],
        "D": dist.flatten().tolist()
    }
    
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[SUCCESS] Calibration saved to: {full_path}")

def generate_point_heatmap(accumulated_mask):
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def main():
    parser = argparse.ArgumentParser(description="Hybrid Calibration for GenX320 Event Cameras")
    parser.add_argument("--side", choices=["left", "right"], required=True, 
                        help="Specify camera position: 'left' or 'right'")
    args = parser.parse_args()
    
    serial = SERIAL_LEFT
    if args.side == "right":
        serial = SERIAL_RIGHT

    filename = f"camera_{args.side}.json"

    print(f"--- HYBRID CALIBRATION: {args.side.upper()} CAMERA ---")
    
    # Initialize Event Iterator
    mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T) 
    
    # --- HARDWARE CONFIGURATION ---
    # Apply bias increase before starting the loop
    configure_biases(mv_it)
    
    height, width = mv_it.get_size()
    
    # Rest of the calibration logic...
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    flags_std = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_M

    objpoints, imgpoints = [], []
    point_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps = 0
    last_capture_time = time.time()

    for evs in mv_it:
        if evs.size == 0: continue

        mask_on = evs['p'] == 1 
        evs_filtered = evs[mask_on] 
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs_filtered['y'], evs_filtered['x']] = 255
        
        kernel = np.ones((2,2), np.uint8)
        im_filled = cv2.dilate(im, kernel, iterations=1)
        im_vis = cv2.bitwise_not(im_filled) 
        display = cv2.cvtColor(im_vis, cv2.COLOR_GRAY2BGR)

        corners_found = None
        method_used = ""
        ret = False

        try:
            ret, corners = cv2.findChessboardCornersSB(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_sb)
            if ret:
                corners_found, method_used = corners, "SB"
        except: pass

        if not ret:
            ret, corners = cv2.findChessboardCorners(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_std)
            if ret:
                corners_found = cv2.cornerSubPix(im_vis, corners, (11, 11), (-1, -1), criteria_subpix)
                method_used = "STD"

        if ret and corners_found is not None:
            cv2.drawChessboardCorners(display, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners_found, ret)
            if time.time() - last_capture_time > COOLDOWN_SECONDS:
                objpoints.append(objp)
                imgpoints.append(corners_found)
                valid_snaps += 1
                last_capture_time = time.time()
                for corner in corners_found:
                    cv2.circle(point_mask, (int(corner[0][0]), int(corner[0][1])), 2, 1.0, -1)
                print(f"[AUTO] Snap {valid_snaps} ({args.side}) via {method_used}")
                cv2.rectangle(display, (0,0), (width, height), (255,255,255), -1)

        heatmap_display = generate_point_heatmap(point_mask)
        cv2.putText(display, f"Camera: {args.side} | Snaps: {valid_snaps}", (10, 30), 1, 1.5, (255, 255, 0), 2)
        cv2.imshow("Hybrid Calibration", display)
        cv2.imshow("Point Density Heatmap", heatmap_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if valid_snaps >= 20:
        print(f"\n[Processing] Computing calibration for {args.side} camera...")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print(f"RMS Error: {ret:.4f} pixels")
        save_prophesee_json(filename, width, height, mtx, dist)
    else:
        print(f"\n[Error] Insufficient data ({valid_snaps}/20).")

if __name__ == "__main__":
    main()
