import cv2
import numpy as np
import json
import time
import os
import argparse
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6     # Internal vertical corners
CHECKERBOARD_COLS = 9     # Internal horizontal corners
SQUARE_SIZE_M = 0.0165    # Checkerboard square size in meters
COOLDOWN_SECONDS = 0.5    # Minimum time between captures

def save_prophesee_json(filename, width, height, mtx, dist):
    """ Save calibration data in Metavision SDK compatible format inside 'config' folder """
    # Ensure the destination directory exists
    os.makedirs("config", exist_ok=True)
    
    # Define full output path
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
    """ Generate a jet-colored heatmap based on accumulated corner detections """
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Hybrid Calibration for GenX320 Event Cameras")
    parser.add_argument("--side", choices=["left", "right"], required=True, 
                        help="Specify camera position: 'left' or 'right'")
    args = parser.parse_args()
    
    # Filename based on the command line argument
    filename = f"camera_{args.side}.json"

    print(f"--- HYBRID CALIBRATION: {args.side.upper()} CAMERA ---")
    print("Instructions:")
    print("  - Move the checkerboard to cover the entire field of view.")
    print("  - Green text: SB Method (High Precision).")
    print("  - Yellow text: Standard Method (Fallback).")
    
    # Subpixel refinement criteria
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Flags for Sector Based (SB) - Highest accuracy
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    
    # Flags for Standard detection
    flags_std = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    # Prepare object points (3D coordinates in real world space)
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_M

    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    # Initialize Event Iterator (Live camera or RAW file)
    mv_it = EventsIterator(input_path="", delta_t=30000) 
    height, width = mv_it.get_size()
    
    point_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps = 0
    last_capture_time = time.time()

    for evs in mv_it:
        if evs.size == 0: continue

        # --- 1. IMAGE GENERATION ---
        # Accumulate events into a frame-like representation
        mask_on = evs['p'] == 1 
        evs_filtered = evs[mask_on] 
        
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs_filtered['y'], evs_filtered['x']] = 255
        
        # Morphological operations to improve corner detection
        kernel = np.ones((2,2), np.uint8)
        im_filled = cv2.dilate(im, kernel, iterations=1)
        im_vis = cv2.bitwise_not(im_filled) # Invert for black-on-white detection
        display = cv2.cvtColor(im_vis, cv2.COLOR_GRAY2BGR)

        # --- 2. HYBRID CORNER DETECTION ---
        corners_found = None
        method_used = ""
        ret = False

        # ATTEMPT 1: Sector Based (SB)
        try:
            ret, corners = cv2.findChessboardCornersSB(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_sb)
            if ret:
                corners_found = corners
                method_used = "SB"
                color_text = (0, 255, 0) # Green
        except: pass

        # ATTEMPT 2: Standard Method (Fallback)
        if not ret:
            ret, corners = cv2.findChessboardCorners(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_std)
            if ret:
                corners_found = cv2.cornerSubPix(im_vis, corners, (11, 11), (-1, -1), criteria_subpix)
                method_used = "STD"
                color_text = (0, 255, 255) # Yellow

        # --- 3. UI AND AUTO-CAPTURE ---
        status_text = "Searching..."
        
        if ret and corners_found is not None:
            cv2.drawChessboardCorners(display, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners_found, ret)
            status_text = f"FOUND: {method_used}"
            
            current_time = time.time()
            if current_time - last_capture_time > COOLDOWN_SECONDS:
                objpoints.append(objp)
                imgpoints.append(corners_found)
                valid_snaps += 1
                last_capture_time = current_time
                
                # Update density heatmap
                for corner in corners_found:
                    x, y = int(corner[0][0]), int(corner[0][1])
                    cv2.circle(point_mask, (x, y), 2, 1.0, -1)
                
                print(f"[AUTO] Snap {valid_snaps} ({args.side}) via {method_used}")
                cv2.rectangle(display, (0,0), (width, height), (255,255,255), -1) # Flash effect

        # Overlay UI info
        heatmap_display = generate_point_heatmap(point_mask)
        cv2.putText(display, f"Camera: {args.side} | Snaps: {valid_snaps}", (10, 30), 1, 1.5, (255, 255, 0), 2)
        cv2.imshow("Hybrid Calibration", display)
        cv2.imshow("Point Density Heatmap", heatmap_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # --- 4. COMPUTATION ---
    if valid_snaps >= 20:
        print(f"\n[Processing] Computing calibration for {args.side} camera...")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        
        print(f"RMS Error: {ret:.4f} pixels")
        if ret < 1.0: print(" [!!!] EXCELLENT Result")
        
        save_prophesee_json(filename, width, height, mtx, dist)
    else:
        print(f"\n[Error] Insufficient data. Captured only {valid_snaps} snaps (min 20 required).")

if __name__ == "__main__":
    main()