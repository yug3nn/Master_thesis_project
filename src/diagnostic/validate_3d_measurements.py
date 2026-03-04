import cv2
import numpy as np
import json
import os
import sys
from queue import Empty
import time
import datetime
import argparse
import glob
import matplotlib
# If you want the plot to open on screen (as well as being saved), comment the line below:
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.camera_streamer import EventReaderThread
from src.utils.settings import (
    SERIAL_LEFT, SERIAL_RIGHT, STEREO_DELTA_T,
    CHECKERBOARD_ROWS, CHECKERBOARD_COLS, SQUARE_SIZE_MM,
    MIN_EVENTS_THRESHOLD, MAX_SYNC_DIFF_US, BIAS_INCREMENT_STEREO, IMG_HEIGHT, IMG_WIDTH, DATA_FOLDER, FLAGS_STEREO
)

def load_calibration(logger):
    """Loads intrinsic and extrinsic parameters for 3D triangulation."""
    try:
        with open("config/camera_left.json", 'r') as f:
            camL = json.load(f)
            mtxL, distL = np.array(camL["K"]), np.array(camL["D"])
        with open("config/camera_right.json", 'r') as f:
            camR = json.load(f)
            mtxR, distR = np.array(camR["K"]), np.array(camR["D"])
        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
            R, T = np.array(stereo["R"]), np.array(stereo["T"])
    except FileNotFoundError as e:
        logger.error(f"Missing config file: {e}")
        sys.exit(1)
        
    return mtxL, distL, mtxR, distR, R, T

def process_3d_measurement(frame_L, frame_R, calib_params, logger):
    """Detects checkerboard in both synchronized frames and computes 3D metrics."""
    mtxL, distL, mtxR, distR, R, T = calib_params
    
    # 1. Find Checkerboard Corners
    retL, cornersL = cv2.findChessboardCornersSB(frame_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_STEREO)
    retR, cornersR = cv2.findChessboardCornersSB(frame_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_STEREO)
    
    if not (retL and retR):
        # We do not log an error here to prevent terminal spam during Auto-Capture
        return None
    
    # 2. Triangulate to 3D Space
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    
    pL_u = cv2.undistortPoints(cornersL, mtxL, distL)
    pR_u = cv2.undistortPoints(cornersR, mtxR, distR)
    
    # --- FIX OPENCV BUG: TRANSPOSE ARRAYS ---
    pL_u_T = pL_u.reshape(-1, 2).T
    pR_u_T = pR_u.reshape(-1, 2).T
    
    pts4D = cv2.triangulatePoints(P1, P2, pL_u_T, pR_u_T)
    # 3D points are implicitly in MILLIMETERS based on baseline T
    pts3D = (pts4D[:3, :] / pts4D[3, :]).T
    
    # Auto-Detect array orientation (Bug fix for diagonal calculation)
    grid_test = pts3D.reshape(CHECKERBOARD_ROWS, CHECKERBOARD_COLS, 3)
    if np.linalg.norm(grid_test[0,0] - grid_test[1,0]) > (SQUARE_SIZE_MM * 1.5): 
        grid3D = pts3D.reshape(CHECKERBOARD_COLS, CHECKERBOARD_ROWS, 3)
        rows, cols = CHECKERBOARD_COLS, CHECKERBOARD_ROWS
    else:
        grid3D = grid_test
        rows, cols = CHECKERBOARD_ROWS, CHECKERBOARD_COLS
    
    # 3. Calculate Depth (Z-axis)
    z_values = pts3D[:, 2] 
    mean_z = float(np.mean(z_values))
    std_z = float(np.std(z_values))
    
    # 4. Calculate Square Sizes (XY-plane validation)
    measured_distances = []
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1: # Horizontal
                measured_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r, c+1]))
            if r < rows - 1: # Vertical
                measured_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r+1, c]))
                
    mean_sq_size = float(np.mean(measured_distances))
    sq_error = float(abs(mean_sq_size - SQUARE_SIZE_MM))
    
    logger.info(f"Measurement Success -> Estimated Z: {mean_z:.2f}mm | Sq Size: {mean_sq_size:.2f}mm")
    
    return {
        "estimated_z_mm": mean_z,
        "std_z_mm": std_z,
        "measured_square_size_mm": mean_sq_size,
        "square_size_error_mm": sq_error
    }

def generate_validation_plots(measurements, logger):
    """Generates and displays plots for thesis documentation."""
    valid_m = [m for m in measurements if m.get("ground_truth_z_mm") is not None]
    
    if not valid_m:
        logger.warning("No ground truth distances provided. Skipping plots.")
        return

    # Sort by Ground Truth Z
    valid_m.sort(key=lambda x: x["ground_truth_z_mm"])
    
    gt_z = [m["ground_truth_z_mm"] for m in valid_m]
    est_z = [m["estimated_z_mm"] for m in valid_m]
    err_z = [m["std_z_mm"] for m in valid_m]
    sq_size = [m["measured_square_size_mm"] for m in valid_m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PLOT 1: Depth Estimation Accuracy (Z vs Z)
    ax1.errorbar(gt_z, est_z, yerr=err_z, fmt='o-', color='tab:blue', linewidth=2, capsize=5, label='Estimated Depth ± Std')
    min_val, max_val = min(gt_z), max(gt_z)
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Ideal Ground Truth (1:1)')
    
    ax1.set_title("Stereo Depth Estimation Accuracy", fontsize=12)
    ax1.set_xlabel("Ground Truth Distance Z (mm)")
    ax1.set_ylabel("Stereo Estimated Z (mm)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # PLOT 2: Metric Scale Stability over Depth
    ax2.plot(gt_z, sq_size, 's-', color='tab:orange', linewidth=2, label='Measured Checkerboard Sq')
    ax2.axhline(SQUARE_SIZE_MM, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True Size ({SQUARE_SIZE_MM}mm)')
    ax2.set_ylim(SQUARE_SIZE_MM - 1.5, SQUARE_SIZE_MM + 1.5)
    
    ax2.set_title("Metric Scale Stability vs Depth", fontsize=12)
    ax2.set_xlabel("Ground Truth Distance Z (mm)")
    ax2.set_ylabel("Measured Square Size (mm)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(DATA_FOLDER, "depth_validation_plots.png")
    plt.savefig(plot_path)
    logger.info(f"Validation plots saved to: {plot_path}")
    
    # Note: with matplotlib.use('Agg') active, plt.show() will not open any GUI window
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Stereo 3D Measurement Validation")
    parser.add_argument('--replot', nargs='?', const='latest', type=str, 
                        help="Regenerate the plot using an existing JSON. Use '--replot' for the latest, or '--replot filename.json' for a specific file.")
    args = parser.parse_args()

    logger = setup_logging("validate_3d", "stereo")
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # --- REPLOT MODE (OFFLINE) ---
    if args.replot:
        if args.replot == 'latest':
            list_of_files = glob.glob(os.path.join(DATA_FOLDER, 'depth_validation_*.json'))
            if not list_of_files:
                logger.error("No 'depth_validation' JSON files found in the data folder.")
                sys.exit(1)
            json_path = max(list_of_files, key=os.path.getctime)
        else:
            json_path = args.replot
            
        logger.info(f"Re-plotting mode activated. Loading data from: {json_path}")
        try:
            with open(json_path, 'r') as f:
                measurements = json.load(f)
            generate_validation_plots(measurements, logger)
        except Exception as e:
            logger.error(f"Error while loading file or generating plot: {e}")
            
        sys.exit(0) # Exits without initializing the hardware cameras
    
    
    # --- NORMAL MODE (ACQUISITION) ---
    calib_params = load_calibration(logger)
    measurements = []
    capture_requested = False
    
    logger.info("Starting hardware streams via CameraStreamer...")
    t_L = EventReaderThread(SERIAL_LEFT, STEREO_DELTA_T, role="SLAVE_LEFT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)
    t_R = EventReaderThread(SERIAL_RIGHT, STEREO_DELTA_T, role="MASTER_RIGHT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)

    t_L.start()
    time.sleep(0.5) # Brief warmup
    t_R.start()
    
    logger.info("Ready. Press [SPACE] to start Auto-Capture. Press [Q] to save & quit.")

    try:
        while t_L.running and t_R.running:
            try:
                # Fetch packets with timestamps
                data_L = t_L.q.get(timeout=0.01)
                data_R = t_R.q.get(timeout=0.01)
            except Empty:
                continue

            ts_L = data_L[1]
            ts_R = data_R[1]

            # --- ACTIVE ALIGNMENT LOOP ---
            while abs(ts_L - ts_R) > MAX_SYNC_DIFF_US:
                if ts_L < ts_R:
                    try:
                        data_L = t_L.q.get(timeout=0.01)
                        ts_L = data_L[1]
                    except Empty: 
                        break 
                else:
                    try:
                        data_R = t_R.q.get(timeout=0.01)
                        ts_R = data_R[1]
                    except Empty: 
                        break

            # If synchronized within the threshold (200 us)
            if abs(ts_L - ts_R) <= MAX_SYNC_DIFF_US:
                frame_L = np.zeros((320, 320), dtype=np.uint8)
                if data_L[0].size > 0: 
                    frame_L[data_L[0]['y'], data_L[0]['x']] = 255
                
                frame_R = np.zeros((320, 320), dtype=np.uint8)
                if data_R[0].size > 0: 
                    frame_R[data_R[0]['y'], data_R[0]['x']] = 255

                # Optional: dilation to help corner detection on event images
                proc_L = cv2.bitwise_not(cv2.morphologyEx(frame_L, cv2.MORPH_CLOSE, np.ones((3,3))))
                proc_R = cv2.bitwise_not(cv2.morphologyEx(frame_R, cv2.MORPH_CLOSE, np.ones((3,3))))

                view = cv2.hconcat([frame_L, frame_R])
                sync_diff = abs(ts_L - ts_R)
                
                # --- UI OVERLAYS ---
                if capture_requested:
                    cv2.putText(view, "SEARCHING FOR CHECKERBOARD... (Press 'c' to cancel)", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(view, "Press SPACE to Auto-Capture, Q to Quit", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Live Z-Probe", view)

                # --- KEYBOARD HANDLING ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exiting loop. Preparing data...")
                    break
                elif key == ord('c') and capture_requested:
                    capture_requested = False
                    logger.info("Auto-Capture canceled by user.")
                elif key == 32 and not capture_requested: # SPACEBAR
                    capture_requested = True
                    logger.info("Auto-Capture started. Please show the checkerboard to both cameras...")
                    
                # --- AUTO-CAPTURE LOGIC ---
                if capture_requested:
                    metrics = process_3d_measurement(proc_L, proc_R, calib_params, logger)
                    
                    if metrics:
                        capture_requested = False # Stop searching once we succeed
                        print("\n" + "-"*50)
                        user_input = input(">>> SUCCESS! Enter Ground Truth Z distance in mm (or press Enter to skip): ")
                        print("-" * 50 + "\n")
                        
                        gt_z = None
                        if user_input.strip():
                            try: 
                                gt_z = float(user_input)
                            except ValueError: 
                                logger.error("Invalid number. Skipping GT association.")
                        
                        metrics["ground_truth_z_mm"] = gt_z
                        measurements.append(metrics)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        # Safely shut down the hardware threads
        logger.info("Shutting down sensor threads...")
        t_L.stop()
        t_R.stop()
        t_L.join()
        t_R.join()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")
        
        # Save JSON and Generate Plots
        if measurements:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"depth_validation_{timestamp}.json"
            json_path = os.path.join(DATA_FOLDER, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(measurements, f, indent=4)
            logger.info(f"Saved {len(measurements)} measurements to {json_path}")
            
            generate_validation_plots(measurements, logger)
        else:
            logger.info("No measurements taken. Exiting without saving.")

if __name__ == "__main__":
    main()