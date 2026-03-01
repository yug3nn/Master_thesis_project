import cv2
import numpy as np
import json
import os
import sys
import threading
from queue import Queue, Empty
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from metavision_core.event_io import EventsIterator
from src.utils.settings import (
    SERIAL_LEFT, SERIAL_RIGHT, STEREO_DELTA_T,
    CHECKERBOARD_ROWS, CHECKERBOARD_COLS, SQUARE_SIZE_MM,
    MIN_EVENTS_THRESHOLD, MAX_SYNC_DIFF_US, BIAS_INCREMENT_STEREO, IMG_HEIGHT, IMG_WIDTH, DATA_FOLDER
)

def camera_worker(serial, side, queue_obj, stop_event, logger):
    """
    Thread dedicated to fetching events from a single camera as fast as possible.
    Includes Hardware Synchronization setup and Polarity filtering.
    """
    try:
        logger.info(f"[{side}] Initializing sensor on {serial}...")
        mv_it = EventsIterator(input_path=serial, delta_t=STEREO_DELTA_T)
        device = mv_it.reader.device if hasattr(mv_it.reader, 'device') else mv_it.reader.get_device()
        
        # Configure Hardware Synchronization
        sync = device.get_i_camera_synchronization()
        if side == "LEFT":
            sync.set_mode_slave()
            logger.info(f"[{side}] Hardware set to SLAVE mode")
        else:
            sync.set_mode_master()
            logger.info(f"[{side}] Hardware set to MASTER mode")

        # Bias configuration for sharper edges
        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", biases.get("bias_diff_on") + BIAS_INCREMENT_STEREO)
            biases.set("bias_diff_off", biases.get("bias_diff_off") + BIAS_INCREMENT_STEREO)
            logger.info(f"[{side}] Biases increased (+{BIAS_INCREMENT_STEREO})")

        for evs in mv_it:
            if stop_event.is_set():
                break
            
            # Filter Polarity 1 (positive changes) and check threshold
            evs_filt = evs[evs['p'] == 1]
            if evs_filt.size >= MIN_EVENTS_THRESHOLD:
                # Policy: Drop old data to keep latency at minimum
                if queue_obj.full():
                    try: 
                        queue_obj.get_nowait()
                    except Empty: 
                        pass
                # Store tuple: (events, timestamp)
                queue_obj.put((evs_filt, mv_it.get_current_time()))
                
    except Exception as e:
        logger.error(f"Worker {side} critical error: {e}")
        stop_event.set()

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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retL, cornersL = cv2.findChessboardCorners(frame_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), None)
    retR, cornersR = cv2.findChessboardCorners(frame_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), None)
    
    if not (retL and retR):
        # We do not log an error here to prevent terminal spam during Auto-Capture
        return None
        
    cv2.cornerSubPix(frame_L, cornersL, (5, 5), (-1, -1), criteria)
    cv2.cornerSubPix(frame_R, cornersR, (5, 5), (-1, -1), criteria)
    
    # 2. Triangulate to 3D Space
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    
    pL_u = cv2.undistortPoints(cornersL, mtxL, distL)
    pR_u = cv2.undistortPoints(cornersR, mtxR, distR)
    
    pts4D = cv2.triangulatePoints(P1, P2, pL_u, pR_u)
    pts3D = (pts4D[:3, :] / pts4D[3, :]).T
    
    # Auto-Detect array orientation (Bug fix for diagonal calculation)
    grid_test = pts3D.reshape(IMG_HEIGHT, IMG_WIDTH, 3)
    if np.linalg.norm(grid_test[0,0] - grid_test[1,0]) > 0.030: 
        grid3D = pts3D.reshape(IMG_WIDTH, IMG_HEIGHT, 3)
        rows, cols = IMG_WIDTH, IMG_HEIGHT
    else:
        grid3D = grid_test
        rows, cols = IMG_HEIGHT, IMG_WIDTH
    
    # 3. Calculate Depth (Z-axis)
    z_values = pts3D[:, 2] * 1000  # Convert to mm
    mean_z = float(np.mean(z_values))
    std_z = float(np.std(z_values))
    
    # 4. Calculate Square Sizes (XY-plane validation)
    measured_distances = []
    for r in range(rows):
        for c in range(cols):
            if c < cols - 1: # Horizontal
                measured_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r, c+1]) * 1000)
            if r < rows - 1: # Vertical
                measured_distances.append(np.linalg.norm(grid3D[r, c] - grid3D[r+1, c]) * 1000)
                
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
    plt.show()

def main():
    logger = setup_logging("validate_3d", "stereo")
    calib_params = load_calibration(logger)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    measurements = []
    capture_requested = False
    
    # Initialize Queues and Thread Event
    q_L = Queue(maxsize=3)
    q_R = Queue(maxsize=3)
    stop_event = threading.Event()
    
    t_L = threading.Thread(target=camera_worker, args=(SERIAL_LEFT, "LEFT", q_L, stop_event, logger))
    t_R = threading.Thread(target=camera_worker, args=(SERIAL_RIGHT, "RIGHT", q_R, stop_event, logger))

    t_L.start()
    time.sleep(0.5)
    t_R.start()
    
    logger.info("Ready. Press [SPACE] to start Auto-Capture. Press [Q] to save & quit.")

    try:
        while not stop_event.is_set():
            try:
                # Fetch packets with timestamps
                data_L = q_L.get(timeout=0.01)
                data_R = q_R.get(timeout=0.01)
            except Empty:
                continue

            ts_L = data_L[1]
            ts_R = data_R[1]

            # --- ACTIVE ALIGNMENT LOOP ---
            while abs(ts_L - ts_R) > MAX_SYNC_DIFF_US:
                if ts_L < ts_R:
                    try:
                        data_L = q_L.get(timeout=0.01)
                        ts_L = data_L[1]
                    except Empty: 
                        break 
                else:
                    try:
                        data_R = q_R.get(timeout=0.01)
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
                proc_L = cv2.bitwise_not(cv2.dilate(frame_L, np.ones((3,3))))
                proc_R = cv2.bitwise_not(cv2.dilate(frame_R, np.ones((3,3))))

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
        stop_event.set()
        t_L.join()
        t_R.join()
        cv2.destroyAllWindows()
        
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
