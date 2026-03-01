import cv2
import numpy as np
import json
import os
import glob
import argparse
import sys
from queue import Empty
import time

# Add the project root to sys.path to load the custom logger
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.camera_streamer import EventReaderThread
from src.utils.settings import SERIAL_LEFT, SERIAL_RIGHT, STEREO_DELTA_T, BIAS_INCREMENT_STEREO, MAX_SYNC_DIFF_US

def load_stereo_calibration(logger):
    """Loads all intrinsic and extrinsic matrices and computes rectification maps."""
    logger.info("Loading calibration parameters from config/ directory...")
    try:
        with open("config/camera_left.json", 'r') as f:
            camL = json.load(f)
            K1, D1 = np.array(camL["K"]), np.array(camL["D"])
            
        with open("config/camera_right.json", 'r') as f:
            camR = json.load(f)
            K2, D2 = np.array(camR["K"]), np.array(camR["D"])
            
        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
            R, T = np.array(stereo["R"]), np.array(stereo["T"])
            
    except FileNotFoundError as e:
        logger.error(f"Missing configuration file: {e}")
        sys.exit(1)

    img_size = (320, 320)

    # Compute Stereo Rectification matrices
    logger.info("Computing optimal stereo rectification transforms...")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, alpha=0
    )

    # Create mapping functions for fast distortion removal and alignment
    mapL_x, mapL_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    mapR_x, mapR_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    return (K1, D1, K2, D2), (R1, R2, P1, P2), (mapL_x, mapL_y, mapR_x, mapR_y), img_size


def draw_epipolar_guides(img, num_lines=15):
    """Draws horizontal green lines to verify epipolar alignment."""
    h, w = img.shape[:2]
    step = h // num_lines
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)
    return img


def run_offline_mode(logger, intrinsics, rect_params, img_size):
    """
    OFFLINE MODE: Validates rectification using the static checkerboard points
    extracted during the calibration phase.
    """
    logger.info("Starting OFFLINE rectification viewer...")
    
    K1, D1, K2, D2 = intrinsics
    R1, R2, P1, P2 = rect_params
    
    files = glob.glob(os.path.join("data_analysis", 'points_stereo_*.json'))
    if not files:
        logger.error("No raw dataset found in data_analysis/.")
        return
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Using dataset: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    img_pts_L = [np.array(i).astype(np.float32).reshape(-1, 2) for i in data["imgpoints_L"]]
    img_pts_R = [np.array(i).astype(np.float32).reshape(-1, 2) for i in data["imgpoints_R"]]

    idx = 0
    while True:
        # Create blank canvases
        canvas_L = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        canvas_R = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        
        # Mathematically undistort and rectify the checkerboard corners
        rect_pts_L = cv2.undistortPoints(img_pts_L[idx], K1, D1, R=R1, P=P1).reshape(-1, 2)
        rect_pts_R = cv2.undistortPoints(img_pts_R[idx], K2, D2, R=R2, P=P2).reshape(-1, 2)
        
        # Draw the points on the canvases
        for ptL, ptR in zip(rect_pts_L, rect_pts_R):
            cv2.circle(canvas_L, tuple(ptL.astype(int)), 3, (255, 0, 0), -1) # Blue (Left)
            cv2.circle(canvas_R, tuple(ptR.astype(int)), 3, (0, 0, 255), -1) # Red (Right)
            
        # Concatenate images horizontally and draw reference lines
        stereo_view = cv2.hconcat([canvas_L, canvas_R])
        stereo_view = draw_epipolar_guides(stereo_view)
        
        # Add UI text
        cv2.putText(stereo_view, f"OFFLINE - Snap {idx:02d}/{len(img_pts_L)-1}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(stereo_view, "Press 'n' for next, 'q' to quit", (10, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Stereo Rectification (Offline)", stereo_view)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            logger.info("Exiting offline viewer.")
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(img_pts_L)
            
    cv2.destroyAllWindows()


def run_live_mode(logger, maps, img_size):
    """
    LIVE MODE (Master-Slave Architecture): 
    Reads from both cameras asynchronously to prevent frame drops,
    applies real-time remapping, and displays the stereo-aligned feed.
    """
    logger.info("Starting LIVE rectification viewer (Master-Slave setup)...")
    mapL_x, mapL_y, mapR_x, mapR_y = maps
    h, w = img_size

    # --- 1. INITIALIZE CENTRALIZED THREADS ---
    logger.info("Starting hardware streams via CameraStreamer...")
    t_L = EventReaderThread(SERIAL_LEFT, STEREO_DELTA_T, role="SLAVE_LEFT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)
    t_R = EventReaderThread(SERIAL_RIGHT, STEREO_DELTA_T, role="MASTER_RIGHT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)

    t_L.start()
    time.sleep(0.5) # Brief warmup
    t_R.start()

    logger.info("System Online. Press 'q' to quit.")

    # 2. Synchronized Processing Loop
    try:
        while t_L.running and t_R.running:
            try:
                # The MASTER dictates the loop timing. We wait for its packet first.
                evs_L = t_L.q.get(timeout=1.0)
                # The SLAVE is retrieved immediately after.
                evs_R = t_R.q.get(timeout=1.0)
            except Empty:
                # Timeout occurred, loop again or check for errors
                continue

            # --- ACTIVE ALIGNMENT ---
            while abs(ts_L - ts_R) > MAX_SYNC_DIFF_US:
                logger.info(f"Syncing... Offset: {ts_L - ts_R}us")
                if ts_L < ts_R:
                    logger.debug(f"Flushing Left... diff: {ts_L - ts_R}")
                    try:
                        data_L = t_L.q.get(timeout=0.01)
                        ts_L = data_L[1]
                    except Empty: break 
                else:
                    logger.debug(f"Flushing Right... diff: {ts_L - ts_R}")
                    try:
                        data_R = t_R.q.get(timeout=0.01)
                        ts_R = data_R[1]
                    except Empty: break

            if abs(ts_L - ts_R) <= MAX_SYNC_DIFF_US:
                # Render Left Frame
                frame_L = np.zeros((h, w), dtype=np.uint8)
                if evs_L.size > 0: 
                    frame_L[evs_L['y'], evs_L['x']] = 255
                frame_L = cv2.cvtColor(frame_L, cv2.COLOR_GRAY2BGR)
                
                # Render Right Frame
                frame_R = np.zeros((h, w), dtype=np.uint8)
                if evs_R.size > 0: 
                    frame_R[evs_R['y'], evs_R['x']] = 255
                frame_R = cv2.cvtColor(frame_R, cv2.COLOR_GRAY2BGR)

                # Apply Geometric Remapping (Undistortion + Epipolar Alignment)
                rect_L = cv2.remap(frame_L, mapL_x, mapL_y, cv2.INTER_LINEAR)
                rect_R = cv2.remap(frame_R, mapR_x, mapR_y, cv2.INTER_LINEAR)

                # Concatenate and add reference lines
                stereo_view = cv2.hconcat([rect_L, rect_R])
                stereo_view = draw_epipolar_guides(stereo_view, num_lines=20)
                
                cv2.putText(stereo_view, "LIVE RECTIFICATION (Master-Slave Synced)", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Live Stereo Rectification", stereo_view)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Shutdown signal received.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # Safely shut down the hardware threads
        logger.info("Shutting down sensor threads...")
        t_L.stop()
        t_R.stop()
        t_L.join()
        t_R.join()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo Rectification Viewer")
    parser.add_argument("--mode", choices=["offline", "live"], required=True, 
                        help="Choose 'offline' to view dataset points, or 'live' for real-time camera stream.")
    args = parser.parse_args()

    # Initialize the custom logger
    logger = setup_logging("visualize_rectification", "stereo")
    
    # Load parameters and matrices
    intrinsics, rect_params, maps, img_size = load_stereo_calibration(logger)

    # Route execution based on user argument
    if args.mode == "offline":
        run_offline_mode(logger, intrinsics, rect_params, img_size)
    elif args.mode == "live":
        run_live_mode(logger, maps, img_size)
