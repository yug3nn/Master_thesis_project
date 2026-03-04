import cv2
import os
import numpy as np
import json
import sys
import time
import queue

# Add the project root to sys.path to load custom modules
sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from src.utils.camera_streamer import EventReaderThread
from src.utils.settings import (
    SERIAL_LEFT, SERIAL_RIGHT, STEREO_DELTA_T, BIAS_INCREMENT_STEREO
)
                  
DISPLAY_SCALE = 3.0 # 3x Zoom for the final display

def load_json_params(logger):
    """ Loads intrinsic and extrinsic parameters from the calibration JSONs """
    try:
        # Load LEFT camera intrinsics
        with open("config/camera_left.json", 'r') as f:
            camL = json.load(f)
            K_L = np.array(camL.get("K", camL.get("camera_matrix")), dtype=np.float64)
            D_L = np.array(camL.get("D", camL.get("dist_coeffs")), dtype=np.float64)
            # Try to get width/height, fallback to GenX320 default resolution
            width = camL.get("width", 320)
            height = camL.get("height", 320)
            
        # Load RIGHT camera intrinsics
        with open("config/camera_right.json", 'r') as f:
            camR = json.load(f)
            K_R = np.array(camR.get("K", camR.get("camera_matrix")), dtype=np.float64)
            D_R = np.array(camR.get("D", camR.get("dist_coeffs")), dtype=np.float64)
            
        # Load STEREO extrinsics
        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
            R = np.array(stereo["R"], dtype=np.float64)
            T = np.array(stereo["T"], dtype=np.float64)
            
    except FileNotFoundError as e:
        logger.error(f"Missing configuration file: {e}. Run stereo calibration first.")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing key in JSON configuration: {e}")
        sys.exit(1)

    # Calculate stereo rectification transforms
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (width, height), R, T, alpha=0)
    
    # Generate undistortion and rectification maps
    m1l, m2l = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (width, height), cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (width, height), cv2.CV_32FC1)
    
    logger.info("Stereo parameters loaded and rectification maps generated.")
    return m1l, m2l, m1r, m2r, width, height

def get_frame(evs, width, height):
    """ Converts a structured array of events into a 2D 8-bit image """
    im = np.zeros((height, width), dtype=np.uint8)
    if evs is not None and evs.size > 0:
        im[evs['y'], evs['x']] = 255
    return im

def nothing(x): 
    pass # Dummy callback for OpenCV trackbars

def main():
    logger = setup_logging("stereo_depth_rt", "stereo")
    logger.info("Starting Real-Time Stereo Depth preview...")

    # 1. Load calibration
    map1_L, map2_L, map1_R, map2_R, w, h = load_json_params(logger)
    
    # 2. Initialize Hardware Streams via Custom Class
    logger.info("Initializing hardware streams via EventReaderThread...")
    t_L = EventReaderThread(SERIAL_LEFT, STEREO_DELTA_T, role="SLAVE_LEFT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)
                            
    t_R = EventReaderThread(SERIAL_RIGHT, STEREO_DELTA_T, role="MASTER_RIGHT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_STEREO, filter_polarity=1)

    # Start SLAVE first, then MASTER
    t_L.start()
    time.sleep(0.5) 
    t_R.start()
    
    # 3. Setup UI Windows and Controls
    cv2.namedWindow("Clean Depth", cv2.WINDOW_NORMAL)
    
    # Trackbars for real-time SGBM tuning
    cv2.createTrackbar('Num Disp (16x)', 'Clean Depth', 4, 8, nothing) 
    cv2.createTrackbar('Block Size', 'Clean Depth', 2, 5, nothing)     
    cv2.createTrackbar('Min Disp', 'Clean Depth', 0, 30, nothing)
    
    # Distance clip threshold: disparities lower than this (far away) are forced to BLACK
    cv2.createTrackbar('Clip Far (Threshold)', 'Clean Depth', 10, 64, nothing)

    # Base SGBM configuration
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 1 * 5**2,
        P2=32 * 1 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print("\n>>> CLEAN DEPTH MODE RUNNING <<<")
    print("Use the 'Clip Far' slider to remove the blue background noise.")
    print("Press 'Q' or 'ESC' on the video window to exit.\n")
    
    # Buffers to hold the last available frame from each camera
    im_L = np.zeros((h, w), dtype=np.uint8)
    im_R = np.zeros((h, w), dtype=np.uint8)

    try:
        while True:
            updated = False
            
            # Non-blocking fetch from SLAVE queue
            try:
                evs_L, _ = t_L.q.get_nowait()
                im_L = get_frame(evs_L, w, h)
                updated = True
            except queue.Empty:
                pass
                
            # Non-blocking fetch from MASTER queue
            try:
                evs_R, _ = t_R.q.get_nowait()
                im_R = get_frame(evs_R, w, h)
                updated = True
            except queue.Empty:
                pass

            # Only process and render if we got new data from at least one camera
            if updated:
                # Rectify images
                rect_L = cv2.remap(im_L, map1_L, map2_L, cv2.INTER_LINEAR)
                rect_R = cv2.remap(im_R, map1_R, map2_R, cv2.INTER_LINEAR)
                
                # Downscale for faster SGBM processing
                small_L = cv2.resize(rect_L, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                small_R = cv2.resize(rect_R, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                
                # Retrieve current trackbar values
                n_disp = (cv2.getTrackbarPos('Num Disp (16x)', 'Clean Depth') + 1) * 16
                blk = (cv2.getTrackbarPos('Block Size', 'Clean Depth') * 2) + 3
                min_d = cv2.getTrackbarPos('Min Disp', 'Clean Depth') // 2
                clip_thresh = cv2.getTrackbarPos('Clip Far (Threshold)', 'Clean Depth')
                
                # Update SGBM parameters
                stereo.setNumDisparities(n_disp)
                stereo.setBlockSize(blk)
                stereo.setMinDisparity(min_d)
                
                # Compute Disparity on the small images
                disp_small = stereo.compute(small_L, small_R).astype(np.float32) / 16.0
                
                # Upscale back to original resolution
                disp_large = cv2.resize(disp_small, (w, h), interpolation=cv2.INTER_NEAREST)
                disp_large *= 2.0 
                
                # --- 1. CLEANUP: EVENT MASK ---
                # If the rectified left image pixel is black (< 30), there is no event,
                # so the depth must be 0. This removes artifact streaks where no motion occurred.
                mask_no_events = rect_L < 30
                disp_large[mask_no_events] = 0

                # --- 2. CLEANUP: DISTANCE CLIP ---
                # If the disparity is low (object is far away), force it to 0.
                # This effectively removes the background "blue noise".
                mask_too_far = disp_large < clip_thresh
                disp_large[mask_too_far] = 0

                # --- VISUAL NORMALIZATION ---
                disp_vis = disp_large / 64.0 
                disp_vis = np.clip(disp_vis, 0, 1)
                disp_color = cv2.applyColorMap((disp_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Re-apply absolute black masks to override the Colormap zero-value (usually dark blue)
                disp_color[mask_no_events] = 0
                disp_color[mask_too_far] = 0

                # Combine the raw left view with the depth map side-by-side
                vis_L = cv2.cvtColor(rect_L, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((vis_L, disp_color))
                
                # Upscale for better viewing on screen
                big_view = cv2.resize(combined, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow("Clean Depth", big_view)
            
            # Keyboard interrupt check (10ms wait)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Exit key pressed.")
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt (Ctrl+C) detected.")
    finally:
        print("\n[CMD] Stop signal received. Shutting down...")
        t_L.stop()
        t_R.stop()
        
        t_L.join()
        t_R.join()
        
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()