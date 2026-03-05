import cv2
import numpy as np
import json
import os
import math
import csv
import sys
import glob
import argparse
from metavision_core.event_io import EventsIterator

# Add the project root to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import (
    TRACKER_DELTA_T, MAX_SYNC_DIFF_US, MAX_AGE, MIN_PARTICLE_AREA, RECORD_FOLDER
)

class KalmanTrack:
    """A linear Kalman Filter with Lifecycle Management (Tentative -> Confirmed)."""
    def __init__(self, track_id, initial_pos, dt_sec):
        self.id = track_id
        self.dt_sec = dt_sec
        
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt_sec, 0, 0],
            [0, 1, 0, 0, dt_sec, 0],
            [0, 0, 1, 0, 0, dt_sec],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        self.kf.statePre = np.zeros((6, 1), np.float32)
        self.kf.statePost = np.zeros((6, 1), np.float32)
        self.kf.statePost[:3, 0] = initial_pos
        
        # Free 3D Tracking Process Noise
        self.kf.processNoiseCov = np.array([
            [1e-3, 0, 0, 0, 0, 0], 
            [0, 1e-3, 0, 0, 0, 0], 
            [0, 0, 1e-3, 0, 0, 0], 
            [0, 0, 0, 1e-2, 0, 0], 
            [0, 0, 0, 0, 1e-2, 0], 
            [0, 0, 0, 0, 0, 1e-2]  
        ], dtype=np.float32)
        
        # Measurement Noise (Trust X/Y, tolerate Z quantization noise)
        self.kf.measurementNoiseCov = np.array([
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 5e-2] 
        ], dtype=np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
        
        # --- LIFECYCLE MANAGEMENT ---
        self.status = "TENTATIVE" # Starts as a candidate
        self.age = 0             
        self.hits = 1            
        self.pos = initial_pos   
        self.vel_norm = 0.0      
        self.is_initialized = False 

    def predict(self):
        pred = self.kf.predict()
        return pred[:3, 0]

    def update(self, measurement):
        meas_arr = np.array(measurement, dtype=np.float32).reshape(3, 1)
        
        # Bootstrap velocity on the first valid move
        if not self.is_initialized:
            vx = (measurement[0] - self.pos[0]) / self.dt_sec
            vy = (measurement[1] - self.pos[1]) / self.dt_sec
            vz = (measurement[2] - self.pos[2]) / self.dt_sec
            
            self.kf.statePre[3:, 0] = [vx, vy, vz]
            self.kf.statePost[3:, 0] = [vx, vy, vz]
            self.is_initialized = True

        self.kf.correct(meas_arr)
        
        self.pos = self.kf.statePost[:3, 0]
        vx, vy, vz = self.kf.statePost[3:, 0]
        self.vel_norm = math.sqrt(vx**2 + vy**2 + vz**2)
        
        self.age = 0
        self.hits += 1
        
        # Promote candidate to Confirmed after 3 consistent frames
        if self.status == "TENTATIVE" and self.hits >= 3:
            self.status = "CONFIRMED"


class ParticleTracker:
    def __init__(self, csv_file, logger):
        self.next_id = 0
        self.tracks = {} 
        self.csv_file = csv_file
        self.logger = logger
        
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_us', 'particle_id', 'x_m', 'y_m', 'z_m', 'vel_mps'])

    def update(self, current_detections, timestamp_us):
        dt_sec = TRACKER_DELTA_T / 1e6
        frame_velocities = []
        
        predictions = {}
        for tid, track in self.tracks.items():
            predictions[tid] = track.predict()
            
        unmatched_detections = list(current_detections)
        
        for tid, pred_pos in predictions.items():
            best_det_idx = None
            
            # --- STRICT PHYSICAL LIMITS ---
            # Max allowed deviation from the Kalman prediction is 5 cm.
            # This completely eliminates "teleportation" wrapping across the screen.
            max_search_radius = 0.05 
            
            for i, det_pos in enumerate(unmatched_detections):
                dist_3d = np.linalg.norm(pred_pos - det_pos)
                if dist_3d < max_search_radius:
                    max_search_radius = dist_3d
                    best_det_idx = i
                    
            if best_det_idx is not None:
                meas_pos = unmatched_detections.pop(best_det_idx)
                self.tracks[tid].update(meas_pos)
                
                # Only log data for CONFIRMED tracks
                if self.tracks[tid].status == "CONFIRMED":
                    frame_velocities.append(self.tracks[tid].vel_norm)
                    pos = self.tracks[tid].pos
                    with open(self.csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_us, tid, float(pos[0]), float(pos[1]), float(pos[2]), float(self.tracks[tid].vel_norm)])
            else:
                self.tracks[tid].age += 1
                
        # Spawn new candidates from unmatched detections
        for det_pos in unmatched_detections:
            self.tracks[self.next_id] = KalmanTrack(self.next_id, det_pos, dt_sec)
            self.next_id += 1
            
        # --- TRACK DEATH LOGIC ---
        alive_tracks = {}
        for tid, t in self.tracks.items():
            if t.status == "TENTATIVE" and t.age > 0:
                # Kill weak candidates instantly if they miss a single frame
                continue
            elif t.status == "CONFIRMED" and t.age > MAX_AGE:
                # Give confirmed tracks a grace period to recover from occlusions
                continue
            else:
                alive_tracks[tid] = t
                
        self.tracks = alive_tracks
        
        return self.tracks, np.mean(frame_velocities) if frame_velocities else 0

def load_stereo_config(logger):
    try:
        with open("config/camera_left.json", 'r') as f:
            camL = json.load(f)
            K_L, D_L = np.array(camL["K"]), np.array(camL["D"])
            w, h = camL.get("width", 320), camL.get("height", 320)
            
        with open("config/camera_right.json", 'r') as f:
            camR = json.load(f)
            K_R, D_R = np.array(camR["K"]), np.array(camR["D"])
            
        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
            R, T = np.array(stereo["R"]), np.array(stereo["T"])
    except FileNotFoundError as e:
        logger.error(f"Missing config file: {e}")
        sys.exit(1)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R, T, alpha=0)
    
    # Generate rectification maps for offline processing
    map1_L, map2_L = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
    map1_R, map2_R = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
    
    focal_length = float(P1[0, 0])
    baseline_mm = float(np.linalg.norm(T)) 
    
    logger.info(f"Calibration loaded. Focal: {focal_length:.2f}px, Baseline: {baseline_mm:.2f}mm")
    return K_L, D_L, R1, P1, K_R, D_R, R2, P2, focal_length, baseline_mm, w, h, map1_L, map2_L, map1_R, map2_R

def get_latest_raw_files(logger):
    left_files = glob.glob(os.path.join(RECORD_FOLDER, "*left*.raw"))
    right_files = glob.glob(os.path.join(RECORD_FOLDER, "*right*.raw"))
    
    if not left_files or not right_files:
        logger.error(f"Could not find matching raw files in {RECORD_FOLDER}")
        sys.exit(1)
        
    return max(left_files, key=os.path.getctime), max(right_files, key=os.path.getctime)

def main():
    logger = setup_logging("offline_tracker", "optical_flow")
    
    parser = argparse.ArgumentParser(description="Offline 3D Particle Tracker for Event Cameras")
    parser.add_argument('--left', type=str, help="Path to the left .raw file")
    parser.add_argument('--right', type=str, help="Path to the right .raw file")
    args = parser.parse_args()

    if args.left and args.right:
        file_L, file_R = args.left, args.right
    else:
        logger.info("Auto-detecting latest recordings...")
        file_L, file_R = get_latest_raw_files(logger)

    logger.info(f"Processing Left:  {file_L}")
    logger.info(f"Processing Right: {file_R}")

    K_L, D_L, R1, P1, K_R, D_R, R2, P2, focal, baseline_mm, w, h, map1_L, map2_L, map1_R, map2_R = load_stereo_config(logger)
    
    csv_name = file_L.replace(".raw", "_tracked.csv")
    tracker = ParticleTracker(csv_name, logger)
    
    mv_L = EventsIterator(input_path=file_L, delta_t=TRACKER_DELTA_T)
    mv_R = EventsIterator(input_path=file_R, delta_t=TRACKER_DELTA_T)

    iter_L = iter(mv_L)
    iter_R = iter(mv_R)

    logger.info("Starting synchronized processing loop. Press 'q' to abort.")
    
    # Kernel adjusted to ensure large fast particles are solid blobs, not empty rings
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    try:
        ev_L = next(iter_L)
        ev_R = next(iter_R)
    except StopIteration:
        logger.error("One or both raw files are empty.")
        return

    while True:
        ts_L = mv_L.get_current_time()
        ts_R = mv_R.get_current_time()
        
        # --- TIMESTAMP SYNCHRONIZATION ---
        diff = ts_L - ts_R
        if abs(diff) > MAX_SYNC_DIFF_US:
            if diff < 0:
                try: ev_L = next(iter_L)
                except StopIteration: break
            else:
                try: ev_R = next(iter_R)
                except StopIteration: break
            continue 
            
        im_L = np.zeros((h, w), dtype=np.uint8)
        im_R = np.zeros((h, w), dtype=np.uint8)
        
        if ev_L.size > 0: im_L[ev_L['y'], ev_L['x']] = 255
        if ev_R.size > 0: im_R[ev_R['y'], ev_R['x']] = 255

        # 1. Close gaps to form solid particles
        im_L_solid = cv2.morphologyEx(im_L, cv2.MORPH_CLOSE, kernel)
        im_R_solid = cv2.morphologyEx(im_R, cv2.MORPH_CLOSE, kernel)

        # 2. Rectify the images FIRST. This fixes both the visual offset and the depth instability!
        im_L_rect = cv2.remap(im_L_solid, map1_L, map2_L, cv2.INTER_NEAREST)
        im_R_rect = cv2.remap(im_R_solid, map1_R, map2_R, cv2.INTER_NEAREST)

        # 3. Find centroids directly on the rectified images
        n_L, _, stats_L, cent_L = cv2.connectedComponentsWithStats(im_L_rect)
        n_R, _, stats_R, cent_R = cv2.connectedComponentsWithStats(im_R_rect)

        pts_L_rect = [cent_L[i] for i in range(1, n_L) if stats_L[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA]
        pts_R_rect = [cent_R[i] for i in range(1, n_R) if stats_R[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA]

        # 4. Epipolar Triangulation
        points_3d_meters = []
        for pL in pts_L_rect:
            best_pR = None
            for pR in pts_R_rect:
                # Epipolar constraint on rectified images (Y must be almost identical)
                if abs(pL[1] - pR[1]) < 3.0 and pL[0] > pR[0]:
                    best_pR = pR
                    break 
                    
            if best_pR is not None:
                disp = pL[0] - best_pR[0]
                z_mm = (focal * baseline_mm) / disp
                # Filter out absurd triangulations
                if 100.0 < z_mm < 1000.0:
                    x_mm = (pL[0] - P1[0, 2]) * z_mm / focal
                    y_mm = (pL[1] - P1[1, 2]) * z_mm / focal
                    points_3d_meters.append((x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0))

        # 5. Update Kalman Filter
        tracks, avg_v = tracker.update(points_3d_meters, ts_L)
        
        # 6. Visualization
        # We draw on the RECTIFIED image, so the math perfectly matches the screen
        vis = cv2.cvtColor(im_L_rect, cv2.COLOR_GRAY2BGR) 
        for tid, t in tracks.items():
            if t.status == "CONFIRMED": 
                # Re-project 3D point (meters) directly back to pixels using the camera matrix
                cx = int((t.pos[0] / t.pos[2]) * focal + P1[0, 2])
                cy = int((t.pos[1] / t.pos[2]) * focal + P1[1, 2])
                
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), 2)
                cv2.putText(vis, f"ID:{tid}", (cx, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(vis, f"Avg Flow: {avg_v:.2f} m/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, f"Sync Offset: {diff} us", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Offline 3D Tracker (Kalman Rectified)", cv2.resize(vis, (w*2, h*2), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
        try:
            ev_L = next(iter_L)
            ev_R = next(iter_R)
        except StopIteration:
            break

    cv2.destroyAllWindows()
    logger.info(f"Processing complete. Data saved to {csv_name}")

if __name__ == "__main__":
    main()