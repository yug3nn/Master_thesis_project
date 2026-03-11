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
from scipy.optimize import linear_sum_assignment

# Add the project root to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import (
    TRACKER_DELTA_T, MAX_SYNC_DIFF_US, MAX_AGE, MIN_PARTICLE_AREA, RECORD_FOLDER
)

class KalmanTrack:
    """A linear Kalman Filter with Constant Acceleration (CA) for tracking fluid vortices."""
    def __init__(self, track_id, initial_pos, dt_sec):
        self.id = track_id
        self.dt_sec = dt_sec
        
        # 9 State variables (x,y,z, vx,vy,vz, ax,ay,az), 3 Measurements (x,y,z)
        self.kf = cv2.KalmanFilter(9, 3)
        
        dt2 = 0.5 * (dt_sec ** 2)
        
        # Transition Matrix (Kinematic model with acceleration)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt_sec, 0,      0,      dt2, 0,   0],
            [0, 1, 0, 0,      dt_sec, 0,      0,   dt2, 0],
            [0, 0, 1, 0,      0,      dt_sec, 0,   0,   dt2],
            [0, 0, 0, 1,      0,      0,      dt_sec, 0, 0],
            [0, 0, 0, 0,      1,      0,      0, dt_sec, 0],
            [0, 0, 0, 0,      0,      1,      0, 0, dt_sec],
            [0, 0, 0, 0,      0,      0,      1, 0, 0],
            [0, 0, 0, 0,      0,      0,      0, 1, 0],
            [0, 0, 0, 0,      0,      0,      0, 0, 1]
        ], np.float32)
        
        # Measurement Matrix (We only observe position)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ], np.float32)
        
        self.kf.statePre = np.zeros((9, 1), np.float32)
        self.kf.statePost = np.zeros((9, 1), np.float32)
        self.kf.statePost[:3, 0] = initial_pos
        
        # Process Noise: Trust the physics for smooth curves
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * 1e-4
        self.kf.processNoiseCov[6:, 6:] = np.eye(3, dtype=np.float32) * 1e-2 # Accel can change
        
        # Measurement Noise: Event cameras have noisy centroids
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 1.0
        
        self.status = "TENTATIVE" 
        self.age = 0             
        self.hits = 1            
        self.pos = initial_pos   
        self.is_initialized = False 

    def predict(self):
        pred = self.kf.predict()
        return pred[:3, 0]

    def update(self, measurement):
        meas_arr = np.array(measurement, dtype=np.float32).reshape(3, 1)
        
        if not self.is_initialized:
            vx = (measurement[0] - self.pos[0]) / self.dt_sec
            vy = (measurement[1] - self.pos[1]) / self.dt_sec
            vz = (measurement[2] - self.pos[2]) / self.dt_sec
            
            self.kf.statePre[3:6, 0] = [vx, vy, vz]
            self.kf.statePost[3:6, 0] = [vx, vy, vz]
            self.is_initialized = True

        self.kf.correct(meas_arr)
        self.pos = self.kf.statePost[:3, 0]
        self.age = 0
        self.hits += 1
        
        if self.status == "TENTATIVE" and self.hits >= 3:
            self.status = "CONFIRMED"


class ParticleTracker:
    """Manages a pool of KalmanTracks. No longer writes to CSV directly."""
    def __init__(self, logger, role=""):
        self.next_id = 0
        self.tracks = {} 
        self.logger = logger
        self.role = role

    def update(self, current_detections_3d):
        dt_sec = TRACKER_DELTA_T / 1e6
        predictions = {tid: track.predict() for tid, track in self.tracks.items()}
        unmatched_detections = list(current_detections_3d)
        
        for tid, pred_pos in predictions.items():
            best_det_idx = None
            max_search_radius_xy = 0.01 # 5 cm physical limit on X-Y plane
            
            for i, det_pos in enumerate(unmatched_detections):
                # --- Z-NOISE FIX ---
                # Calculate X-Y planar distance separately from Z
                dist_xy = math.sqrt((pred_pos[0] - det_pos[0])**2 + (pred_pos[1] - det_pos[1])**2)
                dist_z = abs(pred_pos[2] - det_pos[2])
                
                # Accept if X-Y is close, and Z isn't completely crazy (10 cm tolerance to absorb quantization)
                if dist_xy < max_search_radius_xy and dist_z < 0.01:
                    max_search_radius_xy = dist_xy
                    best_det_idx = i
                    
            if best_det_idx is not None:
                meas_pos = unmatched_detections.pop(best_det_idx)
                self.tracks[tid].update(meas_pos)
            else:
                self.tracks[tid].age += 1
                
        for det_pos in unmatched_detections:
            self.tracks[self.next_id] = KalmanTrack(self.next_id, det_pos, dt_sec)
            self.next_id += 1
            
        alive_tracks = {}
        for tid, t in self.tracks.items():
            if t.status == "TENTATIVE" and t.age > 0:
                continue
            elif t.status == "CONFIRMED" and t.age > MAX_AGE:
                continue
            else:
                alive_tracks[tid] = t
                
        self.tracks = alive_tracks
        return self.tracks


class DipoleBondManager:
    """Links independent positive and negative 3D tracks. Falls back to positive-only if no match is found."""
    def __init__(self, csv_file, logger):
        self.csv_file = csv_file
        self.logger = logger
        self.bonds = {} # Maps pos_tid -> neg_tid
        
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_us', 'particle_id', 'x_m', 'y_m', 'z_m', 'vel_mps'])

    def update_bonds(self, pos_tracks, neg_tracks, timestamp_us):
        active_pos = {tid: t for tid, t in pos_tracks.items() if t.status == "CONFIRMED"}
        active_neg = {tid: t for tid, t in neg_tracks.items() if t.status == "CONFIRMED"}
        
        # 1. Break dead bonds
        for p_id in list(self.bonds.keys()):
            n_id = self.bonds[p_id]
            if p_id not in active_pos or n_id not in active_neg:
                del self.bonds[p_id]
                
        # 2. Form new bonds using the Hungarian Algorithm (Global Optimization)
        MAX_DIPOLE_SEPARATION_XY = 0.015 
        unbonded_pos = [pid for pid in active_pos if pid not in self.bonds]
        unbonded_neg = [nid for nid in active_neg if nid not in self.bonds.values()]
        
        if unbonded_pos and unbonded_neg:
            MAX_PENALTY = 99999.0
            # Create a cost matrix where rows are POS tracks and cols are NEG tracks
            cost_matrix = np.full((len(unbonded_pos), len(unbonded_neg)), MAX_PENALTY)
            
            for i, p_id in enumerate(unbonded_pos):
                p_pos = active_pos[p_id].pos
                for j, n_id in enumerate(unbonded_neg):
                    n_pos = active_neg[n_id].pos
                    dist_xy = math.sqrt((p_pos[0] - n_pos[0])**2 + (p_pos[1] - n_pos[1])**2)
                    
                    if dist_xy < MAX_DIPOLE_SEPARATION_XY:
                        cost_matrix[i, j] = dist_xy # The cost is the physical distance
                        
            # Solve the linear sum assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Apply the optimal bonds, rejecting penalty assignments
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < MAX_PENALTY:
                    p_id = unbonded_pos[r]
                    n_id = unbonded_neg[c]
                    self.bonds[p_id] = n_id
                
        # 3. Calculate positions, velocities, and save to CSV
        avg_velocities = []
        valid_particles = [] 
        
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            for p_id, p_track in active_pos.items():
                if p_id in self.bonds:
                    # BONDED: Use midpoint and average velocity
                    n_id = self.bonds[p_id]
                    n_track = active_neg[n_id]
                    
                    final_pos = (p_track.pos + n_track.pos) / 2.0
                    avg_v_vec = (p_track.kf.statePost[3:, 0] + n_track.kf.statePost[3:, 0]) / 2.0
                    vel_norm = float(np.linalg.norm(avg_v_vec))
                    
                    valid_particles.append((p_id, p_track.pos, n_track.pos, final_pos))
                else:
                    # SOLO POSITIVE: Fallback to just the leading edge (No random matching!)
                    final_pos = p_track.pos
                    v_vec = p_track.kf.statePost[3:, 0]
                    vel_norm = float(np.linalg.norm(v_vec))
                    
                    valid_particles.append((p_id, p_track.pos, None, final_pos))
                    
                avg_velocities.append(vel_norm)
                writer.writerow([timestamp_us, p_id, float(final_pos[0]), float(final_pos[1]), float(final_pos[2]), vel_norm])
                
        return valid_particles, np.mean(avg_velocities) if avg_velocities else 0
    

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

    K_L, D_L, R1, P1, K_R, D_R, R2, P2, focal, baseline_mm, w, h, map1_L, map2_L, map1_R, map2_R = load_stereo_config(logger)
    
    csv_name = file_L.replace(".raw", "_tracked.csv")
    
    # INDEPENDENT TRACKERS AND BOND MANAGER
    pos_tracker = ParticleTracker(logger, role="POS")
    neg_tracker = ParticleTracker(logger, role="NEG")
    bond_manager = DipoleBondManager(csv_name, logger)
    
    mv_L = EventsIterator(input_path=file_L, delta_t=TRACKER_DELTA_T)
    mv_R = EventsIterator(input_path=file_R, delta_t=TRACKER_DELTA_T)

    iter_L = iter(mv_L)
    iter_R = iter(mv_R)

    # Ultra-light kernel just for local edge density
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

    try:
        ev_L = next(iter_L)
        ev_R = next(iter_R)
    except StopIteration:
        return

    while True:
        ts_L = mv_L.get_current_time()
        ts_R = mv_R.get_current_time()
        
        diff = ts_L - ts_R
        if abs(diff) > MAX_SYNC_DIFF_US:
            if diff < 0:
                try: ev_L = next(iter_L)
                except StopIteration: break
            else:
                try: ev_R = next(iter_R)
                except StopIteration: break
            continue 
            
        # --- POLARITY SEPARATION ---
        im_L_pos, im_L_neg = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        im_R_pos, im_R_neg = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        
        if ev_L.size > 0: 
            im_L_pos[ev_L[ev_L['p']==1]['y'], ev_L[ev_L['p']==1]['x']] = 255
            im_L_neg[ev_L[ev_L['p']==0]['y'], ev_L[ev_L['p']==0]['x']] = 255
            
        if ev_R.size > 0: 
            im_R_pos[ev_R[ev_R['p']==1]['y'], ev_R[ev_R['p']==1]['x']] = 255
            im_R_neg[ev_R[ev_R['p']==0]['y'], ev_R[ev_R['p']==0]['x']] = 255

        # 1. Light morphological close
        im_L_rect_pos = cv2.remap(cv2.morphologyEx(im_L_pos, cv2.MORPH_CLOSE, kernel), map1_L, map2_L, cv2.INTER_NEAREST)
        im_L_rect_neg = cv2.remap(cv2.morphologyEx(im_L_neg, cv2.MORPH_CLOSE, kernel), map1_L, map2_L, cv2.INTER_NEAREST)
        im_R_rect_pos = cv2.remap(cv2.morphologyEx(im_R_pos, cv2.MORPH_CLOSE, kernel), map1_R, map2_R, cv2.INTER_NEAREST)
        im_R_rect_neg = cv2.remap(cv2.morphologyEx(im_R_neg, cv2.MORPH_CLOSE, kernel), map1_R, map2_R, cv2.INTER_NEAREST)

        # 2. Extract 2D centroids
        def get_centroids(img_rect):
            n, _, stats, _ = cv2.connectedComponentsWithStats(img_rect)
            return [np.array([stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0, 
                              stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0], dtype=np.float32) 
                    for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA]

        pts_L_pos, pts_L_neg = get_centroids(im_L_rect_pos), get_centroids(im_L_rect_neg)
        pts_R_pos, pts_R_neg = get_centroids(im_R_rect_pos), get_centroids(im_R_rect_neg)

        # 3. Independent Triangulation using Hungarian Algorithm (Penalty Cost Fix)
        def triangulate(pts_L, pts_R):
            pts_3d = []
            if not pts_L or not pts_R:
                return pts_3d

            # Use a large finite penalty instead of np.inf to prevent 'infeasible' errors
            MAX_PENALTY = 99999.0
            cost_matrix = np.full((len(pts_L), len(pts_R)), MAX_PENALTY)

            for i, pL in enumerate(pts_L):
                for j, pR in enumerate(pts_R):
                    y_diff = abs(pL[1] - pR[1])
                    
                    # STRICT CONSTRAINT: Max 2.0 pixels of vertical disparity, Left X > Right X
                    if y_diff <= 2.0 and pL[0] > pR[0]:
                        cost_matrix[i, j] = y_diff # Minimize epipolar misalignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                # Only process assignments that didn't trigger the penalty
                if cost_matrix[r, c] < MAX_PENALTY:
                    pL = pts_L[r]
                    pR = pts_R[c]
                    disp = pL[0] - pR[0]
                    z_mm = (focal * baseline_mm) / disp
                    
                    # Keep only physically plausible depths
                    if 100.0 < z_mm < 1000.0:
                        pts_3d.append(( (pL[0]-P1[0,2])*z_mm/focal/1000.0, 
                                        (pL[1]-P1[1,2])*z_mm/focal/1000.0, 
                                        z_mm/1000.0 ))
            return pts_3d

        points_3d_pos = triangulate(pts_L_pos, pts_R_pos)
        points_3d_neg = triangulate(pts_L_neg, pts_R_neg)

        # 4. Independent Kalman Updates
        pos_tracks = pos_tracker.update(points_3d_pos)
        neg_tracks = neg_tracker.update(points_3d_neg)
        
        # 5. Form Physical Bonds and Log to CSV
        bonded_particles, avg_v = bond_manager.update_bonds(pos_tracks, neg_tracks, ts_L)
        
        # 6. Dumbbell Visualization
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[im_L_rect_neg == 255] = (128, 128, 128) # Gray for NEG
        vis[im_L_rect_pos == 255] = (255, 255, 255) # White for POS
        
        def proj(p_3d):
            return (int((p_3d[0]/p_3d[2])*focal + P1[0,2]), int((p_3d[1]/p_3d[2])*focal + P1[1,2]))
        
        for p_id, p_pos, p_neg, final_pos in bonded_particles:
            cx_p, cy_p = proj(p_pos)
            cx_f, cy_f = proj(final_pos)
            
            if p_neg is not None:
                # FULL DUMBBELL: Pos and Neg track bonded
                cx_n, cy_n = proj(p_neg)
                cv2.line(vis, (cx_p, cy_p), (cx_n, cy_n), (255, 255, 255), 1)
                cv2.circle(vis, (cx_n, cy_n), 3, (255, 0, 0), -1) # Blue OFF
            
            # Draw POS track
            cv2.circle(vis, (cx_p, cy_p), 3, (0, 0, 255), -1) # Red ON
            
            # Draw Final Center
            cv2.circle(vis, (cx_f, cy_f), 6, (0, 255, 0), 2)
            cv2.putText(vis, f"ID:{p_id}", (cx_f, cy_f - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        cv2.putText(vis, f"Avg Flow: {avg_v:.2f} m/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Offline 3D Tracker (Part-Based Kalman)", cv2.resize(vis, (w*2, h*2), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
        try:
            ev_L, ev_R = next(iter_L), next(iter_R)
        except StopIteration:
            break

    cv2.destroyAllWindows()
    logger.info(f"Processing complete. Data saved to {csv_name}")

if __name__ == "__main__":
    main()