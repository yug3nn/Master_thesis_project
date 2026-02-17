import cv2
import numpy as np
import json
import os
import math
import csv
from datetime import datetime
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
SERIAL_LEFT = "genx320 11-003c"
SERIAL_RIGHT = "genx320 10-003c"
DELTA_T = 50000            
MAX_TRACK_DISTANCE = 0.15  
MAX_AGE = 5                
BIAS_INCREMENT = 35        
MIN_PARTICLE_AREA = 20     
OUTPUT_CSV = f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

class ParticleTracker:
    def __init__(self, csv_file):
        self.next_id = 0
        self.tracks = {} 
        self.csv_file = csv_file
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_us', 'particle_id', 'x_m', 'y_m', 'z_m', 'vel_mps'])

    def update(self, current_detections, timestamp_us):
        new_tracks = {}
        frame_velocities = []
        dt_sec = DELTA_T / 1e6
        
        for det_pos in current_detections:
            best_id = None
            min_dist = MAX_TRACK_DISTANCE
            
            for tid, tdata in self.tracks.items():
                prev_pos = tdata["pos"]
                dist = math.sqrt(sum((float(p) - float(c))**2 for p, c in zip(prev_pos, det_pos)))
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id is not None:
                prev_pos = self.tracks[best_id]["pos"]
                vel = math.sqrt(sum((float(c) - float(p))**2 for p, c in zip(prev_pos, det_pos))) / dt_sec
                frame_velocities.append(vel)
                new_tracks[best_id] = {"pos": det_pos, "age": 0, "vel_norm": vel}
                
                with open(self.csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp_us, int(best_id), *[float(p) for p in det_pos], float(vel)])
                del self.tracks[best_id]
            else:
                new_tracks[self.next_id] = {"pos": det_pos, "age": 0, "vel_norm": 0}
                self.next_id += 1
        
        for tid, tdata in self.tracks.items():
            if tdata["age"] < MAX_AGE:
                tdata["age"] += 1
                new_tracks[tid] = tdata
        self.tracks = new_tracks
        avg_vel = np.mean(frame_velocities) if frame_velocities else 0
        return self.tracks, avg_vel

def configure_hardware(iterator, mode, name):
    try:
        device = iterator.reader.device
        i_sync = device.get_i_camera_synchronization()
        if mode == 'master': i_sync.set_mode_master()
        else: i_sync.set_mode_slave()
        
        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", biases.get("bias_diff_on") + BIAS_INCREMENT)
            biases.set("bias_diff_off", biases.get("bias_diff_off") + BIAS_INCREMENT)
        print(f"[HW] {name} Configured as {mode.upper()}")
    except Exception as e: print(f"[ERR] {name} HW config: {e}")

def load_stereo_config():
    path_S = os.path.join("config", "stereo_params.json")
    with open(path_S) as f: data = json.load(f)
    w, h = data["width"], data["height"]
    K_L, D_L = np.array(data["camera_left"]["K"], dtype=np.float64), np.array(data["camera_left"]["D"], dtype=np.float64)
    K_R, D_R = np.array(data["camera_right"]["K"], dtype=np.float64), np.array(data["camera_right"]["D"], dtype=np.float64)
    R, T = np.array(data["stereo"]["R"], dtype=np.float64), np.array(data["stereo"]["T"], dtype=np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R, T, alpha=0)
    m1l, m2l = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (w, h), cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (w, h), cv2.CV_32FC1)
    return m1l, m2l, m1r, m2r, float(P1[0, 0]), float(abs(T[0])), w, h

def main():
    m1l, m2l, m1r, m2r, focal, baseline, w, h = load_stereo_config()
    tracker = ParticleTracker(OUTPUT_CSV)
    mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=DELTA_T)
    mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=DELTA_T)
    configure_hardware(mv_L, 'slave', "LEFT")
    configure_hardware(mv_R, 'master', "RIGHT")

    for evs_L, evs_R in zip(mv_L, mv_R):
        ts = int(mv_L.get_current_time())
        im_L = np.zeros((h, w), dtype=np.uint8)
        im_R = np.zeros((h, w), dtype=np.uint8)
        
        if evs_L.size > 0: im_L[evs_L[evs_L['p']==1]['y'], evs_L[evs_L['p']==1]['x']] = 255
        if evs_R.size > 0: im_R[evs_R[evs_R['p']==1]['y'], evs_R[evs_R['p']==1]['x']] = 255
        
        # Filtro morfologico per unire la sagoma della mano
        kernel = np.ones((3,3), np.uint8)
        im_L = cv2.morphologyEx(im_L, cv2.MORPH_CLOSE, kernel)
        im_R = cv2.morphologyEx(im_R, cv2.MORPH_CLOSE, kernel)
        
        rect_L = cv2.remap(im_L, m1l, m2l, cv2.INTER_LINEAR)
        rect_R = cv2.remap(im_R, m1r, m2r, cv2.INTER_LINEAR)

        n_L, _, stats_L, cent_L = cv2.connectedComponentsWithStats(rect_L)
        n_R, _, stats_R, cent_R = cv2.connectedComponentsWithStats(rect_R)

        points_3d = []
        for i in range(1, n_L):
            if stats_L[i, cv2.CC_STAT_AREA] < MIN_PARTICLE_AREA: continue
            xL, yL = float(cent_L[i][0]), float(cent_L[i][1])
            for j in range(1, n_R):
                if stats_R[j, cv2.CC_STAT_AREA] < MIN_PARTICLE_AREA: continue
                xR, yR = float(cent_R[j][0]), float(cent_R[j][1])
                
                if abs(yL - yR) < 3.0 and xL > xR:
                    disp = xL - xR
                    z = (focal * baseline) / disp
                    if 0.10 < z < 3.0: # Clip distanze ragionevoli
                        points_3d.append(((xL-w/2)*z/focal, (yL-h/2)*z/focal, z))
                        break

        tracks, avg_vel = tracker.update(points_3d, ts)
        
        vis = cv2.cvtColor(rect_L, cv2.COLOR_GRAY2BGR)
        for tid, tdata in tracks.items():
            if tdata["age"] == 0:
                pos = tdata["pos"]
                ix = int((float(pos[0]) * focal / float(pos[2])) + w/2)
                iy = int((float(pos[1]) * focal / float(pos[2])) + h/2)
                cv2.circle(vis, (ix, iy), 6, (0, 255, 0), 2)
                cv2.putText(vis, f"ID:{tid} V:{tdata['vel_norm']:.1f}", (ix+5, iy-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(vis, f"Avg Flow: {avg_vel:.2f} m/s", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("3D Flow Tracker", cv2.resize(vis, (w*2, h*2)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
