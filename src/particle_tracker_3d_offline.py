import cv2
import numpy as np
import json
import os
import math
import csv
import sys
from metavision_core.event_io import EventsIterator

# --- SETTINGS ---
DELTA_T = 20000            
MAX_TRACK_DISTANCE = 0.15  
MAX_AGE = 5                
MIN_PARTICLE_AREA = 5     

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
                dist = math.sqrt(sum((float(p) - float(c))**2 for p, c in zip(tdata["pos"], det_pos)))
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
                    f.flush() # Forza la scrittura su disco per ogni riga (sicurezza per Raspberry Pi)

                del self.tracks[best_id]
            else:
                new_tracks[self.next_id] = {"pos": det_pos, "age": 0, "vel_norm": 0}
                self.next_id += 1
        
        for tid, tdata in self.tracks.items():
            if tdata["age"] < MAX_AGE:
                tdata["age"] += 1
                new_tracks[tid] = tdata
        self.tracks = new_tracks
        return self.tracks, np.mean(frame_velocities) if frame_velocities else 0

def load_stereo_config():
    with open("config/stereo_params.json") as f: data = json.load(f)
    w, h = data["width"], data["height"]
    K_L, D_L = np.array(data["camera_left"]["K"]), np.array(data["camera_left"]["D"])
    K_R, D_R = np.array(data["camera_right"]["K"]), np.array(data["camera_right"]["D"])
    R, T = np.array(data["stereo"]["R"]), np.array(data["stereo"]["T"])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (w, h), R, T, alpha=0)
    return K_L, D_L, R1, P1, K_R, D_R, R2, P2, float(P1[0,0]), float(abs(T.flatten()[0])), w, h

def main(file_L, file_R):
    K_L, D_L, R1, P1, K_R, D_R, R2, P2, focal, baseline, w, h = load_stereo_config()
    csv_name = file_L.replace(".raw", ".csv")
    tracker = ParticleTracker(csv_name)
    
    mv_L = EventsIterator(input_path=file_L, delta_t=DELTA_T)
    mv_R = EventsIterator(input_path=file_R, delta_t=DELTA_T)

    for ev_L, ev_R in zip(mv_L, mv_R):
        ts = int(mv_L.get_current_time())
        im_L = np.zeros((h, w), dtype=np.uint8)
        im_R = np.zeros((h, w), dtype=np.uint8)
        if ev_L.size > 0: im_L[ev_L[ev_L['p']==1]['y'], ev_L[ev_L['p']==1]['x']] = 255
        if ev_R.size > 0: im_R[ev_R[ev_R['p']==1]['y'], ev_R[ev_R['p']==1]['x']] = 255

        # Clustering su immagini grezze (più veloce)
        n_L, _, stats_L, cent_L = cv2.connectedComponentsWithStats(im_L)
        n_R, _, stats_R, cent_R = cv2.connectedComponentsWithStats(im_R)

        # Rettifica puntuale dei soli centroidi (Risparmio CPU enorme)
        pts_L_rect = []
        for i in range(1, n_L):
            if stats_L[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA:
                pt = np.array([cent_L[i]], dtype=np.float32).reshape(-1, 1, 2)
                pts_L_rect.append(cv2.undistortPoints(pt, K_L, D_L, R=R1, P=P1)[0][0])

        pts_R_rect = []
        for i in range(1, n_R):
            if stats_R[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA:
                pt = np.array([cent_R[i]], dtype=np.float32).reshape(-1, 1, 2)
                pts_R_rect.append(cv2.undistortPoints(pt, K_R, D_R, R=R2, P=P2)[0][0])

        # Matching Epipolare
        points_3d = []
        for pL in pts_L_rect:
            for pR in pts_R_rect:
                if abs(pL[1] - pR[1]) < 3.0 and pL[0] > pR[0]:
                    disp = pL[0] - pR[0]
                    z = (focal * baseline) / disp
                    if 0.1 < z < 3.0:
                        points_3d.append(((pL[0]-w/2)*z/focal, (pL[1]-h/2)*z/focal, z))
                        break

        tracks, avg_v = tracker.update(points_3d, ts)
        
        # Visualizzazione (Solo camera Sinistra per debug veloce)
        vis = cv2.cvtColor(im_L, cv2.COLOR_GRAY2BGR) # im_L è grezza, i cerchi saranno leggermente spostati rispetto ai punti originali
        for i in range(1, n_L):
            if stats_L[i, cv2.CC_STAT_AREA] >= MIN_PARTICLE_AREA:
                cp = cent_L[i]
                cv2.circle(vis, (int(cp[0]), int(cp[1])), 5, (0, 255, 0), 1)
        cv2.putText(vis, f"Avg Flow: {avg_v:.2f} m/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Offline Tracker", cv2.resize(vis, (w*2, h*2)))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == "__main__":
    if len(sys.argv) < 3: print("Uso: python3 particle_tracker_offline.py <left.raw> <right.raw>")
    else: main(sys.argv[1], sys.argv[2])
