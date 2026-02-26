import cv2
import numpy as np
import json
import os
import glob
import random

def visualize_rectification():
    print("--- STEREO RECTIFICATION INTERACTIVE CHECK ---")
    print("Instructions: Press 'n' for a random snapshot, any other key to quit.")

    # 1. Load Camera Parameters
    def load_intr(side):
        with open(f"config/camera_{side}.json", 'r') as f:
            d = json.load(f)
        k = np.array(d.get("K", d.get("camera_matrix"))).reshape(3,3).astype(np.float32)
        dist = np.array(d.get("D", d.get("dist_coeffs"))).astype(np.float32)
        return k, dist, d["width"], d["height"]

    try:
        mtx_L, dist_L, w, h = load_intr("left")
        mtx_R, dist_R, _, _ = load_intr("right")

        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
        
        R, T = np.array(stereo["R"]), np.array(stereo["T"])
    except FileNotFoundError as e:
        print(f"[ERROR] Missing configuration file: {e}")
        return

    # 2. Compute Rectification Maps
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_L, dist_L, mtx_R, dist_R, (w, h), R, T, alpha=0
    )
    map_L1, map_L2 = cv2.initUndistortRectifyMap(mtx_L, dist_L, R1, P1, (w, h), cv2.CV_16SC2)
    map_R1, map_R2 = cv2.initUndistortRectifyMap(mtx_R, dist_R, R2, P2, (w, h), cv2.CV_16SC2)

    # 3. Load Points Data
    files = glob.glob("data_analysis/points_stereo_*.json")
    if not files:
        print("[ERROR] No snapshots found in data_analysis/")
        return
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    num_snaps = len(data["imgpoints_L"])
    print(f"[INFO] Loaded {num_snaps} snapshots from {latest_file}")

    while True:
        # 4. Select a random snapshot
        idx = random.randint(0, num_snaps - 1)
        print(f"[INFO] Displaying snapshot #{idx}...")

        # 5. Create synthetic frames
        img_L = np.zeros((h, w, 3), dtype=np.uint8)
        img_R = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw small points (radius 2)
        for p in data["imgpoints_L"][idx]:
            cv2.circle(img_L, tuple(map(int, p[0])), 2, (0, 0, 255), -1)
        for p in data["imgpoints_R"][idx]:
            cv2.circle(img_R, tuple(map(int, p[0])), 2, (0, 0, 255), -1)

        # 6. Apply Rectification
        rect_L = cv2.remap(img_L, map_L1, map_L2, cv2.INTER_LINEAR)
        rect_R = cv2.remap(img_R, map_R1, map_R2, cv2.INTER_LINEAR)

        # 7. Composition and horizontal lines (ON TOP)
        combined = np.hstack((rect_L, rect_R))
        for i in range(0, h, 15):
            cv2.line(combined, (0, i), (w*2, i), (0, 255, 0), 1, cv2.LINE_AA)

        # Add some text info on the image
        cv2.putText(combined, f"Snap: {idx}/{num_snaps-1} (Press 'n' for next)", 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Stereo Rectification Check", combined)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            continue
        else:
            break

    cv2.destroyAllWindows()
    print("[INFO] Visualization closed.")

if __name__ == "__main__":
    visualize_rectification()