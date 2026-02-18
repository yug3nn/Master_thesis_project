import cv2
import numpy as np
import json
import time
import sys
import os
import argparse
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6       
CHECKERBOARD_COLS = 9       
SQUARE_SIZE_M = 0.0165      
COOLDOWN_SECONDS = 2.0      

DELTA_T_VAL = 30000         
MIN_EVENTS_THRESHOLD = 10 
MAX_SYNC_DIFF_US = 200  
BIAS_INCREMENT = 30     

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

FLAGS_SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def configure_biases(iterator, camera_name):
    """ Hardware access to increase bias_diff_on/off by the fixed increment """
    try:
        device = iterator.reader.device
        biases = device.get_i_ll_biases()
        if biases:
            current_on = biases.get("bias_diff_on")
            current_off = biases.get("bias_diff_off")
            biases.set("bias_diff_on", current_on + BIAS_INCREMENT)
            biases.set("bias_diff_off", current_off + BIAS_INCREMENT)
            print(f"[HARDWARE] {camera_name} biases increased (+{BIAS_INCREMENT})")
    except Exception as e:
        print(f"[WARNING] Bias config failed for {camera_name}: {e}")

def generate_point_heatmap(accumulated_mask):
    """ Generate a jet-colored heatmap based on accumulated corner detections """
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def load_prophesee_json(filename):
    filepath = os.path.join("config", filename)
    if not os.path.exists(filepath):
        print(f"{RED}[ERROR] File {filepath} not found!{RESET}")
        sys.exit(1)
    with open(filepath, 'r') as f:
        data = json.load(f)
    k_list = data["K"]
    mtx = np.array([[k_list[0], k_list[1], k_list[2]],
                    [k_list[3], k_list[4], k_list[5]],
                    [k_list[6], k_list[7], k_list[8]]], dtype=np.float32)
    dist = np.array(data["D"], dtype=np.float32)
    return mtx, dist, data["width"], data["height"]

def save_stereo_params(filename, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F, width, height):
    os.makedirs("config", exist_ok=True)
    filepath = os.path.join("config", filename)
    data = {
        "width": width, "height": height,
        "camera_left": {"K": mtx_L.tolist(), "D": dist_L.tolist()},
        "camera_right": {"K": mtx_R.tolist(), "D": dist_R.tolist()},
        "stereo": {"R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist()}
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n{GREEN}[SUCCESS] Stereo calibration saved to: {filepath}{RESET}")

def main():
    parser = argparse.ArgumentParser(description="Stereo Calibration for GenX320")
    parser.add_argument("--headless", action="store_true", help="Run without showing video windows")
    args = parser.parse_args()

    print(f"{YELLOW}--- STEREO CALIBRATION PROCESS (POLARITY 1 FILTER) ---{RESET}")

    mtx_L, dist_L, width_L, height_L = load_prophesee_json("camera_left.json")
    mtx_R, dist_R, width_R, height_R = load_prophesee_json("camera_right.json")
    width, height = width_L, height_L

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    objpoints, imgpoints_L, imgpoints_R = [], [], []

    it_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=DELTA_T_VAL)
    it_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=DELTA_T_VAL)

    # --- AGGIUNTA SINCRONIZZAZIONE MASTER/SLAVE ---
    try:
        dev_L = it_L.reader.device if hasattr(it_L.reader, 'device') else it_L.reader.get_device()
        dev_R = it_R.reader.device if hasattr(it_R.reader, 'device') else it_R.reader.get_device()
        
        # Impostiamo la Sinistra come SLAVE e la Destra come MASTER
        dev_L.get_i_camera_synchronization().set_mode_slave()
        dev_R.get_i_camera_synchronization().set_mode_master()
        
        print(f"{GREEN}[HW] Sincronizzazione attivata: LEFT (Slave) <--- RIGHT (Master){RESET}")
    except Exception as e:
        print(f"{RED}[WARNING] Errore configurazione Sync: {e}{RESET}")
    # ----------------------------------------------

    configure_biases(it_L, "LEFT")
    configure_biases(it_R, "RIGHT")

    valid_snaps = 0
    last_capture_time = time.time()
    point_mask = np.zeros((height, width), dtype=np.float32)

    try:
        for evs_L, evs_R in zip(it_L, it_R):
            ts_L = it_L.get_current_time()
            ts_R = it_R.get_current_time()
            
            if abs(ts_L - ts_R) > MAX_SYNC_DIFF_US: 
                print(f"{abs(ts_L - ts_R)}")
                continue
            
            # --- FILTER: ONLY POLARITY 1 (ON EVENTS) ---
            evs_L = evs_L[evs_L['p'] == 1]
            evs_R = evs_R[evs_R['p'] == 1]

            if evs_L.size < MIN_EVENTS_THRESHOLD or evs_R.size < MIN_EVENTS_THRESHOLD: continue

            im_L = np.zeros((height, width), dtype=np.uint8)
            im_R = np.zeros((height, width), dtype=np.uint8)
            im_L[evs_L['y'], evs_L['x']] = 255
            im_R[evs_R['y'], evs_R['x']] = 255
            
            im_L_proc = cv2.bitwise_not(cv2.dilate(im_L, np.ones((3,3))))
            im_R_proc = cv2.bitwise_not(cv2.dilate(im_R, np.ones((3,3))))

            ret_L, corners_L = cv2.findChessboardCornersSB(im_L_proc, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)
            ret_R, corners_R = cv2.findChessboardCornersSB(im_R_proc, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), FLAGS_SB)

            if ret_L and ret_R:
                if time.time() - last_capture_time > COOLDOWN_SECONDS:
                    objpoints.append(objp)
                    imgpoints_L.append(corners_L)
                    imgpoints_R.append(corners_R)
                    valid_snaps += 1
                    last_capture_time = time.time()
                    
                    for corner in corners_L:
                        cv2.circle(point_mask, (int(corner[0][0]), int(corner[0][1])), 2, 1.0, -1)
                    
                    print(f"{GREEN}[SNAP {valid_snaps}] Sync OK (Pol 1 filtered){RESET}")


            if not args.headless:
                combined = np.hstack((im_L, im_R))
                heatmap_display = generate_point_heatmap(point_mask)
                cv2.imshow("Stereo Calibration", combined)
                cv2.imshow("Stereo Coverage Heatmap", heatmap_display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass

    if not args.headless: cv2.destroyAllWindows()

    if valid_snaps >= 20:
        print(f"\n[PROCESSING] Prima calibrazione per filtraggio outlier...")
        
        # Inizializziamo le matrici di output per stereoCalibrateExtended
        # In OpenCV 4.x, questa funzione restituisce 10 valori
        res = cv2.stereoCalibrateExtended(
            objpoints, imgpoints_L, imgpoints_R,
            mtx_L, dist_L, mtx_R, dist_R,
            (width, height),
            R=None, T=None, E=None, F=None, # Inizializziamo gli output
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Estraiamo i risultati (l'ultimo è perViewErrors)
        ret, M1, D1, M2, D2, R, T, E, F, perViewErrors = res

        # 2. Analisi degli errori per ogni coppia di immagini
        # perViewErrors ha forma (N, 2), prendiamo la media dei due sensori per ogni snap
        errors = np.mean(perViewErrors, axis=1).flatten()
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Soglia: Media + 1 Deviazione Standard (puoi essere più cattivo usando solo la media)
        threshold = mean_error + std_error 
        
        filtered_obj = []
        filtered_img_L = []
        filtered_img_R = []
        
        print(f"\n{YELLOW}{'Snap':<10} {'RMS Error':<15} {'Status':<10}{RESET}")
        for i, err in enumerate(errors):
            status = "KEEP" if err < threshold else "DISCARD"
            color = GREEN if status == "KEEP" else RED
            print(f"{color}{i:<10} {err:<15.4f} {status}{RESET}")
            
            if status == "KEEP":
                filtered_obj.append(objpoints[i])
                filtered_img_L.append(imgpoints_L[i])
                filtered_img_R.append(imgpoints_R[i])

        # 3. Seconda calibrazione con il set pulito (usiamo stereoCalibrate standard)
        print(f"\n[PROCESSING] Ricalcolo con {len(filtered_obj)} snap validi...")
        ret_final, M1f, D1f, M2f, D2f, Rf, Tf, Ef, Ff = cv2.stereoCalibrate(
            filtered_obj, filtered_img_L, filtered_img_R,
            mtx_L, dist_L, mtx_R, dist_R,
            (width, height), flags=cv2.CALIB_FIX_INTRINSIC
        )

        print(f"\n{GREEN}RMS Finale: {ret_final:.4f} pixel (Originale: {ret:.4f}){RESET}")
        save_stereo_params("stereo_params.json", M1f, D1f, M2f, D2f, Rf, Tf, Ef, Ff, width, height)
    else:
        print(f"{RED}[ERROR] Troppi pochi snap ({valid_snaps}){RESET}")

if __name__ == "__main__":
    main()
