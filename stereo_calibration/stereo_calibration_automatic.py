import cv2
import numpy as np
import json
import time
import sys
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
HEADLESS_MODE = False        
CHECKERBOARD_ROWS = 6       
CHECKERBOARD_COLS = 9       
SQUARE_SIZE_M = 0.0165      
COOLDOWN_SECONDS = 3.0      

delta_t_val = 30000         
MIN_EVENTS_THRESHOLD = 5000 
MAX_SYNC_DIFF_US = 1000    

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

FLAGS_SB = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def load_prophesee_json(filename):
    print(f"Loading calibration from {filename}...")
    with open(filename, 'r') as f:
        data = json.load(f)
    k_list = data["K"]
    mtx = np.array([[k_list[0], k_list[1], k_list[2]],
                    [k_list[3], k_list[4], k_list[5]],
                    [k_list[6], k_list[7], k_list[8]]], dtype=np.float32)
    dist = np.array(data["D"], dtype=np.float32)
    return mtx, dist, data["width"], data["height"]

def save_stereo_json(filename, R, T, R1, R2, P1, P2, Q, size):
    data = {
        "width": size[0], "height": size[1],
        "R": R.tolist(), "T": T.tolist(),
        "R1": R1.tolist(), "R2": R2.tolist(),
        "P1": P1.tolist(), "P2": P2.tolist(), "Q": Q.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n{GREEN}[SUCCESS] Stereo params saved to: {filename}{RESET}")

def configure_camera(iterator, mode, name):
    device = iterator.reader.device
    i_sync = device.get_i_camera_synchronization()
    if not i_sync:
        print(f"[{name}] ERRORE: Sync non supportato!")
        return False
        
    if mode == 'master':
        i_sync.set_mode_master()
        print(f"[{name}] MASTER (Clock Output attivo)")
    elif mode == 'slave':
        i_sync.set_mode_slave()
        print(f"[{name}] SLAVE (In attesa di Clock...)")
        
    try:
        biases = device.get_i_ll_biases()
        if biases:
            current_on = biases.get("bias_diff_on")
            current_off = biases.get("bias_diff_off")
            biases.set("bias_diff_on", current_on + 30)
            biases.set("bias_diff_off", current_off + 30)
    except:
        pass
    return True

def get_frame_from_events(evs, width, height):
    if evs.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    mask_on = evs['p'] == 1 
    evs_filtered = evs[mask_on] 
    im = np.zeros((height, width), dtype=np.uint8)
    im[evs_filtered['y'], evs_filtered['x']] = 255
    kernel = np.ones((2,2), np.uint8)
    im = cv2.dilate(im, kernel, iterations=1)
    return cv2.bitwise_not(im)

def generate_point_heatmap(accumulated_mask):
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def main():
    mode_str = "HEADLESS (No GUI)" if HEADLESS_MODE else "GUI (Windowed)"
    print(f"--- STEREO CALIBRATION: {mode_str} ---")
    
    try:
        mtx_L, dist_L, w_L, h_L = load_prophesee_json("calib_left.json")
        mtx_R, dist_R, w_R, h_R = load_prophesee_json("calib_right.json")
    except FileNotFoundError:
        print("[ERROR] Mancano i file JSON di calibrazione singola!")
        return

    width, height = w_L, h_L
    
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_M

    objpoints = [] 
    imgpoints_L = []
    imgpoints_R = []

    mask_L = np.zeros((height, width), dtype=np.float32)
    mask_R = np.zeros((height, width), dtype=np.float32)
    
    # --- CACHE VISUALIZZAZIONE ---
    # Inizializziamo l'immagine Heatmap nera (così imshow non crasha al primo giro)
    # La aggiorneremo SOLO quando scattiamo una foto
    heatmap_view = np.zeros((height, width * 2, 3), dtype=np.uint8)

    print("\n1. Avvio SLAVE (Left)...")
    try:
        mv_it_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=delta_t_val)
        configure_camera(mv_it_L, 'slave', "LEFT")
    except Exception as e:
        print(f"Errore Left: {e}")
        return

    time.sleep(2.0)

    print("2. Avvio MASTER (Right)...")
    try:
        mv_it_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=delta_t_val)
        configure_camera(mv_it_R, 'master', "RIGHT")
    except Exception as e:
        print(f"Errore Right: {e}")
        return

    print("\n>>> STREAMING AVVIATO <<<")
    
    valid_snaps = 0
    last_capture_time = time.time()
    
    try:
        for evs_L, evs_R in zip(mv_it_L, mv_it_R): 
            # --- 1. CONTROLLO SYNC ---
            if evs_L.size > 0 and evs_R.size > 0:
                ts_L = evs_L['t'][-1]
                ts_R = evs_R['t'][-1]
                diff = abs(ts_L - ts_R)
                if diff > MAX_SYNC_DIFF_US:
                    print(f"\n{RED}[SYNC WARNING] Diff: {diff} µs!{RESET}")

            # --- 2. LOGICA RILEVAMENTO ---
            is_flash = (evs_L.size > MIN_EVENTS_THRESHOLD) and (evs_R.size > MIN_EVENTS_THRESHOLD)
            
            im_L = get_frame_from_events(evs_L, width, height)
            im_R = get_frame_from_events(evs_R, width, height)

            status_msg = "Waiting..."
            found_L, found_R = False, False
            corners_L, corners_R = None, None

            if is_flash:
                fast_flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH
                has_grid_L, _ = cv2.findChessboardCorners(im_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=fast_flags)
                has_grid_R, _ = cv2.findChessboardCorners(im_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=fast_flags)
                
                if has_grid_L and has_grid_R:
                    found_L, corners_L = cv2.findChessboardCornersSB(im_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=FLAGS_SB)
                    found_R, corners_R = cv2.findChessboardCornersSB(im_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=FLAGS_SB)

                    if found_L and found_R:
                        status_msg = f"{YELLOW}BOARD SEEN{RESET}"

                        curr_time = time.time()
                        if (curr_time - last_capture_time) > COOLDOWN_SECONDS:
                            objpoints.append(objp)
                            imgpoints_L.append(corners_L)
                            imgpoints_R.append(corners_R)
                            valid_snaps += 1
                            last_capture_time = curr_time

                            # --- AGGIORNAMENTO HEATMAP (SOLO QUI!) ---
                            # Questo codice pesante gira solo una volta ogni 3 secondi
                            for c in corners_L: cv2.circle(mask_L, (int(c[0][0]), int(c[0][1])), 5, 1, -1)
                            for c in corners_R: cv2.circle(mask_R, (int(c[0][0]), int(c[0][1])), 5, 1, -1)
                            
                            hm_L = generate_point_heatmap(mask_L)
                            hm_R = generate_point_heatmap(mask_R)
                            heatmap_view = np.hstack((hm_L, hm_R)) # Aggiorna la cache
                            
                            print(f"\n{GREEN}>>> SNAP {valid_snaps} CATTURATO! <<<{RESET}")
                            print('\a')
                            status_msg = "CAPTURED!"

            # --- 3. SEZIONE VISUALIZZAZIONE ---
            if not HEADLESS_MODE:
                vis_L = cv2.cvtColor(im_L, cv2.COLOR_GRAY2BGR)
                vis_R = cv2.cvtColor(im_R, cv2.COLOR_GRAY2BGR)
                
                if found_L: cv2.drawChessboardCorners(vis_L, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners_L, found_L)
                if found_R: cv2.drawChessboardCorners(vis_R, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners_R, found_R)
                
                if status_msg == "CAPTURED!":
                    cv2.rectangle(vis_L, (0,0), (width, height), (0,255,0), 5)

                combined = np.hstack((vis_L, vis_R))
                cv2.imshow("Stereo View", combined)
                
                # MOSTRA LA HEATMAP CACHATA (Costo quasi zero)
                cv2.imshow("Coverage", heatmap_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                sys.stdout.write(f"\r[HEADLESS] Snaps: {valid_snaps} | Evs: {evs_L.size}/{evs_R.size} | Status: {status_msg}" + " "*10)
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nInterruzione manuale.")

    if not HEADLESS_MODE:
        cv2.destroyAllWindows()

    if valid_snaps >= 15:
        print("\n[CALCOLO IN CORSO...]")
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        
        ret, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_L, imgpoints_R,
            mtx_L, dist_L, mtx_R, dist_R,
            (width, height), criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC
        )
        print(f"Stereo RMS Error: {ret:.4f}")
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(M1, D1, M2, D2, (width, height), R, T)
        save_stereo_json("stereo_params.json", R, T, R1, R2, P1, P2, Q, (width, height))
    else:
        print(f"\n{RED}[ERRORE] Pochi snap ({valid_snaps}).{RESET}")

if __name__ == "__main__":
    main()
