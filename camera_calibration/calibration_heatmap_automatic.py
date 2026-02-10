import cv2
import numpy as np
import json
import time
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
CHECKERBOARD_ROWS = 6     # Angoli interni verticali
CHECKERBOARD_COLS = 9     # Angoli interni orizzontali
SQUARE_SIZE_M = 0.0165     # Lato del quadrato in metri
COOLDOWN_SECONDS = 0.5    # Tempo tra scatti

def save_prophesee_json(filename, width, height, mtx, dist):
    """ Save calibration in Metavision SDK format """
    data = {
        "type": "pinhole",
        "width": width,
        "height": height,
        "K": [mtx[0,0], mtx[0,1], mtx[0,2], 
              mtx[1,0], mtx[1,1], mtx[1,2], 
              mtx[2,0], mtx[2,1], mtx[2,2]],
        "D": dist.flatten().tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[SUCCESS] Calibration saved to: {filename}")

def generate_point_heatmap(accumulated_mask):
    """ Mostra i singoli punti accumulati """
    vis = accumulated_mask * 20.0
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    heatmap_color[vis == 0] = [0, 0, 0]
    return heatmap_color

def main():
    print("--- HYBRID CALIBRATION (SB + Standard) ---")
    print("INSTRUCTIONS:")
    print("  - GREEN TEXT = SB Method (High Precision)")
    print("  - YELLOW TEXT = Standard Method (Fallback)")
    
    # Criteri per cornerSubPix (usato solo nel metodo standard)
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Flags per il metodo SB (Sector Based) - Alta precisione
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    
    # Flags per il metodo Standard
    flags_std = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_M

    objpoints = [] 
    imgpoints = [] 

    mv_it = EventsIterator(input_path="", delta_t=30000) 
    height, width = mv_it.get_size()
    
    point_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps = 0
    last_capture_time = time.time()

    for evs in mv_it:
        if evs.size == 0: continue

        # --- 1. Generazione Immagine ---
        # Filtro polarità facoltativo (puoi toglierlo se usi scacchiera statica + shake)
        mask_on = evs['p'] == 1 
        evs_filtered = evs[mask_on] 
        
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs_filtered['y'], evs_filtered['x']] = 255
        
        # Dilatazione leggera
        kernel = np.ones((2,2), np.uint8)
        im_filled = cv2.dilate(im, kernel, iterations=1)
        im_vis = cv2.bitwise_not(im_filled) 
        display = cv2.cvtColor(im_vis, cv2.COLOR_GRAY2BGR)

        # --- 2. RILEVAMENTO IBRIDO ---
        corners_found = None
        method_used = ""
        ret = False

        # TENTATIVO 1: Sector Based (SB) - Il migliore
        try:
            ret, corners = cv2.findChessboardCornersSB(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_sb)
            if ret:
                corners_found = corners
                method_used = "SB (Precise)"
                color_text = (0, 255, 0) # Verde
        except cv2.error:
            pass # Ignora errori se la versione OpenCV è vecchia o fallisce

        # TENTATIVO 2: Standard + SubPix (Fallback)
        if not ret:
            ret, corners = cv2.findChessboardCorners(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), flags=flags_std)
            if ret:
                # Qui dobbiamo raffinare manualmente
                corners_found = cv2.cornerSubPix(im_vis, corners, (11, 11), (-1, -1), criteria_subpix)
                method_used = "Standard (SubPix)"
                color_text = (0, 255, 255) # Giallo

        # --- 3. VISUALIZZAZIONE E CATTURA ---
        status_text = "Searching..."
        
        if ret and corners_found is not None:
            # Disegna
            cv2.drawChessboardCorners(display, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners_found, ret)
            status_text = f"FOUND: {method_used}"
            
            # Auto-Capture Logic
            current_time = time.time()
            if current_time - last_capture_time > COOLDOWN_SECONDS:
                objpoints.append(objp)
                imgpoints.append(corners_found)
                valid_snaps += 1
                last_capture_time = current_time
                
                # Update Heatmap
                for corner in corners_found:
                    x, y = int(corner[0][0]), int(corner[0][1])
                    cv2.circle(point_mask, (x, y), 2, 1.0, -1)
                
                print(f"[AUTO] Snap {valid_snaps} via {method_used}")
                cv2.rectangle(display, (0,0), (width, height), (255,255,255), -1) # Flash
                status_text = "CAPTURED!"

        # UI Info
        heatmap_display = generate_point_heatmap(point_mask)
        cv2.putText(display, f"Snaps: {valid_snaps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if status_text != "CAPTURED!":
             cv2.putText(display, status_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text if ret else (0,0,255), 2)

        cv2.imshow("Hybrid Calibration", display)
        cv2.imshow("Point Density", heatmap_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # --- 4. CALCOLO ---
    if valid_snaps >= 20:
        print("\n[Processing] Calculating calibration parameters...")
        # Nota: OpenCV non sa quale metodo hai usato per ogni frame, 
        # prende solo le coordinate float dei punti.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        
        print(f"RMS Error: {ret:.4f} pixels")
        if ret < 1.0: print(" [!!!] ECCELLENTE (< 1.0 px)")
        elif ret < 3.0: print(" [OK] BUONO")
        
        save_prophesee_json("genx320_intrinsics.json", width, height, mtx, dist)
    else:
        print(f"\n[Error] Too few snaps ({valid_snaps}).")

if __name__ == "__main__":
    main()
