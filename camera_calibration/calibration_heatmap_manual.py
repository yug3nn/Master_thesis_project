import cv2
import numpy as np
import json
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
CHECKERBOARD_ROWS = 6     # Angoli interni verticali
CHECKERBOARD_COLS = 9     # Angoli interni orizzontali
SQUARE_SIZE_M = 0.018     # Lato del quadrato in metri

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

def generate_heatmap_vis(accumulated_mask):
    """ Converts the accumulated binary mask into a colored heatmap. """
    vis = accumulated_mask.copy()
    vis = np.clip(vis, 0, 10) 
    vis = (vis * 25.5).astype(np.uint8)
    vis_blur = cv2.GaussianBlur(vis, (31, 31), 0)
    heatmap_color = cv2.applyColorMap(vis_blur, cv2.COLORMAP_JET)
    return heatmap_color

def main():
    print("--- DEBUG CALIBRATION TOOL ---")
    print("INSTRUCTIONS:")
    print("  - SPACEBAR: Cattura foto e apri finestra di ispezione (ANCHE SE NON VALIDA).")
    print("  - Q: Esci e calcola.")
    print(f"  - Target: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} angoli interni.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_ROWS, 0:CHECKERBOARD_COLS].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_M

    objpoints = [] 
    imgpoints = [] 

    # NOTA: 50000us (50ms) potrebbe essere tanto e creare motion blur. 
    # Se vedi le immagini "strisciate", prova ad abbassare a 20000 o 30000.
    mv_it = EventsIterator(input_path="", delta_t=100000) 
    height, width = mv_it.get_size()
    
    coverage_mask = np.zeros((height, width), dtype=np.float32)
    valid_snaps = 0

    for evs in mv_it:
        if evs.size == 0: continue

        # Modifica filtro polarità
        mask_on = evs['p'] == 1
        evs_filtered = evs[mask_on]

        # 1. Generazione Immagine
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs_filtered['y'], evs_filtered['x']] = 255
        
        # --- PRE-PROCESSING OPZIONALE (Decommenta se serve aiuto) ---
        # kernel = np.ones((2,2), np.uint8)
        # im = cv2.dilate(im, kernel, iterations=1) # "Ingrassa" i pixel
        
        im_vis = cv2.bitwise_not(im) # Inverti per OpenCV
        display = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # 2. Rilevamento
        ret, corners = cv2.findChessboardCorners(im_vis, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), None)

        status_text = "Move Board..."
        color_text = (0, 0, 255) # Rosso

        if ret:
            corners2 = cv2.cornerSubPix(im_vis, corners, (11, 11), (-1, -1), criteria)
            # Disegna la scacchiera trovata sull'immagine LIVE
            cv2.drawChessboardCorners(display, (CHECKERBOARD_ROWS, CHECKERBOARD_COLS), corners2, ret)
            status_text = "READY (Press SPACE)"
            color_text = (0, 255, 0) # Verde

        # 3. Gestione Input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32: # BARRA SPAZIATRICE
            # Creiamo una copia per la finestra di ispezione
            snapshot_view = display.copy()
            
            if ret:
                # --- CASO VALIDO ---
                objpoints.append(objp)
                imgpoints.append(corners2)
                valid_snaps += 1
                
                # Aggiorna Heatmap
                for corner in corners2:
                    x, y = int(corner[0][0]), int(corner[0][1])
                    cv2.circle(coverage_mask, (x, y), 5, 1.0, -1)
                
                print(f"[Captured] Snapshot {valid_snaps} SALVATO.")
                
                # Feedback Visivo nella finestra snapshot
                cv2.putText(snapshot_view, "VALID: SAVED", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(display, (0,0), (width, height), (255,255,255), -1) # Flash
            
            else:
                # --- CASO NON VALIDO (Solo visualizzazione) ---
                print("[DEBUG] Foto scattata ma NON VALIDA (Scacchiera non trovata).")
                
                # Scriviamo in ROSSO sull'immagine
                cv2.putText(snapshot_view, "INVALID: NOT SAVED", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(snapshot_view, "Check: Focus / Conteggio / Frame", (10, height-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # --- APRI LA FINESTRA DI ISPEZIONE ---
            cv2.imshow("Snapshot Inspection", snapshot_view)

        if key == ord('q'):
            break

        # 4. Aggiorna finestre principali
        heatmap_display = generate_heatmap_vis(coverage_mask)
        
        cv2.putText(display, f"Snaps: {valid_snaps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, status_text, (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)

        cv2.imshow("DEBUG VIEW (Raw)", im_vis)        
        cv2.imshow("Live Camera", display)
        cv2.imshow("Coverage Heatmap", heatmap_display)

    cv2.destroyAllWindows()

    # 5. Calcolo Finale
    if valid_snaps >= 15:
        print("\n[Processing] Calculating calibration parameters...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        print(f"RMS Error: {ret:.4f} pixels")
        save_prophesee_json("genx320_intrinsics.json", width, height, mtx, dist)
    else:
        print("\n[Error] Not enough valid data (<15 snaps).")

if __name__ == "__main__":
    main()
