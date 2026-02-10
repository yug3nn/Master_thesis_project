import cv2
import numpy as np
import json
import os
import time
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
JSON_FILE = "genx320_intrinsics.json"
DELTA_T = 30000 

def load_calibration(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    K = np.array(data["K"]).reshape(3, 3)
    D = np.array(data["D"])
    return K, D, data["width"], data["height"]

def create_displacement_heatmap(mapx, mapy, width, height):
    """
    Crea una mappa statica che mostra QUANTI pixel vengono spostati in ogni punto.
    """
    # Crea griglia di coordinate originali (x, y)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calcola la distanza tra dove il pixel ERA e dove è finito
    # dx = mapx - x_originale
    # dy = mapy - y_originale
    dx = mapx - grid_x.astype(np.float32)
    dy = mapy - grid_y.astype(np.float32)
    
    # Magnitudine dello spostamento (Teorema di Pitagora)
    displacement = np.sqrt(dx**2 + dy**2)
    
    # Statistiche per l'utente
    max_disp = np.max(displacement)
    print(f"\n[ANALISI LENTE]")
    print(f" - Spostamento massimo ai bordi: {max_disp:.2f} pixel")
    print(f" - Spostamento medio: {np.mean(displacement):.2f} pixel")
    
    # Normalizza per visualizzare (0..255)
    # Moltiplichiamo per un fattore per renderlo visibile anche se basso
    disp_vis = (displacement / max_disp * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    return heatmap, max_disp

def main():
    print("--- CALIBRATION STRESS TEST ---")
    print("1. FLICKER VIEW: Guarda i BORDI. Se 'respirano', funziona.")
    print("2. HEATMAP: Ti mostra di quanto stiamo deformando l'immagine.")
    
    K, D, width, height = load_calibration(JSON_FILE)
    
    # Pre-calcola mappe
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (width, height), 5)
    
    # Crea la mappa di spostamento (statica)
    heatmap_img, max_shift_px = create_displacement_heatmap(mapx, mapy, width, height)
    
    mv_it = EventsIterator(input_path="", delta_t=DELTA_T)
    
    last_switch = time.time()
    show_undistorted = False

    for evs in mv_it:
        if evs.size == 0: continue

        # Genera frame
        im = np.zeros((height, width), dtype=np.uint8)
        im[evs['y'], evs['x']] = 255
        img_raw = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        
        # Undistort
        img_undist = cv2.remap(img_raw, mapx, mapy, cv2.INTER_LINEAR)
        
        # --- MODO 1: FLICKER TEST (Alterna ogni 0.8s) ---
        now = time.time()
        if now - last_switch > 0.8:
            show_undistorted = not show_undistorted
            last_switch = now
            
        if show_undistorted:
            view = img_undist.copy()
            label = "UNDISTORTED (Corrected)"
            color = (0, 255, 0)
            # Aggiungi un bordo verde per far capire che è attiva
            cv2.rectangle(view, (0,0), (width-1, height-1), (0,255,0), 4)
        else:
            view = img_raw.copy()
            label = "RAW (Distorted)"
            color = (0, 0, 255)
            # Aggiungi un bordo rosso
            cv2.rectangle(view, (0,0), (width-1, height-1), (0,0,255), 4)
            
        cv2.putText(view, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # --- MODO 2: HEATMAP OVERLAY ---
        # Sovrapponi la heatmap all'immagine raw per vedere DOVE agisce
        # Blend: 70% immagine, 30% heatmap
        overlay = cv2.addWeighted(img_raw, 0.7, heatmap_img, 0.3, 0)
        cv2.putText(overlay, f"Max Shift: {max_shift_px:.1f}px", (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Mostra
        cv2.imshow("1. FLICKER TEST (Watch Edges)", view)
        cv2.imshow("2. DISPLACEMENT MAP (Heatmap)", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
