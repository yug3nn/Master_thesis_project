import cv2
import numpy as np
import json
import os
import time
import argparse # Aggiunto per gestire gli argomenti
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
DELTA_T = 30000 

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

def load_calibration(filename):
    # Costruisce il percorso verso la cartella config
    filepath = os.path.join("config", filename)
    with open(filepath, 'r') as f:
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
    disp_vis = (displacement / max_disp * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    return heatmap, max_disp

def main():
    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Calibration Stress Test for GenX320")
    parser.add_argument("--side", choices=["left", "right"], required=True, 
                        help="Specifica se testare 'left' o 'right'")
    args = parser.parse_args()
    
    # Definizione del file in base al lato scelto
    json_file = f"camera_{args.side}.json"

    serial = SERIAL_LEFT
    if args.side == "right":
        serial = SERIAL_RIGHT

    print(f"--- CALIBRATION STRESS TEST: {args.side.upper()} CAMERA ---")
    print("1. FLICKER VIEW: Guarda i BORDI. Se 'respirano', funziona.")
    print("2. HEATMAP: Ti mostra di quanto stiamo deformando l'immagine.")
    
    try:
        K, D, width, height = load_calibration(json_file)
    except FileNotFoundError:
        print(f"[ERRORE] File config/{json_file} non trovato!")
        return
    
    # Pre-calcola mappe
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (width, height), 5)
    
    # Crea la mappa di spostamento (statica)
    heatmap_img, max_shift_px = create_displacement_heatmap(mapx, mapy, width, height)
    
    mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T)
    
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
            cv2.rectangle(view, (0,0), (width-1, height-1), (0,255,0), 4)
        else:
            view = img_raw.copy()
            label = "RAW (Distorted)"
            color = (0, 0, 255)
            cv2.rectangle(view, (0,0), (width-1, height-1), (0,0,255), 4)
            
        cv2.putText(view, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # --- MODO 2: HEATMAP OVERLAY ---
        overlay = cv2.addWeighted(img_raw, 0.7, heatmap_img, 0.3, 0)
        cv2.putText(overlay, f"Max Shift: {max_shift_px:.1f}px", (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Mostra
        cv2.imshow(f"1. FLICKER TEST ({args.side})", view)
        cv2.imshow(f"2. DISPLACEMENT MAP ({args.side})", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
