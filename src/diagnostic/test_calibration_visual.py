import cv2
import numpy as np
import json
import os
import time
import argparse
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
DELTA_T = 30000 

SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c" 

def load_calibration(filepath):
    """
    Carica la calibrazione supportando sia il formato standard (K, D) 
    che quello generato da analyze_rms (camera_matrix, dist_coeffs).
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Supporto per i diversi nomi delle chiavi nei JSON
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"])
        D = np.array(data["dist_coeffs"])
    elif "K" in data:
        K = np.array(data["K"]).reshape(3, 3)
        D = np.array(data["D"])
    else:
        raise KeyError(f"Formato JSON non riconosciuto in {filepath}")
        
    # Dimensioni (default 320 per le tue GenX320 se mancano nel JSON)
    width = data.get("width", 320)
    height = data.get("height", 320)
    
    return K, D, width, height

def create_displacement_heatmap(mapx, mapy, width, height):
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    dx = mapx - grid_x.astype(np.float32)
    dy = mapy - grid_y.astype(np.float32)
    displacement = np.sqrt(dx**2 + dy**2)
    
    max_disp = np.max(displacement)
    print(f"\n[LENS ANALYSIS]")
    print(f" - Max displacement at edges: {max_disp:.2f} pixels")
    print(f" - Average displacement: {np.mean(displacement):.2f} pixels")
    
    disp_vis = (displacement / max_disp * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return heatmap, max_disp

def main():
    parser = argparse.ArgumentParser(description="Calibration Stress Test for GenX320")
    parser.add_argument("--side", choices=["left", "right"], required=True, 
                        help="Specify camera side (used for serial selection)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to a specific JSON file (e.g. data_analysis/intrinsics_right.json)")
    args = parser.parse_args()
    
    # Logica di selezione del file
    if args.json:
        json_path = args.json
    else:
        json_path = os.path.join("config", f"camera_{args.side}.json")

    # Selezione seriale in base al lato
    serial = SERIAL_LEFT if args.side == "left" else SERIAL_RIGHT

    print(f"--- CALIBRATION STRESS TEST: {args.side.upper()} CAMERA ---")
    print(f"Loading parameters from: {json_path}")
    print("1. FLICKER VIEW: Watch the EDGES. If they 'breathe', it's working.")
    print("2. HEATMAP: Shows the intensity of the undistortion warp.")
    
    try:
        K, D, width, height = load_calibration(json_path)
    except Exception as e:
        print(f"[ERROR] Could not load calibration: {e}")
        return
    
    # Pre-calcolo mappe
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (width, height), 5)
    
    heatmap_img, max_shift_px = create_displacement_heatmap(mapx, mapy, width, height)
    
    mv_it = EventsIterator(input_path=serial, delta_t=DELTA_T)
    
    last_switch = time.time()
    show_undistorted = False

    for evs in mv_it:
        if evs.size == 0: continue

        im = np.zeros((height, width), dtype=np.uint8)
        im[evs['y'], evs['x']] = 255
        img_raw = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        
        # Undistort
        img_undist = cv2.remap(img_raw, mapx, mapy, cv2.INTER_LINEAR)
        
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
        
        overlay = cv2.addWeighted(img_raw, 0.7, heatmap_img, 0.3, 0)
        cv2.putText(overlay, f"Max Shift: {max_shift_px:.1f}px", (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(f"1. FLICKER TEST ({args.side})", view)
        cv2.imshow(f"2. DISPLACEMENT MAP ({args.side})", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
