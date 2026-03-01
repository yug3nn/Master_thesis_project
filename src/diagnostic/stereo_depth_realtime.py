import cv2
import os
import numpy as np
import json
import sys
from metavision_core.event_io import EventsIterator

sys.path.append(os.getcwd())

from src.utils.settings import (
    SERIAL_LEFT, SERIAL_RIGHT, STEREO_DELTA_T, BIAS_INCREMENT_STEREO
)
                  
DISPLAY_SCALE = 3.0               # Zoom 3x

def load_json_params():
    # Caricamento dal percorso config/stereo_params.json
    path_S = os.path.join("config", "stereo_params.json")
    try:
        with open(path_S) as f: 
            data_S = json.load(f)
    except FileNotFoundError:
        print(f"ERRORE: File {path_S} non trovato.")
        exit()
    
    width, height = data_S["width"], data_S["height"]
    
    # Estrazione parametri con casting esplicito per compatibilità OpenCV
    K_L = np.array(data_S["camera_left"]["K"], dtype=np.float64)
    D_L = np.array(data_S["camera_left"]["D"], dtype=np.float64)
    K_R = np.array(data_S["camera_right"]["K"], dtype=np.float64)
    D_R = np.array(data_S["camera_right"]["D"], dtype=np.float64)
    R = np.array(data_S["stereo"]["R"], dtype=np.float64)
    T = np.array(data_S["stereo"]["T"], dtype=np.float64)

    # Calcolo rettifica
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (width, height), R, T, alpha=0)
    
    # Mappe di rettifica
    m1l, m2l = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (width, height), cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (width, height), cv2.CV_32FC1)
    
    return m1l, m2l, m1r, m2r, width, height

def configure_camera(iterator, mode):
    device = iterator.reader.device
    try:
        i_sync = device.get_i_camera_synchronization()
        if mode == 'master': i_sync.set_mode_master()
        else: i_sync.set_mode_slave()
        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", biases.get("bias_diff_on") + BIAS_INCREMENT_STEREO)
            biases.set("bias_diff_off", biases.get("bias_diff_off") + BIAS_INCREMENT_STEREO)
    except: pass

def get_frame(evs, width, height):
    im = np.zeros((height, width), dtype=np.uint8)
    if evs.size > 0:
        im[evs['y'], evs['x']] = 255
    kernel = np.ones((3,3), np.uint8)
    #im = cv2.dilate(im, kernel, iterations=1)
    return im

def nothing(x): pass

def main():
    map1_L, map2_L, map1_R, map2_R, w, h = load_json_params()
    
    mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=STEREO_DELTA_T)
    configure_camera(mv_L, 'slave')
    mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=STEREO_DELTA_T)
    configure_camera(mv_R, 'master')
    
    cv2.namedWindow("Clean Depth", cv2.WINDOW_NORMAL)
    
    # --- SLIDER ---
    cv2.createTrackbar('Num Disp (16x)', 'Clean Depth', 4, 8, nothing) 
    cv2.createTrackbar('Block Size', 'Clean Depth', 2, 5, nothing)     
    cv2.createTrackbar('Min Disp', 'Clean Depth', 0, 30, nothing)
    
    # --- NUOVO SLIDER: CLIP FAR ---
    # Tutto ciò che ha una disparità inferiore a X (cioè è lontano) diventa NERO
    cv2.createTrackbar('Clip Far (Threshold)', 'Clean Depth', 10, 64, nothing)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 1 * 5**2,
        P2=32 * 1 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print(">>> CLEAN MODE AVVIATA <<<")
    print("Usa 'Clip Far' per eliminare lo sfondo azzurro.")
    
    try:
        for evs_L, evs_R in zip(mv_L, mv_R):
            
            im_L = get_frame(evs_L, w, h)
            im_R = get_frame(evs_R, w, h)
            
            rect_L = cv2.remap(im_L, map1_L, map2_L, cv2.INTER_LINEAR)
            rect_R = cv2.remap(im_R, map1_R, map2_R, cv2.INTER_LINEAR)
            
            small_L = cv2.resize(rect_L, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            small_R = cv2.resize(rect_R, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            
            # Parametri
            n_disp = (cv2.getTrackbarPos('Num Disp (16x)', 'Clean Depth') + 1) * 16
            blk = (cv2.getTrackbarPos('Block Size', 'Clean Depth') * 2) + 3
            min_d = cv2.getTrackbarPos('Min Disp', 'Clean Depth') // 2
            clip_thresh = cv2.getTrackbarPos('Clip Far (Threshold)', 'Clean Depth')
            
            stereo.setNumDisparities(n_disp)
            stereo.setBlockSize(blk)
            stereo.setMinDisparity(min_d)
            
            # Calcolo Disparità
            disp_small = stereo.compute(small_L, small_R).astype(np.float32) / 16.0
            
            # Upscale
            disp_large = cv2.resize(disp_small, (w, h), interpolation=cv2.INTER_NEAREST)
            disp_large *= 2.0 
            
            # --- 1. PULIZIA: EVENT MASK ---
            # Se nell'immagine rettificata sinistra il pixel è nero (< 30),
            # allora NON c'è evento, quindi la profondità deve essere 0.
            # Questo elimina le strisce dove non c'è movimento.
            mask_no_events = rect_L < 30
            disp_large[mask_no_events] = 0

            # --- 2. PULIZIA: DISTANCE CLIP ---
            # Se la disparità è bassa (oggetto lontano), forza a 0.
            # Questo elimina le strisce azzurre di sfondo.
            mask_too_far = disp_large < clip_thresh
            disp_large[mask_too_far] = 0

            # Normalizzazione Visuale
            # Ora che abbiamo pulito il fondo, possiamo normalizzare meglio
            disp_vis = disp_large / 64.0 
            disp_vis = np.clip(disp_vis, 0, 1)
            disp_color = cv2.applyColorMap((disp_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Riapplichiamo il nero assoluto dove abbiamo tagliato (per sicurezza post-colormap)
            disp_color[mask_no_events] = 0
            disp_color[mask_too_far] = 0

            vis_L = cv2.cvtColor(rect_L, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((vis_L, disp_color))
            
            big_view = cv2.resize(combined, (0,0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Clean Depth", big_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stop.")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

