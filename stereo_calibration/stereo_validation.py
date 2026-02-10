import cv2
import numpy as np
import json
import time
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
SERIAL_LEFT = "genx320 11-003c"   # SLAVE
SERIAL_RIGHT = "genx320 10-003c"  # MASTER
delta_t_val = 30000               # 30ms accumulo per vedere bene
BIAS_VAL = 60                     # Contrasto medio

def load_json_params():
    print("Caricamento parametri...")
    # 1. Carica parametri intrinseci singoli
    with open("calib_left.json") as f: data_L = json.load(f)
    with open("calib_right.json") as f: data_R = json.load(f)
    
    K_L = np.array(data_L["K"]).reshape(3,3)
    D_L = np.array(data_L["D"])
    K_R = np.array(data_R["K"]).reshape(3,3)
    D_R = np.array(data_R["D"])
    
    width = data_L["width"]
    height = data_L["height"]

    # 2. Carica parametri stereo (Rettifica)
    with open("stereo_params.json") as f: data_S = json.load(f)
    
    R1 = np.array(data_S["R1"])
    R2 = np.array(data_S["R2"])
    P1 = np.array(data_S["P1"])
    P2 = np.array(data_S["P2"])
    
    # 3. Genera le mappe di rettifica (Lookup Tables)
    # Queste dicono a ogni pixel dove spostarsi per "raddrizzare" l'immagine
    m1l, m2l = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (width, height), cv2.CV_16SC2)
    m1r, m2r = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (width, height), cv2.CV_16SC2)
    
    return m1l, m2l, m1r, m2r, width, height

def configure_camera(iterator, mode):
    device = iterator.reader.device
    i_sync = device.get_i_camera_synchronization()
    if mode == 'master': i_sync.set_mode_master()
    else: i_sync.set_mode_slave()
    
    try:
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on", BIAS_VAL)
        biases.set("bias_diff_off", BIAS_VAL)
    except: pass

def get_frame(evs, width, height):
    if evs.size == 0: return np.zeros((height, width), dtype=np.uint8)
    im = np.zeros((height, width), dtype=np.uint8)
    # Usiamo solo eventi ON per chiarezza
    mask = evs['p'] == 1
    im[evs[mask]['y'], evs[mask]['x']] = 255
    return im

def main():
    try:
        map1_L, map2_L, map1_R, map2_R, w, h = load_json_params()
    except Exception as e:
        print(f"Errore caricamento file JSON: {e}")
        print("Assicurati di avere calib_left.json, calib_right.json e stereo_params.json nella cartella.")
        return

    print("\n--- AVVIO VALIDAZIONE STEREO ---")
    print("Controlla che le linee verdi attraversino gli stessi punti nelle due immagini.")
    
    mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=delta_t_val)
    configure_camera(mv_L, 'slave')
    
    mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=delta_t_val)
    configure_camera(mv_R, 'master')
    
    cv2.namedWindow("Stereo Rectification Check", cv2.WINDOW_NORMAL)
    
    for evs_L, evs_R in zip(mv_L, mv_R):
        # Genera immagini grezze
        im_L = get_frame(evs_L, w, h)
        im_R = get_frame(evs_R, w, h)
        
        # --- FASE CRUCIALE: RETTIFICA ---
        # "Stiriamo" le immagini usando le matrici calcolate
        rect_L = cv2.remap(im_L, map1_L, map2_L, cv2.INTER_LINEAR)
        rect_R = cv2.remap(im_R, map1_R, map2_R, cv2.INTER_LINEAR)
        
        # Converti a colori per disegnare linee
        vis_L = cv2.cvtColor(rect_L, cv2.COLOR_GRAY2BGR)
        vis_R = cv2.cvtColor(rect_R, cv2.COLOR_GRAY2BGR)
        
        # Unisci
        combined = np.hstack((vis_L, vis_R))
        
        # --- DIAGNOSTICA VISIVA ---
        # Disegna linee orizzontali ogni 30 pixel
        for y in range(0, h, 30):
            # Linea verde attraverso tutta l'immagine combinata
            cv2.line(combined, (0, y), (w*2, y), (0, 255, 0), 1)
            
        cv2.imshow("Stereo Rectification Check", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
