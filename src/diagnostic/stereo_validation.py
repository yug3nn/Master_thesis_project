import cv2
import numpy as np
import json
import time
import os
from metavision_core.event_io import EventsIterator

# --- CONFIGURATION ---
SERIAL_LEFT = "genx320 11-003c"   # SLAVE
SERIAL_RIGHT = "genx320 10-003c"  # MASTER
delta_t_val = 30000               
BIAS_VAL = 60                     

def load_json_params():
    print("Loading parameters from config folder...")
    path_S = os.path.join("config", "stereo_params.json")
    
    if not os.path.exists(path_S):
        raise FileNotFoundError(f"Missing {path_S}")

    with open(path_S) as f: 
        data_S = json.load(f)
    
    width = data_S["width"]
    height = data_S["height"]

    # --- FIX: FORCE FLOAT64 DATA TYPE FOR ALL MATRICES ---
    # OpenCV requires consistent data types for stereoRectify
    K_L = np.array(data_S["camera_left"]["K"], dtype=np.float64)
    D_L = np.array(data_S["camera_left"]["D"], dtype=np.float64)
    K_R = np.array(data_S["camera_right"]["K"], dtype=np.float64)
    D_R = np.array(data_S["camera_right"]["D"], dtype=np.float64)
    R = np.array(data_S["stereo"]["R"], dtype=np.float64)
    T = np.array(data_S["stereo"]["T"], dtype=np.float64)

    print("Computing Stereo Rectification...")
    # Compute Rectification Transforms
    # alpha=0 crops the image to only valid pixels, alpha=1 keeps all pixels
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_L, D_L, K_R, D_R, (width, height), R, T, alpha=0
    )

    # Generate Rectification Maps
    # Using CV_32FC1 for better performance on Raspberry Pi 5
    m1l, m2l = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (width, height), cv2.CV_32FC1)
    m1r, m2r = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (width, height), cv2.CV_32FC1)
    
    return m1l, m2l, m1r, m2r, width, height

def configure_camera(iterator, mode):
    """ Hardware setup for GenX320 """
    try:
        device = iterator.reader.device
        biases = device.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", BIAS_VAL)
            biases.set("bias_diff_off", BIAS_VAL)
        
        i_sync = device.get_i_camera_synchronization()
        if mode == 'master':
            i_sync.set_mode_master()
        else:
            i_sync.set_mode_slave()
    except Exception as e:
        print(f"Hardware config info for {mode}: {e}")

def get_frame(evs, w, h):
    im = np.zeros((h, w), dtype=np.uint8)
    if evs.size > 0:
        # Use only Polarity 1 (ON events) for cleaner edge detection
        mask = evs['p'] == 1
        im[evs[mask]['y'], evs[mask]['x']] = 255
    return im

def main():
    try:
        map1_L, map2_L, map1_R, map2_R, w, h = load_json_params()
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("\n--- STEREO RECTIFICATION CHECK ---")
    print("Alignment check: The green lines must cross the same objects in both views.")
    
    mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=delta_t_val)
    configure_camera(mv_L, 'slave')
    
    mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=delta_t_val)
    configure_camera(mv_R, 'master')
    
    cv2.namedWindow("Stereo Rectification Check", cv2.WINDOW_NORMAL)
    
    for evs_L, evs_R in zip(mv_L, mv_R):
        im_L = get_frame(evs_L, w, h)
        im_R = get_frame(evs_R, w, h)
        
        # Apply rectification remap
        rect_L = cv2.remap(im_L, map1_L, map2_L, cv2.INTER_LINEAR)
        rect_R = cv2.remap(im_R, map1_R, map2_R, cv2.INTER_LINEAR)
        
        vis_L = cv2.cvtColor(rect_L, cv2.COLOR_GRAY2BGR)
        vis_R = cv2.cvtColor(rect_R, cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack((vis_L, vis_R))
        
        # Draw horizontal lines every 30 pixels
        for y in range(30, h, 30):
            cv2.line(combined, (0, y), (w*2, y), (0, 255, 0), 1)
            
        cv2.imshow("Stereo Rectification Check", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
