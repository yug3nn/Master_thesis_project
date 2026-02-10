import numpy as np
import cv2
from metavision_core.event_io import EventsIterator

def detect_shapes_blind(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500: continue # Ignora rumore piccolo
        
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        if vertices == 3: found.append("Triangolo")
        elif vertices == 4: found.append("Quadrilatero")
        elif vertices > 6: found.append("Cerchio")
    
    return found

def main():
    print("Avvio Shape Detection (Modalità Testuale)...")
    # Delta_t = 30ms (più reattivo)
    mv_it = EventsIterator(input_path="", delta_t=30000)
    height, width = mv_it.get_size()
    
    try:
        for i, evs in enumerate(mv_it):
            if evs.size == 0: continue

            # Creazione immagine binaria (veloce)
            img_binary = np.zeros((height, width), dtype=np.uint8)
            img_binary[evs['y'], evs['x']] = 255
            
            # Pulizia rumore (opzionale ma consigliata)
            # img_binary = cv2.GaussianBlur(img_binary, (5, 5), 0)

            # Rilevamento
            shapes = detect_shapes_blind(img_binary)
            
            if shapes:
                print(f"Frame {i}: Ho visto -> {', '.join(shapes)}")
                
    except KeyboardInterrupt:
        print("\nChiusura script.")

if __name__ == "__main__":
    main()