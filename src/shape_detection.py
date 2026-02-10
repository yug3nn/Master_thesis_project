import numpy as np
import cv2
from metavision_core.event_io import EventsIterator
import time

def detect_shapes(img, output_img):
    # --- LA TUA LOGICA DI DETECTION (versione minimal) ---
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 5000: continue

        hull = cv2.convexHull(cnt)
        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        vertices = len(approx)
        
        shape_name = ""
        color = (255, 255, 255)

        if vertices == 3:
            shape_name = "Triangolo"
            color = (0, 255, 0)
        elif vertices == 4: 
            x, y, w, h = cv2.boundingRect(approx)
            ar = float(w)/h
            if 0.85 <= ar <= 1.15:
                shape_name = "Quadrato"
                color = (255, 0, 0)
            else:
                shape_name = "Rettangolo"
                color = (255, 0, 0)
        elif vertices > 4:
            shape_name = "Cerchio"
            color = (0, 0, 255)

        if shape_name:
            cv2.drawContours(output_img, [approx], 0, color, 2)
            M = cv2.moments(hull)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(output_img, shape_name, (cx-40, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    print("--- DETECTOR CON REGISTRAZIONE ---")
    print("Premi 'r' per avviare/fermare la registrazione.")
    print("Premi 'q' per uscire.")
    
    # Impostiamo 20ms (che corrisponde a 50 FPS teorici: 1000/20 = 50)
    delta_t = 20000 
    mv_it = EventsIterator(input_path="", delta_t=delta_t)
    height, width = mv_it.get_size()
    print(height, width)
    
    # Configurazione VideoWriter
    # Usiamo 'mp4v' che genera file .mp4 compatibili con Windows/Mac
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 50.0 
    # Il nome del file include un timestamp per non sovrascrivere i vecchi
    filename = f"tesi_demo_{int(time.time())}.mp4"
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    is_recording = False
    
    # Kernel per la pulizia
    erode_kernel = np.ones((2, 2), np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)

    try:
        for evs in mv_it:
            if evs.size == 0: continue

            # 1. Pipeline Immagine
            img_binary = np.zeros((height, width), dtype=np.uint8)
            img_binary[evs['y'], evs['x']] = 255
            
            #img_clean = cv2.erode(img_binary, erode_kernel, iterations=1)
            #img_solid = cv2.dilate(img_clean, dilate_kernel, iterations=1)
            img_clean = cv2.dilate(img_binary, dilate_kernel, iterations=1)
            img_solid = cv2.erode(img_clean, erode_kernel, iterations=1)
            img_final = cv2.GaussianBlur(img_solid, (3, 3), 0)
            
            img_display = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
            
            # 2. Rilevamento
            detect_shapes(img_final, img_display)

            # 3. Gestione Registrazione
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                is_recording = not is_recording
                if is_recording:
                    print(f"REC AVVIATA -> Salvataggio su {filename}")
                else:
                    print("REC STOPPATA")

            if is_recording:
                # Disegna un pallino rosso in alto a destra per far capire che sta registrando
                cv2.circle(img_display, (width - 20, 20), 8, (0, 0, 255), -1)
                # Scrivi il frame nel file video
                out.write(img_display)

            # 4. Mostra a schermo
            cv2.imshow("Detection (Premi 'r' per REC)", img_display)
            
            if key == ord('q'):
                break

    finally:
        # Rilascio risorse fondamentale per non corrompere il file video
        print("Salvataggio e chiusura...")
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
