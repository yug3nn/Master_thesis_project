from metavision_core.event_io import EventsIterator
import cv2
import numpy as np
import time

# IL SERIALE DELLA CAMERA "MALATA"
# (Conferma che sia questo quello che ti dava problemi nei log)
SERIAL_TARGET = "genx320 10-003c" 

def main():
    print(f"--- TEST SINGOLO DISPERATO: {SERIAL_TARGET} ---")
    
    # 1. Apertura Camera
    try:
        # Delta_t basso per reattività
        mv_iterator = EventsIterator(input_path=SERIAL_TARGET, delta_t=10000)
        print(f"[OK] Camera {SERIAL_TARGET} aperta.")
    except Exception as e:
        print(f"[ERRORE FATALE] Impossibile aprire la camera: {e}")
        return

    # 2. Reset Hardware (Forza Master / Free Run)
    try:
        # Questo comando dice alla camera: "Non aspettare nessuno, vai e basta!"
        mv_iterator.reader.device.get_i_camera_synchronization().set_mode_master()
        print("[OK] Modalità MASTER forzata (Free Run).")
    except Exception as e:
        print(f"[WARN] Impossibile settare Master mode: {e}")

    # 3. Ottieni risoluzione
    height, width = mv_iterator.get_size()
    print(f"Risoluzione: {width}x{height}")
    
    print(">>> AVVIO STREAMING... (Premi 'q' per uscire)")
    start_time = time.time()
    frame_count = 0
    
    try:
        for evs in mv_iterator:
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Conta eventi
            count = evs.size
            
            # Visualizzazione Semplice (Accumulo)
            im = np.zeros((height, width), dtype=np.uint8)
            if count > 0:
                # Mettiamo a bianco i pixel dove c'è un evento
                im[evs['y'], evs['x']] = 255
            
            # Creiamo un'immagine a colori per scriverci sopra
            im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            
            # Diagnostica a video
            if count == 0:
                status_text = "NESSUN EVENTO (MORTA?)"
                color = (0, 0, 255) # Rosso
            else:
                status_text = f"VIVA: {count} ev"
                color = (0, 255, 0) # Verde
                
            cv2.putText(im_color, f"T: {elapsed:.1f}s | {status_text}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow(f"Test {SERIAL_TARGET}", im_color)
            
            # Stampa su terminale ogni tanto
            if frame_count % 50 == 0:
                print(f"[{elapsed:.1f}s] Frame {frame_count}: {count} eventi")

            # Uscita
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"\n[CRASH] Lo script è morto durante il loop: {e}")
    
    cv2.destroyAllWindows()
    print("Test finito.")

if __name__ == "__main__":
    main()
