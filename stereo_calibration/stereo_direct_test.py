from metavision_hal import DeviceDiscovery
from metavision_core.event_io import EventsIterator
import cv2
import numpy as np

def main():
    print("--- STEREO TEST (DIRECT DISCOVERY) ---")
    
    # 1. Scansiona i dispositivi disponibili
    dev_list = DeviceDiscovery.list()
    print(f"Dispositivi trovati: {len(dev_list)}")
    
    if len(dev_list) < 2:
        print("[ERRORE] Meno di 2 camere trovate! Controlla i cavi o i driver.")
        for d in dev_list: print(f" - Trovato: {d}")
        return

    # Stampiamo chi sono per capire l'ordine
    print(f"Cam A (Indice 0): {dev_list[0]}")
    print(f"Cam B (Indice 1): {dev_list[1]}")

    # 2. Apriamo usando l'INDICE (più sicuro delle stringhe a volte)
    # Nota: EventsIterator di solito vuole il seriale, ma proviamo a passargli 
    # direttamente l'oggetto wrapper o usiamo il seriale estratto dalla lista pulita.
    
    try:
        # Metodo A: Usiamo i seriali usciti dalla lista fresca
        serial_0 = dev_list[0]
        serial_1 = dev_list[1]
        
        print(f"Tentativo apertura Cam A: {serial_0}")
        mv_A = EventsIterator(input_path=serial_0, delta_t=20000)
        
        print(f"Tentativo apertura Cam B: {serial_1}")
        mv_B = EventsIterator(input_path=serial_1, delta_t=20000)
        
    except Exception as e:
        print(f"ERRORE CRITICO APERTURA: {e}")
        return

    # Se siamo arrivati qui, le camere sono aperte!
    print(">>> CAMERE APERTE! Streaming avviato...")
    
    height, width = mv_A.get_size()
    
    for evs_A, evs_B in zip(mv_A, mv_B):
        # Visualizzazione minimale per non intasare
        im_A = np.zeros((height, width), dtype=np.uint8)
        if evs_A.size > 0: im_A[evs_A['y'], evs_A['x']] = 255
        
        im_B = np.zeros((height, width), dtype=np.uint8)
        if evs_B.size > 0: im_B[evs_B['y'], evs_B['x']] = 255
        
        combined = np.hstack((im_A, im_B))
        cv2.imshow("Stereo Direct", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()