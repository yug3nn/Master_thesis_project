import cv2
import numpy as np
from metavision_core.event_io import EventsIterator

# --- CONFIGURAZIONE ---
# I seriali che abbiamo confermato funzionare
SERIAL_MASTER = "genx320 11-003c"  # Proviamo questa come Master (CAM 0)
SERIAL_SLAVE  = "genx320 10-003c"  # Proviamo questa come Slave (CAM 1)

# Se le immagini risultano invertite (sinistra a destra), scambia questi nomi:
WINDOW_MASTER = "Master View (Left?)"
WINDOW_SLAVE  = "Slave View (Right?)"

def setup_sync(iterator, mode):
    """ Configura la modalità di sincronizzazione (Master/Slave) """
    try:
        device = iterator.reader.device
        i_sync = device.get_i_camera_synchronization()
    except AttributeError:
        print(f"[ERR] Impossibile accedere al modulo Sync per {mode}")
        return False

    if not i_sync:
        print(f"[ERR] {mode} non supporta la sincronizzazione!")
        return False

    print(f" -> Impostando {mode}...", end=" ")
    if mode == 'master':
        res = i_sync.set_mode_master()
    elif mode == 'slave':
        res = i_sync.set_mode_slave()
    
    print("SUCCESS" if res else "FAIL")
    return res

def events_to_image(evs, width, height):
    """ Converte eventi in immagine visibile (accumulo) """
    if evs.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    im = np.zeros((height, width), dtype=np.uint8)
    # Imposta a 255 i pixel dove è successo qualcosa
    im[evs['y'], evs['x']] = 255
    return im

def main():
    print("--- STEREO SYNC TEST ---")
    
    # 1. Apertura Camere (Iteratori)
    # Delta_t = 20000 us (20ms) è un buon frame rate (50fps)
    print("1. Apertura connessioni...")
    try:
        mv_master = EventsIterator(input_path=SERIAL_MASTER, delta_t=20000)
        mv_slave  = EventsIterator(input_path=SERIAL_SLAVE, delta_t=20000)
    except Exception as e:
        print(f"ERRORE APERTURA: {e}")
        print("SUGGERIMENTO: Esegui 'sudo pkill -9 python3' e riprova.")
        return

    height, width = mv_master.get_size()

    # 2. Configurazione Sync
    # ORDINE CRUCIALE: Prima lo Slave (si mette in ascolto), poi il Master (inizia a urlare)
    print("2. Configurazione Hardware Sync...")
    if not setup_sync(mv_slave, 'slave'):
        print("[WARN] Slave setup fallito. I timestamp potrebbero driftare.")
    
    if not setup_sync(mv_master, 'master'):
        print("[WARN] Master setup fallito.")

    print("\n>>> AVVIO STREAMING. Premi 'q' per uscire.")
    print(">>> IMPORTANTE: Muovi le mani davanti alle camere per vedere qualcosa!")

    for evs_M, evs_S in zip(mv_master, mv_slave):
        
        # --- Analisi Temporale ---
        ts_M = evs_M['t'][0] if evs_M.size > 0 else 0
        ts_S = evs_S['t'][0] if evs_S.size > 0 else 0
        
        # Calcolo differenza solo se entrambe vedono qualcosa
        diff_msg = "N/A"
        if ts_M > 0 and ts_S > 0:
            diff = abs(ts_M - ts_S)
            status = "SYNC OK" if diff < 200 else "SYNC BAD"
            diff_msg = f"{diff} us [{status}]"

        # --- Visualizzazione ---
        im_M = events_to_image(evs_M, width, height)
        im_S = events_to_image(evs_S, width, height)

        # Creiamo immagini a colori per scriverci sopra il testo
        vis_M = cv2.cvtColor(im_M, cv2.COLOR_GRAY2BGR)
        vis_S = cv2.cvtColor(im_S, cv2.COLOR_GRAY2BGR)

        # Info a video
        cv2.putText(vis_M, f"Master ({evs_M.size} ev)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_S, f"Slave ({evs_S.size} ev)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Uniamo le immagini
        combined = np.hstack((vis_M, vis_S))
        cv2.putText(combined, f"Time Diff: {diff_msg}", (width - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        cv2.imshow("Stereo View", combined)

        # Premi q per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Test completato.")

if __name__ == "__main__":
    main()
