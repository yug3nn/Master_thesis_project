from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI
SERIAL_LEFT = "genx320 11-003c"
SERIAL_RIGHT = "genx320 10-003c"

def main():
    print("--- BLIND TEST (NO GUI) ---")
    
    # 1. Apriamo le camere
    try:
        mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=10000)
        mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=10000)
    except:
        print("Errore apertura camere (sono busy?)")
        return

    # 2. RESET FORZATO DELLO STATO (Cura per la Causa 3)
    # Impostiamo entrambe come MASTER per essere sicuri che non aspettino nessuno
    try:
        mv_L.reader.device.get_i_camera_synchronization().set_mode_master()
        mv_R.reader.device.get_i_camera_synchronization().set_mode_master()
        print("Reset stato camere a MASTER (Free Run)")
    except:
        pass

    print("Avvio conteggio (Premi CTRL+C per fermare)...")
    start_time = time.time()
    
    # Ciclo infinito SENZA imshow
    for i, (evs_L, evs_R) in enumerate(zip(mv_L, mv_R)):
        
        # Stampa solo ogni 50 cicli per non intasare la CPU
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Frame {i} -> L: {evs_L.size} ev | R: {evs_R.size} ev")
            
            # Se una muore, lo vediamo subito
            if evs_L.size == 0 and evs_R.size > 0:
                print("!!! LA SINISTRA È MORTA !!!")
                break

if __name__ == "__main__":
    main()
