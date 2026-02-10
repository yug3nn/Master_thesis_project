from metavision_core.event_io import EventsIterator
import metavision_hal
import time

SERIAL_L = "genx320 10-003c"
SERIAL_R = "genx320 11-003c"

def main():
    print("--- STEREO TEST CON ROI V4 (LAST CHANCE) ---")

    try:
        # Apriamo le camere
        mv_L = EventsIterator(input_path=SERIAL_L, delta_t=10000)
        mv_R = EventsIterator(input_path=SERIAL_R, delta_t=10000)
        
        # DEFINIZIONE ROI (Centrale)
        x, y, w, h = 0, 60, 320, 200
        print(f"Tentativo di impostare ROI: {w}x{h} offset({x},{y})")
        
        for name, it in [("L", mv_L), ("R", mv_R)]:
            hal = it.reader.device
            i_roi = hal.get_i_roi()
            
            if i_roi:
                i_roi.enable(True)
                
                # --- TENTATIVO 1: Classe Annidata ---
                try:
                    # Spesso la classe Window è dentro I_ROI
                    rect = metavision_hal.I_ROI.Window(x, y, w, h)
                    i_roi.set_windows([rect])
                    print(f"[{name}] ROI attivata (Metodo 1: I_ROI.Window).")
                    continue
                except Exception as e1:
                    pass # Fallito, proviamo il prossimo

                # --- TENTATIVO 2: Lista di Tuple (Binding automatico) ---
                try:
                    # Alcuni binding C++ accettano direttamente la tupla
                    i_roi.set_windows([(x, y, w, h)])
                    print(f"[{name}] ROI attivata (Metodo 2: Tupla).")
                    continue
                except Exception as e2:
                    print(f"[{name}] ERRORE ROI: {e2}")
                    # Se fallisce la ROI, proviamo a continuare lo stesso per vedere se crasha
            else:
                print(f"[{name}] Hardware ROI non disponibile.")

        # Master Mode
        mv_L.reader.device.get_i_camera_synchronization().set_mode_master()
        mv_R.reader.device.get_i_camera_synchronization().set_mode_master()
        
    except Exception as e:
        print(f"Setup fallito: {e}")
        return

    print(">>> AVVIO STREAMING... Se arrivi a 15s, la tesi è salva.")
    start_time = time.time()
    
    try:
        for i, (evs_L, evs_R) in enumerate(zip(mv_L, mv_R)):
            if i % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] L: {evs_L.size} | R: {evs_R.size}")
                
                if evs_L.size == 0 and evs_R.size > 0:
                    print("\n!!! LA SINISTRA È MORTA (ROI NON HA FUNZIONATO) !!!")
                    #break
                
                if elapsed > 15:
                    print("\n*** SUCCESSO! IL SISTEMA REGGE! ***")
                    break
    except KeyboardInterrupt:
        print("Stop manuale.")

if __name__ == "__main__":
    main()
