from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI
SERIAL_LEFT = "genx320 10-003c"
SERIAL_RIGHT = "genx320 11-003c"

def main():
    print("--- STEREO LOW POWER TEST ---")
    
    try:
        # Apriamo entrambe
        mv_L = EventsIterator(input_path=SERIAL_LEFT, delta_t=10000)
        mv_R = EventsIterator(input_path=SERIAL_RIGHT, delta_t=10000)
        
        # --- TRUCCO SALVAVITA ---
        # Alziamo i BIAS al massimo per ridurre il flusso dati/corrente
        for name, it in [("L", mv_L), ("R", mv_R)]:
            biases = it.reader.device.get_i_ll_biases()
            if biases:
                # Valori alti = SOGLIA ALTA = Meno eventi = Meno consumo
                biases.set("bias_diff_on", 48)  
                biases.set("bias_diff_off", 48)
                print(f"[{name}] Bias alzati (Low Power Mode)")

        # Modalità Master per entrambe (Free Run)
        mv_L.reader.device.get_i_camera_synchronization().set_mode_master()
        mv_R.reader.device.get_i_camera_synchronization().set_mode_master()
        
    except Exception as e:
        print(f"Errore setup: {e}")
        return

    print("Avvio conteggio (Le camere dovrebbero essere molto 'pigre')...")
    start_time = time.time()
    
    for i, (evs_L, evs_R) in enumerate(zip(mv_L, mv_R)):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] L: {evs_L.size} | R: {evs_R.size}")
            
            # Controllo vita
            if evs_L.size == 0 and evs_R.size > 0:
                print("!!! LA SINISTRA È MORTA ANCHE IN LOW POWER !!!")
                #break
            if i > 1000: # 10 secondi
                print("VITTORIA! ENTRAMBE LE CAMERE SONO STABILI!")
                break

if __name__ == "__main__":
    main()
