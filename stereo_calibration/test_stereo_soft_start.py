from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI
SERIAL_L = "genx320 11-003c"
SERIAL_R = "genx320 10-003c"

def open_camera(serial, delay=0):
    if delay > 0:
        print(f"[{serial}] Aspetto {delay}s prima di avviare...")
        time.sleep(delay)
    
    print(f"[{serial}] Tentativo di accensione...")
    try:
        # Delta t standard
        mv = EventsIterator(input_path=serial, delta_t=10000)
        # Forza Master
        mv.reader.device.get_i_camera_synchronization().set_mode_master()
        print(f"[{serial}] ACCESA CON SUCCESSO!")
        return mv
    except Exception as e:
        print(f"[{serial}] ERRORE AVVIO: {e}")
        return None

def main():
    print("--- STEREO SOFT START TEST V2 (FIXED) ---")
    
    # 1. Avvia la Sinistra
    mv_L = open_camera(SERIAL_L, delay=0)
    if mv_L is None: return

    # 2. Aspetta che il picco di corrente passi
    print("... Stabilizzazione tensione (2 secondi) ...")
    time.sleep(2.0)
    
    # 3. Avvia la Destra
    mv_R = open_camera(SERIAL_R, delay=0)
    if mv_R is None: return

    print(">>> ENTRAMBE ACCESE. Inizio loop...")
    start_time = time.time()
    
    # --- CORREZIONE QUI ---
    # Trasformiamo gli oggetti in iteratori Python veri
    it_L = iter(mv_L)
    it_R = iter(mv_R)
    
    try:
        while True:
            # Usiamo next() sugli iteratori
            evs_L = next(it_L)
            evs_R = next(it_R)
            
            elapsed = time.time() - start_time
            
            # Stampa ogni 0.5s circa (ogni 50 cicli se delta_t=10000us)
            if int(elapsed * 100) % 50 == 0:
                print(f"[{elapsed:.1f}s] L: {evs_L.size} | R: {evs_R.size}")
                
                # CHECK DI VITA
                if evs_L.size == 0 and evs_R.size > 0:
                    print("!!! LA SINISTRA È MORTA !!!")
                    break
                if evs_L.size > 0 and evs_R.size == 0:
                    print("!!! LA DESTRA È MORTA !!!")
                    break
                
                # Se regge 15 secondi, è fatta
                if elapsed > 15.0:
                    print("\n*** SUCCESSO TOTALE! ***")
                    print("Il sistema è stabile con avvio sequenziale.")
                    break

    except StopIteration:
        print("Fine dello stream.")
    except KeyboardInterrupt:
        print("Stop manuale.")
    except Exception as e:
        print(f"Crash generico: {e}")

if __name__ == "__main__":
    main()
