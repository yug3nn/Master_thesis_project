from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI
# Usiamo la camera "sana" (Destra) come MASTER per dare un clock stabile
# Usiamo la camera "fragile" (Sinistra) come SLAVE
SERIAL_SLAVE = "genx320 10-003c"  # Sinistra
SERIAL_MASTER = "genx320 11-003c" # Destra

def main():
    print("--- STEREO SYNC: MASTER/SLAVE SOFT START ---")
    print(f"MASTER: {SERIAL_MASTER} | SLAVE: {SERIAL_SLAVE}")
    
    try:
        # 1. AVVIA PRIMA LA SLAVE
        # La Slave deve essere pronta ad ascoltare quando il Master inizia a parlare.
        print(f"1. Inizializzo SLAVE ({SERIAL_SLAVE})...")
        mv_slave = EventsIterator(input_path=SERIAL_SLAVE, delta_t=10000)
        
        # Imposta modalità SLAVE
        # La camera ora aspetta il segnale elettrico sul cavetto per sincronizzare i timestamp
        sync_slave = mv_slave.reader.device.get_i_camera_synchronization()
        sync_slave.set_mode_slave()
        print("   -> Slave impostata (In attesa del clock...)")
        
        # 2. PAUSA TATTICA (SOFT START)
        # Serve a stabilizzare la tensione dopo l'accensione della prima camera
        print("... Attesa stabilizzazione (2 secondi) ...")
        time.sleep(2.0)
        
        # 3. AVVIA LA MASTER
        print(f"2. Inizializzo MASTER ({SERIAL_MASTER})...")
        mv_master = EventsIterator(input_path=SERIAL_MASTER, delta_t=10000)
        
        # Imposta modalità MASTER
        # Appena fatto questo, la Master inizia a mandare impulsi di clock alla Slave
        sync_master = mv_master.reader.device.get_i_camera_synchronization()
        sync_master.set_mode_master()
        print("   -> Master impostata (Clock avviato!)")

    except Exception as e:
        print(f"ERRORE CRITICO AVVIO: {e}")
        return

    print("\n>>> SISTEMA SINCRONIZZATO AVVIATO. Inizio loop...")
    start_time = time.time()
    
    try:
        # Usiamo zip per leggere le coppie di frame sincronizzati
        for i, (evs_S, evs_M) in enumerate(zip(mv_slave, mv_master)):
            
            # Stampa ogni ~0.5 secondi
            if i % 50 == 0:
                elapsed = time.time() - start_time
                
                # Nota: evs_S è Slave (Sinistra), evs_M è Master (Destra)
                print(f"[{elapsed:.1f}s] Slave(L): {evs_S.size} | Master(R): {evs_M.size}")
                
                # CONTROLLO MORTE
                if evs_S.size == 0 and evs_M.size > 0:
                    print("!!! LA SLAVE (Sinistra) È MORTA !!!")
                    print("Possibile causa: Cavo sync scollegato o calo di tensione.")
                    break
                
                if elapsed > 15:
                    print("\n*** SUCCESSO! STABILE IN MASTER/SLAVE ***")
                    break
                    
    except KeyboardInterrupt:
        print("Stop manuale.")
    except Exception as e:
        print(f"Crash durante lo stream: {e}")

if __name__ == "__main__":
    main()
