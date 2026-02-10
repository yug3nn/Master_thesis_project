from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI (Controlla che siano giusti!)
SERIAL_LEFT = "genx320 11-003c"  
SERIAL_RIGHT = "genx320 10-003c"

def test_camera(serial_number, name):
    print(f"--- TESTING CAMERA: {name} ({serial_number}) ---")
    try:
        # Proviamo ad aprire la camera SENZA sync, in modalità raw
        iterator = EventsIterator(input_path=serial_number, delta_t=100000)
        print(f" [OK] Camera {name} aperta con successo!")
        
        # Leggiamo qualche evento per essere sicuri
        for i, evs in enumerate(iterator):
            if evs.size > 0:
                print(f" [OK] Ricevuti {evs.size} eventi. Timestamp: {evs['t'][0]}")
                break
            if i > 10:
                print(" [WARN] Camera aperta ma NESSUN evento (è buio o tappo messo?)")
                break
        
        # IMPORTANTE: Distruggere l'oggetto rilascia la risorsa
        del iterator
        print(f" [OK] Camera {name} rilasciata.\n")
        return True
        
    except Exception as e:
        print(f" [FAIL] Errore aprendo {name}: {e}\n")
        return False

if __name__ == "__main__":
    # Testiamo prima una...
    res_L = test_camera(SERIAL_LEFT, "LEFT")
    
    # ...attendiamo un attimo per sicurezza...
    time.sleep(1)
    
    # ...poi l'altra.
    res_R = test_camera(SERIAL_RIGHT, "RIGHT")

    if res_L and res_R:
        print(">>> TUTTO OK: Entrambe le camere funzionano singolarmente via Python.")
    else:
        print(">>> PROBLEMA: Una o entrambe le camere sono bloccate o irraggiungibili.")
