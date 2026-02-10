from metavision_core.event_io import EventsIterator
import time

# I TUOI SERIALI
SERIAL_L = "genx320 10-003c"
SERIAL_R = "genx320 11-003c"

def reset_and_test(serial, label):
    print(f"--- RESET CONFIGURAZIONE: {label} ({serial}) ---")
    try:
        # Apriamo la camera
        mv = EventsIterator(input_path=serial, delta_t=10000)
        device = mv.reader.device
        
        # 1. FORZIAMO I BIAS A VALORI STANDARD (DEFAULT)
        # Per GenX320 i valori standard sono attorno a 80-100 per diff_on/off
        biases = device.get_i_ll_biases()
        if biases:
            # Impostiamo valori identici e sicuri
            valore_standard = 85
            biases.set("bias_diff_on", valore_standard)
            biases.set("bias_diff_off", valore_standard + 10) # Spesso OFF è leggermente più alto
            print(f"[{label}] Bias forzati a ON={valore_standard}, OFF={valore_standard+10}")
        else:
            print(f"[{label}] ERRORE: Impossibile accedere ai Bias!")

        # 2. Resettiamo Sync a Master (Free Run)
        device.get_i_camera_synchronization().set_mode_master()
        
        # 3. Test rapido di streaming
        print(f"[{label}] Avvio streaming di prova...")
        start = time.time()
        for i, evs in enumerate(mv):
            if i > 50: break # Testiamo solo mezzo secondo
            
        # Se arriviamo qui senza crashare, leggiamo quanti eventi ha l'ultimo pacchetto
        print(f"[{label}] Ultimo pacchetto: {evs.size} eventi. (Se è 0, è hardware).")
        
    except Exception as e:
        print(f"[{label}] CRASH DURANTE IL SETUP: {e}")

def main():
    # Testiamo una alla volta per essere sicuri che il comando arrivi
    reset_and_test(SERIAL_L, "SINISTRA (Quella sospetta)")
    print("-" * 30)
    reset_and_test(SERIAL_R, "DESTRA (Quella sana)")

if __name__ == "__main__":
    main()
