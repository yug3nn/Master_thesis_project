import os
import threading
from metavision_core.event_io import EventsIterator
from datetime import datetime

# --- CONFIGURATION ---
SERIAL_LEFT = "genx320 11-003c"   
SERIAL_RIGHT = "genx320 10-003c"  
DELTA_T = 20000                   
BIAS_INCREMENT = 30               
RECORD_DIR = "data"

def record_camera(serial, mode, name, file_path, stop_event):
    """ Funzione eseguita in un thread separato per ogni camera """
    try:
        print(f"Inizializzazione {name}...")
        mv = EventsIterator(input_path=serial, delta_t=DELTA_T)
        
        # Recupero Device e Configurazione HW
        dev = mv.reader.get_device() if hasattr(mv.reader, 'get_device') else mv.reader.device
        
        # Sincronizzazione Master/Slave
        i_sync = dev.get_i_camera_synchronization()
        if mode == 'master': i_sync.set_mode_master()
        else: i_sync.set_mode_slave()
        
        # Bias
        biases = dev.get_i_ll_biases()
        if biases:
            biases.set("bias_diff_on", biases.get("bias_diff_on") + BIAS_INCREMENT)
            biases.set("bias_diff_off", biases.get("bias_diff_off") + BIAS_INCREMENT)
            
        # Avvio Log RAW
        dev.get_i_events_stream().log_raw_data(file_path)
        print(f"[HW] {name} in registrazione su: {os.path.basename(file_path)}")

        # Loop di mantenimento finché non viene segnalato lo stop
        for _ in mv:
            if stop_event.is_set():
                break
        
        dev.get_i_events_stream().stop_log_raw_data()
        print(f"[OK] {name} salvata correttamente.")
        
    except Exception as e:
        print(f"[ERR] Errore su {name}: {e}")

def main():
    if not os.path.exists(RECORD_DIR): os.makedirs(RECORD_DIR)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    file_L = os.path.abspath(os.path.join(RECORD_DIR, f"left_{timestamp}.raw"))
    file_R = os.path.abspath(os.path.join(RECORD_DIR, f"right_{timestamp}.raw"))
    
    stop_event = threading.Event()

    # Creazione dei due Thread per indipendenza totale
    thread_L = threading.Thread(target=record_camera, args=(SERIAL_LEFT, 'slave', "LEFT", file_L, stop_event))
    thread_R = threading.Thread(target=record_camera, args=(SERIAL_RIGHT, 'master', "RIGHT", file_R, stop_event))

    thread_L.start()
    thread_R.start()

    print("\n>>> REGISTRAZIONE MULTI-THREAD AVVIATA <<<")
    print("Premi ENTER per fermare la registrazione...")
    
    try:
        input() # Aspetta l'invio dell'utente
    except KeyboardInterrupt:
        pass
    finally:
        print("\nArresto in corso...")
        stop_event.set()
        thread_L.join()
        thread_R.join()
        print("Tutti i file sono stati chiusi.")

if __name__ == "__main__":
    main()
