import os
import sys
import time
import cv2
import select
import numpy as np
import queue
from datetime import datetime

sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging
from src.utils.camera_streamer import EventReaderThread
from src.utils.settings import (
    TRACKER_DELTA_T, BIAS_INCREMENT_TRACKER, SERIAL_LEFT, SERIAL_RIGHT, RECORD_FOLDER, IMG_HEIGHT, IMG_WIDTH
)

def render_events(events, width=IMG_WIDTH, height=IMG_HEIGHT):
    """ Converte l'array di eventi in un'immagine visualizzabile da OpenCV """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if events is not None and len(events) > 0:
        # Disegna gli eventi in bianco su sfondo nero
        img[events['y'], events['x']] = 255
    return img

def main():
    if not os.path.exists(RECORD_FOLDER): 
        os.makedirs(RECORD_FOLDER)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_L = os.path.abspath(os.path.join(RECORD_FOLDER, f"left_{timestamp}.raw"))
    file_R = os.path.abspath(os.path.join(RECORD_FOLDER, f"right_{timestamp}.raw"))

    logger = setup_logging("record_stereo")
    logger.info("Starting multi-thread stereo recording...")

    # Passiamo il raw_file_path direttamente ai thread!
    t_L = EventReaderThread(SERIAL_LEFT, TRACKER_DELTA_T, role="SLAVE_LEFT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_TRACKER, filter_polarity=1,
                            raw_file_path=file_L)
                            
    t_R = EventReaderThread(SERIAL_RIGHT, TRACKER_DELTA_T, role="MASTER_RIGHT", logger=logger, 
                            bias_increment=BIAS_INCREMENT_TRACKER, filter_polarity=1,
                            raw_file_path=file_R)

    t_L.start()
    time.sleep(0.5) # Warmup per il clock di sync
    t_R.start()

    print("\n>>> RECORDING STARTED AND LIVE PREVIEW RUNNING <<<")
    print(f"Files saving to: {os.path.abspath(RECORD_FOLDER)}")
    print(">>> PRESS 'ENTER' IN THIS TERMINAL TO STOP AND SAVE...\n")
    
    try:
        # Loop non bloccante per visualizzare il video e aspettare l'INVIO
        while True:
            # Recupera gli eventi dalle code (se disponibili)
            try:
                evs_L, _ = t_L.q.get_nowait()
                evs_R, _ = t_R.q.get_nowait()
                cv2.imshow("Preview", np.hstack([render_events(evs_L), render_events(evs_R)]))
            except queue.Empty:
                pass
                
            try:
                evs_R, _ = t_R.q.get_nowait()
                cv2.imshow("Preview - MASTER RIGHT", render_events(evs_R))
            except queue.Empty:
                pass

            cv2.waitKey(10) # Necessario per aggiornare le finestre OpenCV

            # Controlla se l'utente ha premuto INVIO nel terminale (solo su Linux/Mac/Raspberry)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input() # Consuma il tasto premuto
                break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected.")
    finally:
        print("\n[CMD] Stop signal received. Saving and shutting down...")
        
        # Fermando i thread, verranno chiusi automaticamente anche i file RAW
        t_R.stop()
        t_L.stop()
        
        t_R.join()
        t_L.join()
        
        cv2.destroyAllWindows()
        logger.info("Shutdown complete. Files saved.")

if __name__ == "__main__":
    main()