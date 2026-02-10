import numpy as np
import cv2
from metavision_core.event_io import EventsIterator
from metavision_sdk_ui import EventLoop, Window

def main():
    # Aumentiamo delta_t per dare respiro alla CPU (50ms)
    mv_it = EventsIterator(input_path="", delta_t=50000)
    height, width = mv_it.get_size()

    # Buffer dei timestamp: inizializzato a zero
    last_ts = np.zeros((height, width), dtype=np.int64)
    
    window = Window("Software Filter: Sinistra (Raw) | Destra (Filtrato)", width * 2, height, Window.RenderMode.BGR)

    # SOGLIA CRITICA: 
    # Se la differenza di tempo è MINORE di 10ms (10000us), è un riflesso veloce -> FILTRA
    # Se è MAGGIORE, è un oggetto lento -> TIENI
    min_threshold = 10000 

    for evs in mv_it:
        EventLoop.poll_and_dispatch()
        if window.should_close(): break
        
        if evs.size == 0: continue

        # --- LOGICA DI FILTRAGGIO ---
        # 1. Prendi i vecchi timestamp per questi eventi
        t_old = last_ts[evs['y'], evs['x']]
        
        # 2. Calcola il Delta T
        dt = evs['t'] - t_old
        
        # 3. Maschera: teniamo solo chi ha dt > soglia (movimenti lenti)
        mask = dt > min_threshold
        evs_filtered = evs[mask]
        
        # 4. Aggiorna i timestamp (solo dopo il filtro per evitare feedback)
        last_ts[evs['y'], evs['x']] = evs['t']

        # --- VISUALIZZAZIONE ---
        img_raw = np.zeros((height, width, 3), dtype=np.uint8)
        img_filt = np.zeros((height, width, 3), dtype=np.uint8)
        
        img_raw[evs['y'], evs['x']] = (255, 255, 255)
        if evs_filtered.size > 0:
            img_filt[evs_filtered['y'], evs_filtered['x']] = (0, 255, 0)

        display = np.hstack((img_raw, img_filt))
        window.show(display)

if __name__ == "__main__":
    main()