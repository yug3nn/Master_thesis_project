# src/utils/camera_streamer.py

import threading
import queue
from metavision_core.event_io import EventsIterator

class EventReaderThread(threading.Thread):
    """
    Universal multi-thread engine for Prophesee event camera acquisition.
    Automatically handles: Hardware Sync, Bias Tuning, Polarity Filtering, and Anti-Latency.
    """
    def __init__(self, serial, delta_t, role="STANDALONE", logger=None, 
                 max_queue_size=2, bias_increment=0, filter_polarity=None):
        super().__init__()
        self.serial = serial
        self.delta_t = delta_t
        self.role = role.upper()
        self.logger = logger
        self.bias_increment = bias_increment
        self.filter_polarity = filter_polarity  # 1 (positive), 0 (negative), None (all)
        
        self.q = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.error = False

    def run(self):
        self.running = True
        try:
            if self.logger:
                self.logger.info(f"[{self.role}] Initializing sensor on {self.serial}...")
            
            mv_it = EventsIterator(input_path=self.serial, delta_t=self.delta_t)
            device = mv_it.reader.device if hasattr(mv_it.reader, 'device') else mv_it.reader.get_device()

            # --- HARDWARE SYNC SETUP ---
            if "MASTER" in self.role:
                sync = device.get_i_camera_synchronization()
                if sync: 
                    sync.set_mode_master()
                if self.logger: 
                    self.logger.info(f"[{self.role}] Hardware Sync: MASTER")
            elif "SLAVE" in self.role:
                sync = device.get_i_camera_synchronization()
                if sync: 
                    sync.set_mode_slave()
                if self.logger: 
                    self.logger.info(f"[{self.role}] Hardware Sync: SLAVE")

            # --- BIAS SETUP ---
            if self.bias_increment > 0:
                biases = device.get_i_ll_biases()
                if biases:
                    biases.set("bias_diff_on", biases.get("bias_diff_on") + self.bias_increment)
                    biases.set("bias_diff_off", biases.get("bias_diff_off") + self.bias_increment)
                    if self.logger: 
                        self.logger.info(f"[{self.role}] Contrast increased (+{self.bias_increment})")

            # --- ACQUISITION LOOP ---
            for evs in mv_it:
                if not self.running:
                    break
                
                # Apply polarity filter if requested (e.g., 1 for bright edges only)
                if self.filter_polarity is not None:
                    evs = evs[evs['p'] == self.filter_polarity]

                # Anti-Latency Policy: if the queue is full, drop the oldest frame
                if self.q.full():
                    try: 
                        self.q.get_nowait()
                    except queue.Empty: 
                        pass
                
                # Insert tuple into the queue: (events, timestamp_microseconds)
                self.q.put((evs, mv_it.get_current_time()))
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"[{self.role}] Critical error: {e}")
            self.error = True
            
        if self.logger:
            self.logger.info(f"[{self.role}] Thread terminated cleanly.")

    def stop(self):
        """Stops the thread and empties the queue to unblock any waiting processes."""
        self.running = False
        while not self.q.empty():
            try: 
                self.q.get_nowait()
            except queue.Empty: 
                break