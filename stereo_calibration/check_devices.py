from metavision_hal import DeviceDiscovery

# Ottieni la lista dei dispositivi
# In questa versione dell'SDK, restituisce direttamente la lista dei seriali (es. ['00050512', '00050513'])
devices = DeviceDiscovery.list()

print(f"Dispositivi trovati: {len(devices)}")

for i, serial in enumerate(devices):
    print(f"--- Camera {i+1} ---")
    print(f"Serial: {serial}")
    # Nota: Poiché 'serial' è già una stringa, non possiamo chiedere altre info qui senza aprire la camera.
    # Ma per il tuo scopo (configurazione) basta questo!