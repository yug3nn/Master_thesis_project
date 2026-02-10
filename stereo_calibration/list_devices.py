from metavision_hal import DeviceDiscovery

def main():
    print("--- SCANSIONE DISPOSITIVI ---")
    serial_list = DeviceDiscovery.list()
    
    if len(serial_list) == 0:
        print("NESSUNA CAMERA RILEVATA!")
        return

    print(f"Trovate {len(serial_list)} telecamere:")
    for i, serial in enumerate(serial_list):
        print(f"  Camera {i+1}: {serial}")

if __name__ == "__main__":
    main()
