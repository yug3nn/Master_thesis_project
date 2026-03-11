import json
import numpy as np
import cv2
import matplotlib
# Use 'TkAgg' for interactive GUI display (crucial for VNC and 3D rotation)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob
import sys

# Add the current working directory (project root) to sys.path for custom modules
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import DATA_FOLDER, SQUARE_SIZE_MM

def get_latest_json():
    """Retrieves the latest raw dataset to extract image points."""
    list_of_files = glob.glob(os.path.join(DATA_FOLDER, 'points_stereo_*.json'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def generate_validation_plots(logger):
    logger.info("--- 3D RECONSTRUCTION VALIDATION (FINAL MODEL) ---")
    
    json_path = get_latest_json()
    if not json_path:
        logger.error(f"No data file found in {DATA_FOLDER}")
        return

    # 1. Load raw image points from the dataset
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_pts_L = [np.array(i).astype(np.float32) for i in data["imgpoints_L"]]
    img_pts_R = [np.array(i).astype(np.float32) for i in data["imgpoints_R"]]
    bw, bh = data.get("board_size", (9, 6))

    # 2. Load the DEFINITIVE calibration parameters
    try:
        with open("config/camera_left.json", 'r') as f:
            camL = json.load(f)
            mtxL = np.array(camL["K"])
            distL = np.array(camL["D"])
            
        with open("config/camera_right.json", 'r') as f:
            camR = json.load(f)
            mtxR = np.array(camR["K"])
            distR = np.array(camR["D"])
            
        with open("config/stereo_params.json", 'r') as f:
            stereo = json.load(f)
            R = np.array(stereo["R"])
            T = np.array(stereo["T"])
    except FileNotFoundError as e:
        logger.error(f"Missing configuration file: {e}")
        return

    # Projection matrices (Normalized Coordinates)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))

    errors = []
    positions = []

    logger.info("Triangulating all points using the final model...")

    for i in range(len(img_pts_L)):
        pL_u = cv2.undistortPoints(img_pts_L[i].reshape(-1, 1, 2), mtxL, distL)
        pR_u = cv2.undistortPoints(img_pts_R[i].reshape(-1, 1, 2), mtxR, distR)
        
        pts4D = cv2.triangulatePoints(P1, P2, pL_u, pR_u)
        pts3D = (pts4D[:3, :] / pts4D[3, :]).T
        
        # Auto-Detect array orientation to prevent diagonal distance calculation
        grid_test = pts3D.reshape(bh, bw, 3)
        if np.linalg.norm(grid_test[0,0] - grid_test[1,0]) > SQUARE_SIZE_MM * 1.5:
            grid3D = pts3D.reshape(bw, bh, 3)
            gridImg = img_pts_L[i].reshape(bw, bh, 2)
            rows, cols = bw, bh
        else:
            grid3D = grid_test
            gridImg = img_pts_L[i].reshape(bh, bw, 2)
            rows, cols = bh, bw

        # Calculate real distances and save pixel positions for the heatmap
        for r in range(rows):
            for c in range(cols):
                if c < cols - 1: # Horizontal edge
                    dist = np.linalg.norm(grid3D[r, c] - grid3D[r, c+1])
                    errors.append(abs(dist - SQUARE_SIZE_MM))
                    positions.append((gridImg[r, c] + gridImg[r, c+1]) / 2)
                if r < rows - 1: # Vertical edge
                    dist = np.linalg.norm(grid3D[r, c] - grid3D[r+1, c])
                    errors.append(abs(dist - SQUARE_SIZE_MM))
                    positions.append((gridImg[r, c] + gridImg[r+1, c]) / 2)

    errors_mm = np.array(errors)
    positions = np.array(positions)

    # --- STATISTICS ---
    mae = np.mean(errors_mm)
    median = np.median(errors_mm)
    rmse = np.sqrt(np.mean(errors_mm**2))
    std = np.std(errors_mm)

    logger.info(f"--- FINAL 3D METRICS ({len(img_pts_L)} snapshots evaluated) ---")
    logger.info(f"Mean Absolute Error (MAE): {mae:.4f} mm")
    logger.info(f"Median Error:              {median:.4f} mm")
    logger.info(f"RMSE:                      {rmse:.4f} mm")
    logger.info(f"Standard Deviation:        {std:.4f} mm")
    logger.info(f"Max Error detected:        {np.max(errors_mm):.4f} mm")
    logger.info("---------------------------------------------------------")

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Error Histogram
    ax1.hist(errors_mm, bins=40, color='mediumaquamarine', edgecolor='black', alpha=0.8)
    ax1.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}mm')
    ax1.axvline(mae, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mae:.2f}mm')
    ax1.set_title("3D Reconstruction Error Distribution")
    ax1.set_xlabel("Absolute Error (mm)")
    ax1.set_ylabel("Frequency (Number of edges)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Spatial Heatmap on the Sensor
    im = ax2.scatter(positions[:, 0], positions[:, 1], c=errors_mm, 
                     cmap='turbo', s=20, alpha=0.7, edgecolor='none')
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Metric Error (mm)')
    ax2.set_title("Spatial Error Distribution (Left Sensor FOV)")
    ax2.set_xlim(0, 320)
    ax2.set_ylim(320, 0) # Inverted Y-axis because image origin is top-left, set to 320 for GenX320
    ax2.set_xlabel("Pixel X")
    ax2.set_ylabel("Pixel Y")
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(DATA_FOLDER, "final_reconstruction_heatmap.png")
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Initialize logger for final validation
    logger = setup_logging("check_error", "stereo")
    generate_validation_plots(logger)
