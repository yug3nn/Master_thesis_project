import pandas as pd
import matplotlib
# Use 'TkAgg' for interactive GUI display (crucial for VNC and 3D rotation)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import glob
import argparse

# Add the project root to sys.path
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import RECORD_FOLDER, DATA_FOLDER

# The tracker now saves to the same folder as the raw files. 
# We'll check both DATA_FOLDER and RECORD_FOLDER just in case.
SEARCH_FOLDERS = [DATA_FOLDER, RECORD_FOLDER]

def get_latest_csv(logger):
    """Finds the most recent _tracked.csv file."""
    csv_files = []
    for folder in SEARCH_FOLDERS:
        csv_files.extend(glob.glob(os.path.join(folder, "*_tracked.csv")))
        
    if not csv_files:
        logger.error("No '*_tracked.csv' files found. Run the offline tracker first.")
        sys.exit(1)
        
    latest_csv = max(csv_files, key=os.path.getctime)
    return latest_csv

def plot_analysis(csv_file, logger, min_track_length=5):
    logger.info(f"Loading data from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # Basic data validation
    if df.empty:
        logger.warning("The CSV file is empty. No particles were tracked.")
        return

    # Setup the figure
    fig = plt.figure(figsize=(16, 7))

    # --- 1. 3D TRAJECTORIES PLOT ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    tracked_count = 0

    # Get unique particle IDs
    unique_ids = df['particle_id'].unique()
    
    for p_id in unique_ids:
        p_data = df[df['particle_id'] == p_id]
        
        # Filter out "ghost" particles that only appeared for a few frames
        if len(p_data) >= min_track_length:
            # OpenCV coordinates: X is right, Y is down, Z is forward (depth)
            # For 3D plotting, mapping Z to depth, X to width, and Y to height (inverted)
            x = p_data['x_m'].values
            z = p_data['z_m'].values 
            y = -p_data['y_m'].values # Negative to make 'up' visually correct
            
            ax1.plot(x, z, y, alpha=0.8, linewidth=1.5)
            # Add a small dot at the end of the trajectory to show direction
            ax1.scatter(x[-1], z[-1], y[-1], s=10, c='red') 
            tracked_count += 1
    
    ax1.set_title(f"3D Particle Paths (Filtered: >{min_track_length} frames)")
    ax1.set_xlabel("X - Horizontal (m)")
    ax1.set_ylabel("Z - Depth (m)")
    ax1.set_zlabel("Y - Vertical (m)")
    
    logger.info(f"Visualized {tracked_count} stable trajectories out of {len(unique_ids)} total IDs.")

    # --- 2. VELOCITY HISTOGRAM ---
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Filter out absolute zero velocities (mostly newly spawned particles)
    valid_velocities = df[df['vel_mps'] > 0.001]['vel_mps']
    
    if not valid_velocities.empty:
        mean_v = valid_velocities.mean()
        median_v = valid_velocities.median()
        
        ax2.hist(valid_velocities, bins=50, color='royalblue', edgecolor='black', alpha=0.7)
        ax2.axvline(mean_v, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_v:.3f} m/s')
        ax2.axvline(median_v, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median_v:.3f} m/s')
        
        ax2.set_title("Particle Velocity Distribution")
        ax2.set_xlabel("Velocity (m/s)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No valid velocity data", ha='center', va='center')
        ax2.set_title("Particle Velocity Distribution")

    # --- SAVE PLOT ---
    plt.tight_layout()
    output_img = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_img, dpi=150)
    logger.info(f"Analysis plot successfully saved to: {output_img}")

    # --- SHOW PLOT INTERACTIVELY ---
    logger.info("Opening interactive 3D plot. Close the window to exit.")
    plt.show()

def main():
    logger = setup_logging("analyze_flow", "optical_flow")
    
    parser = argparse.ArgumentParser(description="Visualize 3D Particle Flow from CSV")
    parser.add_argument('--csv', type=str, help="Path to the specific _tracked.csv file to analyze.")
    parser.add_argument('--min-frames', type=int, default=5, help="Minimum track length to display (filters noise).")
    args = parser.parse_args()

    # Determine input file
    if args.csv:
        target_csv = args.csv
    else:
        logger.info("No CSV specified. Auto-detecting latest tracked data...")
        target_csv = get_latest_csv(logger)

    plot_analysis(target_csv, logger, args.min_frames)

if __name__ == "__main__":
    main()