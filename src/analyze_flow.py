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

    if df.empty:
        logger.warning("The CSV file is empty. No particles were tracked.")
        return

    # Setup the figure (2x2 grid)
    fig = plt.figure(figsize=(16, 12))

    # --- 1. 3D TRAJECTORIES PLOT ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    tracked_count = 0

    # Lists to collect data for the Vector Field
    vf_x, vf_y, vf_u, vf_v = [], [], [], []

    unique_ids = df['particle_id'].unique()
    
    for p_id in unique_ids:
        # Sort by timestamp to ensure correct chronological order
        p_data = df[df['particle_id'] == p_id].sort_values('timestamp_us')
        
        if len(p_data) >= min_track_length:
            x = p_data['x_m'].values
            z = p_data['z_m'].values 
            y = -p_data['y_m'].values # Invert Y for visual consistency
            
            # Plot 3D line
            ax1.plot(x, z, y, alpha=0.8, linewidth=1.5)
            ax1.scatter(x[-1], z[-1], y[-1], s=10, c='red') 
            tracked_count += 1
            
            # Calculate overall 2D velocity vector for this specific track
            dt_sec = (p_data['timestamp_us'].iloc[-1] - p_data['timestamp_us'].iloc[0]) / 1e6
            if dt_sec > 0:
                dx = p_data['x_m'].iloc[-1] - p_data['x_m'].iloc[0]
                dy = -p_data['y_m'].iloc[-1] - (-p_data['y_m'].iloc[0])
                
                vf_x.append(p_data['x_m'].iloc[0])
                vf_y.append(-p_data['y_m'].iloc[0])
                vf_u.append(dx / dt_sec)
                vf_v.append(dy / dt_sec)
            
    ax1.set_title(f"3D Particle Paths (Filtered: >{min_track_length} frames)")
    ax1.set_xlabel("X - Horizontal (m)")
    ax1.set_ylabel("Z - Depth (m)")
    ax1.set_zlabel("Y - Vertical (m)")
    
    logger.info(f"Visualized {tracked_count} stable trajectories.")

    # --- 2. VELOCITY HISTOGRAM ---
    ax2 = fig.add_subplot(2, 2, 2)
    valid_velocities = df[(df['vel_mps'] > 0.001) & (df['vel_mps'] < 5.0)]['vel_mps']
    
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

    # --- 3. 2D VECTOR FIELD (QUIVER) ---
    ax3 = fig.add_subplot(2, 2, 3)
    
    if vf_x:
        # Calculate speeds for colormapping
        speeds = np.hypot(vf_u, vf_v)
        
        # Plot the vector field
        q = ax3.quiver(vf_x, vf_y, vf_u, vf_v, speeds, cmap='jet', alpha=0.8, 
                       angles='xy', scale_units='xy', scale=None)
        fig.colorbar(q, ax=ax3, label='Speed (m/s)')
        
        ax3.set_title("2D Flow Vector Field (X-Y Plane)")
        ax3.set_xlabel("X - Horizontal (m)")
        ax3.set_ylabel("Y - Vertical (m)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # --- 4. MEAN GLOBAL FLOW DIRECTION (POLAR) ---
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        
        mean_u = np.mean(vf_u)
        mean_v = np.mean(vf_v)
        mean_speed = np.hypot(mean_u, mean_v)
        mean_angle = np.arctan2(mean_v, mean_u)
        
        # Draw the big mean direction arrow
        ax4.annotate('', xy=(mean_angle, mean_speed), xytext=(0, 0),
                     arrowprops=dict(facecolor='red', edgecolor='darkred', width=5, headwidth=15))
        
        # Configure the polar plot to look like a compass
        ax4.set_title("Mean Global Flow Direction")
        ax4.set_theta_zero_location('E') # 0 degrees is standard X-axis right
        ax4.set_theta_direction(1) # Counter-clockwise
        ax4.set_rmax(mean_speed * 1.5 if mean_speed > 0 else 1)
        ax4.set_rticks([mean_speed]) 
        ax4.set_rlabel_position(-22.5) 
        
        # Add a text label showing the exact angle and magnitude
        deg = np.degrees(mean_angle) % 360
        ax4.text(mean_angle, mean_speed * 1.7, f"{deg:.1f}°\n({mean_speed:.2f} m/s)", 
                 horizontalalignment='center', verticalalignment='center', 
                 weight='bold', color='red', fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No vector data", ha='center', va='center')
        ax3.set_title("2D Flow Vector Field")
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.text(0.5, 0.5, "No direction data", ha='center', va='center')
        ax4.set_title("Mean Global Flow Direction")

    # --- SAVE AND SHOW PLOT ---
    plt.tight_layout()
    output_img = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_img, dpi=150)
    logger.info(f"Analysis plot successfully saved to: {output_img}")

    logger.info("Opening interactive 3D plot. Close the window to exit.")
    plt.show()

def main():
    logger = setup_logging("analyze_flow", "optical_flow")
    
    parser = argparse.ArgumentParser(description="Visualize 3D Particle Flow from CSV")
    parser.add_argument('--csv', type=str, help="Path to the specific _tracked.csv file to analyze.")
    parser.add_argument('--min-frames', type=int, default=5, help="Minimum track length to display.")
    args = parser.parse_args()

    if args.csv:
        target_csv = args.csv
    else:
        logger.info("No CSV specified. Auto-detecting latest tracked data...")
        target_csv = get_latest_csv(logger)

    plot_analysis(target_csv, logger, args.min_frames)

if __name__ == "__main__":
    main()