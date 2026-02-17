import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Cruciale per VNC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

def plot_analysis(csv_file):
    try:
        df = pd.read_csv(csv_file)
        # Pulizia dati NumPy
        for col in ['x_m', 'y_m', 'z_m', 'vel_mps']:
            if df[col].dtype == object:
                df[col] = df[col].str.replace('[', '', regex=False).str.replace(']', '', regex=False).astype(float)
    except Exception as e:
        print(f"Errore: {e}")
        return

    fig = plt.figure(figsize=(14, 7))

    # --- 1. TRAIETTORIE 3D (FILTRATE) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    min_points = 10  # <--- MODIFICA QUESTO: ignora i "flash" brevi
    tracked_count = 0

    for p_id in df['particle_id'].unique():
        p_data = df[df['particle_id'] == p_id]
        
        # Disegniamo la linea solo se la particella è "stabile"
        if len(p_data) >= min_points:
            ax1.plot(p_data['x_m'], p_data['z_m'], p_data['y_m'], alpha=0.8, linewidth=1.5)
            tracked_count += 1
    
    ax1.set_title(f"3D Paths (Filtered: >{min_points} points)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Depth Z (m)")
    ax1.set_zlabel("Y (m)")
    print(f"Visualizzate {tracked_count} traiettorie significative su {len(df['particle_id'].unique())} totali.")

    # --- 2. ISTOGRAMMA VELOCITÀ ---
    ax2 = fig.add_subplot(1, 2, 2)
    # Filtriamo le velocità assurde o nulle
    valid_vel = df[(df['vel_mps'] > 0.05) & (df['vel_mps'] < 10.0)]['vel_mps']
    ax2.hist(valid_vel, bins=30, color='royalblue', edgecolor='white')
    ax2.set_title("Flow Velocity Distribution")
    ax2.set_xlabel("Speed (m/s)")

    plt.tight_layout()
    output_img = csv_file.replace('.csv', '_clean.png')
    plt.savefig(output_img)
    plt.show()
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_analysis(sys.argv[1])
    else:
        print("Uso: python3 analyze_flow.py <file.csv>")
