import numpy as np
import cv2
import json
import os
import glob
import matplotlib.pyplot as plt
import argparse
import sys

# Adds the current working directory (project root) to sys.path
sys.path.append(os.getcwd())

from src.utils.logger_setup import setup_logging

def analyze_rms_hybrid_priority(objpoints_all, imgpoints_all, img_size, camera_side, logger):
    h, w = img_size
    stats_snapshots = []
    stats_rms = []
    stats_std = []
    
    best_rms = float('inf')
    best_results = {"mtx": None, "dist": None, "count": 0, "used": 0}

    # 1. SCORING PHASE: Hybrid selection (80% Edge / 20% Center)
    scored_data = []
    for objp, imgp in zip(objpoints_all, imgpoints_all):
        imgp_arr = np.array(imgp, dtype=np.float32).reshape(-1, 2)
        dist_x = np.minimum(imgp_arr[:, 0], w - imgp_arr[:, 0])
        dist_y = np.minimum(imgp_arr[:, 1], h - imgp_arr[:, 1])
        edge_score = np.min(np.minimum(dist_x, dist_y))
        scored_data.append({'obj': objp, 'img': imgp, 'score': edge_score})

    edge_prio = sorted(scored_data, key=lambda x: x['score'])
    center_prio = sorted(scored_data, key=lambda x: x['score'], reverse=True)

    hybrid_pool = []
    e_idx, c_idx = 0, 0
    while len(hybrid_pool) < len(scored_data):
        for _ in range(4):
            if e_idx < len(edge_prio):
                if edge_prio[e_idx] not in hybrid_pool: hybrid_pool.append(edge_prio[e_idx])
                e_idx += 1
        if c_idx < len(center_prio):
            if center_prio[c_idx] not in hybrid_pool: hybrid_pool.append(center_prio[c_idx])
            c_idx += 1
    
    # 2. BATCH PROCESSING
    flags = cv2.CALIB_FIX_K3
    OUTLIER_MULTIPLIER = 1.5

    logger.info(f"Processing hybrid pool for {camera_side} camera...")

    for count in range(10, len(hybrid_pool) + 1, 5):
        batch = hybrid_pool[:count]
        curr_obj = [np.array(x['obj'], dtype=np.float32) for x in batch]
        curr_img = [np.array(x['img'], dtype=np.float32) for x in batch]

        try:
            # Stage 1: Initial Calib
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(curr_obj, curr_img, img_size, None, None, flags=flags)

            # Stage 2: Refinement (Outlier Removal)
            errors = []
            for i in range(len(curr_obj)):
                p2, _ = cv2.projectPoints(curr_obj[i], rvecs[i], tvecs[i], mtx, dist)
                errors.append(cv2.norm(curr_img[i], p2, cv2.NORM_L2) / len(p2))

            median_err = np.median(errors)
            refined_indices = [i for i, err in enumerate(errors) if err <= median_err * OUTLIER_MULTIPLIER]
            ref_obj = [curr_obj[i] for i in refined_indices]
            ref_img = [curr_img[i] for i in refined_indices]

            ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(ref_obj, ref_img, img_size, None, None, flags=flags)

            # Metrics
            ref_errors = []
            for i in range(len(ref_obj)):
                p2, _ = cv2.projectPoints(ref_obj[i], rvecs_r[i], tvecs_r[i], mtx_r, dist_r)
                ref_errors.append(cv2.norm(ref_img[i], p2, cv2.NORM_L2) / len(p2))

            curr_std = np.std(ref_errors)
            stats_snapshots.append(count)
            stats_rms.append(ret_r)
            stats_std.append(curr_std)

            if ret_r < best_rms:
                best_rms = ret_r
                best_results = {"mtx": mtx_r, "dist": dist_r, "count": count, "used": len(ref_obj)}

            logger.info(f"Snap {count:03d}: RMS = {ret_r:.4f} ± {curr_std:.4f} ({len(ref_obj)}/{count} kept)")

        except cv2.error as e:
            logger.error(f"Error at count {count}: {e}")

    save_outputs(camera_side, best_results, stats_snapshots, stats_rms, stats_std, logger)

def save_outputs(side, results, snaps, rms, std, logger):
    # Save config for HW/Stereo use (Metavision compatible)
    os.makedirs("config", exist_ok=True)
    cfg_path = os.path.join("config", f"camera_{side}.json")
    cfg_data = {
        "type": "pinhole", "width": 320, "height": 320,
        "K": results['mtx'].flatten().tolist(),
        "D": results['dist'].flatten().tolist()
    }
    with open(cfg_path, 'w') as f:
        json.dump(cfg_data, f, indent=4)
    logger.info(f"Best parameters exported to: {cfg_path}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(snaps, rms, yerr=std, fmt='-o', color='indigo', ecolor='salmon', capsize=3)
    plt.title(f'RMS Progression - {side.upper()} (Hybrid Priority)')
    plt.grid(True, alpha=0.4)
    plot_path = os.path.join("data_analysis", f"rms_plot_{side}.png")
    plt.savefig(plot_path)
    logger.info(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=["left", "right"])
    args = parser.parse_args()
    logger = setup_logging("analyze_rms", args.side)

    data_dir = "data_analysis"
    files = glob.glob(os.path.join(data_dir, f"points_{args.side}_*.json"))
    if not files:
        logger.error(f"No point files found for {args.side}")
        sys.exit(1)

    latest = max(files, key=os.path.getmtime)
    with open(latest, 'r') as f:
        data = json.load(f)
    
    analyze_rms_hybrid_priority(data['objpoints'], data['imgpoints'], (320, 320), args.side, logger)
