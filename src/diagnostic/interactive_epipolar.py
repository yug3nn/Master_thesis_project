import json
import numpy as np
import cv2
import os
import glob
import sys

# Add the project root to sys.path to load the logger
sys.path.append(os.getcwd())
from src.utils.logger_setup import setup_logging
from src.utils.settings import DATA_FOLDER, IMG_WIDTH, IMG_HEIGHT

def get_latest_json():
    """Retrieves the latest raw dataset to extract image points."""
    files = glob.glob(os.path.join(DATA_FOLDER, 'points_stereo_*.json'))
    return max(files, key=os.path.getctime) if files else None

def draw_epipolar_line(img, line, color):
    """Draws a line given its equation ax + by + c = 0"""
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [IMG_WIDTH, -(line[2] + line[0] * IMG_WIDTH) / line[1]])
    cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

def point_to_line_distance(point, line):
    """Calculates the orthogonal distance from a point to a line."""
    return abs(line[0]*point[0] + line[1]*point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)

class EpipolarInteractiveViewer:
    def __init__(self, logger):
        self.logger = logger
        self.json_path = get_latest_json()
        
        if not self.json_path:
            self.logger.error("No dataset found.")
            sys.exit(1)
            
        self.load_data()
        self.current_snap_idx = 0  # Start with the first snapshot
        self.selected_point_idx = -1
        
        # UI Windows
        self.win_L = "LEFT SENSOR (Click a point)"
        self.win_R = "RIGHT SENSOR (Epipolar Line)"
        cv2.namedWindow(self.win_L)
        cv2.namedWindow(self.win_R)
        cv2.setMouseCallback(self.win_L, self.on_mouse_click)

    def load_data(self):
        self.logger.info(f"Loading points from: {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        self.img_pts_L = [np.array(i).astype(np.float32).reshape(-1, 2) for i in data["imgpoints_L"]]
        self.img_pts_R = [np.array(i).astype(np.float32).reshape(-1, 2) for i in data["imgpoints_R"]]
        
        stereo_cfg = "config/stereo_params.json"
        self.logger.info(f"Loading Fundamental Matrix from: {stereo_cfg}")
        with open(stereo_cfg, 'r') as f:
            stereo = json.load(f)
            self.F = np.array(stereo["F"])

    def on_mouse_click(self, event, x, y, flags, param):
        """Finds the closest point to the mouse click in the Left view."""
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_L = self.img_pts_L[self.current_snap_idx]
            # Calculate distances to all points
            distances = np.linalg.norm(pts_L - np.array([x, y]), axis=1)
            closest_idx = np.argmin(distances)
            
            # If the click is reasonably close to a point
            if distances[closest_idx] < 20: 
                self.selected_point_idx = closest_idx
                self.update_views()

    def update_views(self):
        """Redraws both canvases based on the current selection."""
        # Create blank black canvases
        img_L = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        img_R = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        pts_L = self.img_pts_L[self.current_snap_idx]
        pts_R = self.img_pts_R[self.current_snap_idx]
        
        # Draw all points as small dim dots
        for ptL, ptR in zip(pts_L, pts_R):
            cv2.circle(img_L, tuple(ptL.astype(int)), 2, (100, 100, 100), -1)
            cv2.circle(img_R, tuple(ptR.astype(int)), 2, (100, 100, 100), -1)

        # Draw UI Text
        cv2.putText(img_L, f"Snap: {self.current_snap_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_L, "Press 'n' for next snap", (10, IMG_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # If a point is selected, draw the epipolar magic
        if self.selected_point_idx != -1:
            pt_L = pts_L[self.selected_point_idx]
            pt_R = pts_R[self.selected_point_idx]
            
            # Highlight selected points
            cv2.circle(img_L, tuple(pt_L.astype(int)), 4, (0, 255, 0), -1)
            cv2.circle(img_R, tuple(pt_R.astype(int)), 4, (0, 0, 255), -1)
            
            # Calculate Epipolar Line in Right image for the point in Left image
            # cv2.computeCorrespondEpilines takes: points, which_image (1=Left, 2=Right), F
            pt_L_reshaped = pt_L.reshape(-1, 1, 2)
            line_R = cv2.computeCorrespondEpilines(pt_L_reshaped, 1, self.F)[0].flatten()
            
            # Draw the line
            draw_epipolar_line(img_R, line_R, (255, 0, 0)) # Blue line
            
            # Calculate exact pixel distance (Epipolar Error)
            err = point_to_line_distance(pt_R, line_R)
            self.logger.info(f"Point {self.selected_point_idx:02d} | Epipolar Error: {err:.4f} pixels")
            
            # Display error on screen
            cv2.putText(img_R, f"Error: {err:.3f} px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(self.win_L, img_L)
        cv2.imshow(self.win_R, img_R)

    def run(self):
        self.logger.info("Starting Interactive Epipolar Viewer.")
        self.update_views()
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                break
            elif key == ord('n'): # Next snapshot
                self.current_snap_idx = (self.current_snap_idx + 1) % len(self.img_pts_L)
                self.selected_point_idx = -1
                self.logger.info(f"Switched to snapshot {self.current_snap_idx}")
                self.update_views()
                
        cv2.destroyAllWindows()
        self.logger.info("Viewer closed.")

if __name__ == "__main__":
    logger = setup_logging("interactive_epipolar", "stereo")
    viewer = EpipolarInteractiveViewer(logger)
    viewer.run()
