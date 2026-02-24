import cv2
import numpy as np
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ SETTINGS ============
chessboard_size = (9, 6)   # inner corners (columns, rows)
square_size = 25.0         # mm (change to your real square size)
left_folder = "left/*.jpg"
right_folder = "right/*.jpg"
max_threads = 8
# ==================================

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

# Load image paths
left_images = sorted(glob.glob(left_folder))
right_images = sorted(glob.glob(right_folder))

assert len(left_images) == len(right_images), "Mismatch in image count!"

def process_pair(left_path, right_path):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        cornersL = cv2.cornerSubPix(
            grayL, cornersL, (11,11), (-1,-1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )
        cornersR = cv2.cornerSubPix(
            grayR, cornersR, (11,11), (-1,-1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        )
        return objp, cornersL, cornersR

    return None

# ================= MULTITHREADING =================
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [
        executor.submit(process_pair, l, r)
        for l, r in zip(left_images, right_images)
    ]

    for future in as_completed(futures):
        result = future.result()
        if result:
            objpoints.append(result[0])
            imgpoints_left.append(result[1])
            imgpoints_right.append(result[2])
# ==================================================

if len(objpoints) == 0:
    raise RuntimeError("No valid chessboard detections found!")

img_shape = cv2.imread(left_images[0]).shape[:2][::-1]

# Calibrate each camera individually
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, None, None)

retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape, None, None)

# Stereo calibration
criteria = (cv2.TermCriteria_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-5)

retStereo, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, \
R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtxL,
    distL,
    mtxR,
    distR,
    img_shape,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC
)

print("\nStereo Calibration RMS error:", retStereo)
print("\nRotation Matrix:\n", R)
print("\nTranslation Vector:\n", T)

# Save results
np.savez("stereo_calibration_data.npz",
         cameraMatrixL=cameraMatrixL,
         distCoeffsL=distCoeffsL,
         cameraMatrixR=cameraMatrixR,
         distCoeffsR=distCoeffsR,
         R=R, T=T, E=E, F=F)

print("\nCalibration data saved to stereo_calibration_data.npz")