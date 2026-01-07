import numpy as np
import cv2
import glob

BOARD_SIZE = (8, 6) 
SQUARE_SIZE_MM = 25.0 

# Path to your images
IMAGES_PATH = 'images/*.jpg'

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(IMAGES_PATH)
print(f"Found {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
        # Optional: Draw and verify
        # cv2.drawChessboardCorners(img, BOARD_SIZE, corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
    else:
        print(f"Skipping {fname} - corners not found")

cv2.destroyAllWindows()

print("Calibrating...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\n" + "="*40)
print(f"Reprojection Error: {ret:.4f} (Lower is better, aim for < 0.5)")
print("="*40)
print("\nCopy this into DetectionWorker.cpp:\n")

print(f"// Calibration Result (RMS Error: {ret:.4f})")
print("camera_matrix = (cv::Mat_<double>(3, 3) <<")
print(f"    {mtx[0,0]:.4f}, {mtx[0,1]:.4f}, {mtx[0,2]:.4f},")
print(f"    {mtx[1,0]:.4f}, {mtx[1,1]:.4f}, {mtx[1,2]:.4f},")
print(f"    {mtx[2,0]:.4f}, {mtx[2,1]:.4f}, {mtx[2,2]:.4f}")
print(");")
print("\ndist_coeffs = (cv::Mat_<double>(1, 5) <<")
print(f"    {dist[0,0]:.4f},")
print(f"    {dist[0,1]:.4f},")
print(f"    {dist[0,2]:.4f},")
print(f"    {dist[0,3]:.4f},")
print(f"    {dist[0,4]:.4f}")
print(");")