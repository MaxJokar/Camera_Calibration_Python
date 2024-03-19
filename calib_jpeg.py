"""Test quality of different types of images using CameraCalibration  to see the result .
here there are 2 types of .png and .jpeg which the .png gives more accurate result in this survey .
As a result, we get more straight line relatively with two sides in the picture with good quality .
"""
import numpy as np
import cv2 as cv
import glob
import pickle
from matplotlib import pyplot as plt
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# 1. size of images are very importatnt for the small ones and big ones given below !

#  exmaple : For jpeg image
chessboardSize = (9,6)
frameSize = (640,480)




# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.






images = glob.glob('images/jpg_images/*.jpeg')

# images = glob.glob('C:\Users\PYTHON\Desktop\cam_calib\images\*.jpeg')
for image in images:
    # print(image)

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000) # milsec
        # cv.waitKey(5000) # milsec


cv.destroyAllWindows()


# ############## CALIBRATION #####################################################
# WE SPECIFY THE FRAME SIZE HERE : ALL WE NEED IS OBJECT POINT AND FRAME SIZE 
#  calib  .shape[::-1]
# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None, None)
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints,frameSize, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
# pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
# pickle.dump(dist, open( "dist.pkl", "wb" ))
print("==="*20)
print("RESULT - "*10)
print("Camera Calibrated :\n",ret)
print("camera Matrix :\n",cameraMatrix)
print("distortaion Parameters :\n",dist)
print("Rotation Vectors :\n",rvecs)
print("Translation Vector :\n",tvecs)

# print("reproje pixle  {:.4f}".format)

# ############## UNDISTORTION #####################################################

# img = cv.imread('images/calib1.jpeg')

img = cv.imread('images/jpg_images/calib6.jpeg')
h,  w = img.shape[:2] 
# getOptimalNewCameraMatrix:  to get more accurate result 
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


#  to compare two methods undistorion  as following :
# Undistort 1. Using cv.undistort()
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# result of calibration saves here 
# cv.imwrite('calibratedcalib4jpeg.jpeg', dst)
cv.imwrite('JpegCalibResult6Undistort.jpeg', dst)


# or 2. Using <strong>remapping</strong>
# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('JpegcaliResult6RectifyMap.jpeg', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
print("\n\n\n")


# ============================================================
# RESULT - RESULT - RESULT - RESULT - RESULT - RESULT - RESULT - RESULT - RESULT - RESULT -
# Camera Calibrated :
#  4.822480582894426
# camera Matrix :
#  [[207.98555715   0.         233.83363443]
#  [  0.         215.80447513 131.22614746]
#  [  0.           0.           1.        ]]
# distortaion Parameters :
#  [[-0.54220951  0.14590738 -0.02944134 -0.06591464 -0.01985249]]
# Rotation Vectors :
#  (array([[ 0.50189156],
#        [-0.24058469],
#        [ 0.08305786]]), array([[-0.18165508],
#        [-0.43868706],
#        [-0.18028031]]), array([[-0.27398524],
#        [-1.0570826 ],
#        [-0.21057734]]), array([[ 0.30344289],
#        [ 1.0952081 ],
#        [-2.75363674]]))
# Translation Vector :
#  (array([[-201.40680986],
#        [-131.04321349],
#        [ 221.84636408]]), array([[-251.42253234],
#        [ -88.94856003],
#        [ 160.46151499]]), array([[-136.43894572],
#        [ -93.58242911],
#        [ 136.73193896]]), array([[-64.92590337],
#        [ 26.69613923],
#        [264.61734897]]))
# total error: 0.6536647620094173







