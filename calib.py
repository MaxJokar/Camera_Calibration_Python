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
#  size of images are very importatnt for the small ones and big ones given below !

# chessboardSize = (9,6)
# frameSize = (640,480)
chessboardSize = (24,17)
frameSize = (1440,1080)


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




# images = glob.glob('images/*.jpeg')
images = glob.glob('images/*.png')

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
img = cv.imread('images/calib8.png')
h,  w = img.shape[:2] 
# getOptimalNewCameraMatrix:  to get more accurate result 
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort 1. Using <strong>cv.undistort()</strong>
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
# result of calibration saves here 
# cv.imwrite('calibratedcalib4jpeg.jpeg', dst)
cv.imwrite('calibResult8.png', dst)


# or 2. Using <strong>remapping</strong>
# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult81.png', dst)




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
#  1.2915046165333723
# camera Matrix :
#  [[1.08447791e+03 0.00000000e+00 7.40007822e+02]
#  [0.00000000e+00 1.08523372e+03 5.72964984e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# distortaion Parameters :
#  [[-0.20874147  0.11251551  0.00262093 -0.00103003 -0.05798302]]
# Rotation Vectors :
#  (array([[-0.0101702 ],
#        [ 0.00950659],
#        [ 0.03641409]]), array([[-0.01767589],
#        [ 0.00649912],
#        [ 0.02003055]]), array([[ 0.30085187],
#        [ 0.00654742],
#        [-0.0192971 ]]), array([[-0.00790216],
#        [ 0.00468764],
#        [-0.00076035]]), array([[-0.0164335 ],
#        [ 0.00331607],
#        [-1.5881517 ]]), array([[-1.72092564e-02],
#        [ 1.58392238e-03],
#        [-1.59404724e+00]]))
# Translation Vector :
#  (array([[-928.41319407],
#        [ 227.28980417],
#        [1305.55520648]]), array([[-622.39682228],
#        [-419.6633396 ],
#        [1309.62704725]]), array([[-529.50686257],
#        [ 231.44934237],
#        [1191.37580598]]), array([[-426.60551171],
#        [ 267.86028019],
#        [1300.06676781]]), array([[-404.27294283],
#        [ 459.56273976],
#        [1299.24929065]]), array([[ -98.23489831],
#        [ 548.38187192],
#        [1297.80596571]]))
# total error: 0.04418011714162698









