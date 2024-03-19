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

# For png images
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





images = glob.glob('images/png_images/*.png')


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



img = cv.imread('images/png_images/calib8.png')

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
cv.imwrite('PngUndistortedCalib8Result.png', dst)


# or 2. Using <strong>remapping</strong>
# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('PngUndistortedRectifyMapcalib8Result.png', dst)




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
#  2.6607217469720355
# camera Matrix :
#  [[1.17639406e+03 0.00000000e+00 7.60068834e+02]
#  [0.00000000e+00 1.17606818e+03 6.11176374e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# distortaion Parameters :
#  [[-0.26760853  0.16331904 -0.00057991 -0.00343929 -0.08674731]]
# Rotation Vectors :
#  (array([[ 0.00893382],
#        [-0.01198409],
#        [-0.01744905]]), array([[-0.01257653],
#        [-0.00752526],
#        [ 0.00065422]]), array([[-0.62407887],
#        [ 0.0168631 ],
#        [-0.05739988]]), array([[0.00426732],
#        [0.42869574],
#        [0.00556661]]), array([[-4.66694487e-03],
#        [-1.56438004e-03],
#        [-1.58691757e+00]]), array([[-0.01678097],
#        [-0.00317141],
#        [-1.59347532]]), array([[-0.01224853],
#        [-0.01219885],
#        [ 0.038913  ]]), array([[ 0.00499976],
#        [-0.00738271],
#        [ 0.02203762]]), array([[ 0.33050469],
#        [-0.01012184],
#        [-0.01405995]]))
# Translation Vector :
#  (array([[-946.57798198],
#        [-460.1880871 ],
#        [1398.8216477 ]]), array([[-450.63203351],
#        [ 221.54749263],
#        [1406.66060396]]), array([[-214.16693532],
#        [-409.12711132],
#        [1386.26740889]]), array([[-808.79895077],
#        [-166.67029457],
#        [1389.34948055]]), array([[-428.56523918],
#        [ 413.34722145],
#        [1406.36056049]]), array([[-122.56901279],
#        [ 502.50328934],
#        [1407.64700619]]), array([[-951.82589706],
#        [ 179.39963912],
#        [1399.92370969]]), array([[-645.98272475],
#        [-466.7674689 ],
#        [1404.09985846]]), array([[-551.44385247],
#        [ 188.54253318],
#        [1285.73235446]]))
# total error: 0.09340562668449368








