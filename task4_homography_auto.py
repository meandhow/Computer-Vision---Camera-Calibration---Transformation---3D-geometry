#!/usr/bin/env python

import cv2
import numpy as np

# if __name__ == '__main__' :

#     # Read source image.
#     im_src = cv2.imread('HG1.jpeg')
#     # Four corners of the book in source image
#     pts_src = np.array([[2282, 584], [2900, 1611],[1944,2729],[1360, 932]])


#     # Read destination image.
#     im_dst = cv2.imread('HG2.jpeg')
#     # Four corners of the book in destination image.
#     pts_dst = np.array([[2522, 534],[2778, 1705],[1563, 2450],[1543, 569]])

#     # Calculate Homography
#     h, status = cv2.findHomography(pts_src, pts_dst)
    
#     # Warp source image to destination based on homography
#     im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    
#     # Display images
#     cv2.imshow("Source Image", im_src)
#     cv2.imshow("Destination Image", im_dst)
#     cv2.imshow("Warped Source Image", im_out)

#     cv2.waitKey(0)

img1 = cv2.imread('h1.jpeg',0)  #queryimage # left image
img2 = cv2.imread('h2.jpeg',0) #trainimage # right image

sift = cv2.SIFT_create(contrastThreshold = 0.01)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

des1 = np.float32(des1)
des2 = np.float32(des2)
# detected keypoints need to be matched to each other
# this is done by using flann KNN matching. It is better than eucledian+histogram matching
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []
# matches aren't always perfect. Code below checks if keypoints meet Lowe's criteria
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
# Calculate Homography
h, status = cv2.findHomography(pts1, pts2)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))

# Display images
cv2.imshow("Source Image", img1)
cv2.imshow("Destination Image", img2)
cv2.imshow("Warped Source Image", img2)

cv2.waitKey(0)