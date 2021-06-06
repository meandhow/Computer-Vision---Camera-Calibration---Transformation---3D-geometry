import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import match_descriptors, ORB, plot_matches

# this functionis used to paint epipolar lines and corresponding points
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1src.shape
    img1color = img1src
    img2color = img2src
    # img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2RGB)
    # img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2RGB)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 20)
        img1color = cv2.circle(img1color, tuple(pt1), 45, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 45, color, -1)
    return img1color, img2color

img1 = cv2.imread('nL.jpeg',1)  #queryimage # left image
img2 = cv2.imread('nP.jpeg',1) #trainimage # right image
img1_gray = cv2.imread('nL.jpeg',0)  #queryimage # left image
img2_gray = cv2.imread('nP.jpeg',0) #trainimage # right image

cv2.imshow("first image",img1)

# keypoint_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2,matches[300:500], None, **draw_params)
# cv2.imshow("Keypoint matches", keypoint_matches)

# time to fidn fundamental matrix!
# itr transform one image coordinate ssytem into anothers, allowing for funky
# transofrmations between them
# pts1=[[2038 633],[3637 1810], [1814 1860],[370 1492], [584 252],[1167 2750],[3704 529],[3361 2430],[1159 2081],[1392 1042]]
# pts2=[[2163 8650], [3685 2069],[1757 2112],[746 1594],[978 439],[1096 2913],[3750 804],[3154 2736],[1309 2209],[1697 1148]]
pts1=[(2038, 633),(3637, 1810), (1814, 1860),(370, 1492), (584, 252),(1167, 2750),(3704, 529),(3361, 2430),(1159, 2081),(1392, 1042)]
pts2=[(2163, 8650), (3685, 2069),(1757, 2112),(746, 1594),(978, 439),(1096, 2913),(3750, 804),(3154, 2736),(1309, 2209),(1697, 1148)]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
print(pts1)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
print("F is coming")
print(F)

# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
img_epilines = np.concatenate((img5, img3), axis=1)
cv2.imshow("Epilines", img_epilines)
cv2.imwrite("epilines_fig.jpeg",img_epilines)
# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.suptitle("Epilines in both images")
# plt.show()

# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1, _ = img5.shape
h2, w2, _ = img3.shape
__, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
)

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
img1_r_gray = cv2.warpPerspective(img1_gray, H1, (w1, h1))
img2_r_gray = cv2.warpPerspective(img2_gray, H2, (w2, h2))
cv2.imwrite("rectified_1.png", img1_rectified)
cv2.imwrite("rectified_2.png", img2_rectified)
img_rectified = np.concatenate((img1_rectified, img2_rectified), axis=1)
cv2.imshow("Rectified", img_rectified)
cv2.imwrite("rectified_fig.jpeg",img_rectified)

import math
slope = math.degrees(math.atan(0.2647/0.3856))
print(slope)
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
img1_rectified = rotate_image(img1_rectified, slope)
img2_rectified = rotate_image(img2_rectified, slope)
# img1_r_gray = rotate_image(img1_r_gray, slope)
# img2_r_gray = rotate_image(img2_r_gray, slope)
img_rectified = np.concatenate((img1_rectified, img2_rectified), axis=1)
cv2.imshow("Rectified", img_rectified)
cv2.imwrite("rectified_fig.jpeg",img_rectified)
# # Draw the rectified images
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1_rectified, cmap="gray")
# axes[1].imshow(img2_rectified, cmap="gray")
# # axes[0].axhline(250)
# # axes[1].axhline(250)
# # axes[0].axhline(450)
# # axes[1].axhline(450)
# plt.suptitle("Rectified images")
# plt.savefig("rectified_images.png")
# plt.show()

# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 10
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity_SGBM = stereo.compute(img1_r_gray, img2_r_gray)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
disparity_SGBM_r = rotate_image(disparity_SGBM, slope)
plt.imshow(disparity_SGBM_r, cmap='plasma')
plt.colorbar()
plt.show()
cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM_r)


# img1 = cv2.imread('imL.png',0)  #queryimage # left image
# img2 = cv2.imread('imR.png',0) #trainimage # right image

# descriptor_extractor = ORB()

# descriptor_extractor.detect_and_extract(img1)
# kp1 = descriptor_extractor.keypoints
# des1 = descriptor_extractor.descriptors

# descriptor_extractor.detect_and_extract(img2)
# kp2 = descriptor_extractor.keypoints
# des2 = descriptor_extractor.descriptors