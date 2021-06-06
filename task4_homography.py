#!/usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__' :

    # Read source image.
    im_src = cv2.imread('HG2.jpeg')
    height , width , _ = im_src.shape
    # Four corners of the book in source image
    pts_src = np.array([[2522, 534],[2778, 1705],[1563, 2450],[1543, 569]])
    # pts_src = np.array([[1676, 793], [1789, 1693],[63,1844],[1813, 2642],[1290,1787]])


    # Read destination image.
    im_dst = cv2.imread('HG1.jpeg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[2282, 584], [2900, 1611],[1944,2729],[1360, 932]])
    # pts_dst = np.array([[3076, 289],[2588, 1635],[99, 7670],[1991, 2986],[1835, 1437]])

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    print("H is coming")
    print(h)
    # for pt2 in pts_src:
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    # # img_dst = cv2.line(img_dst, (x0, y0), (x1, y1), color, 1)
    #     im_src = cv2.circle(im_src, tuple(pt2), 20, color, -1)
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    for i in range(len(pts_src)):
        # pos = [[pts_src[i,0]],[pts_src[i,1]],[1]]
        pos = [pts_src[i,0],pts_src[i,1],1]
        new_pos = np.matmul(h,pos)
        pts_src[i, 0] = new_pos[0]/new_pos[2]
        pts_src[i, 1]= new_pos[1]/new_pos[2]
    # Display images

    np.random.seed(2)
    # for pt1, pt2 in zip(pts_dst, pts_src):
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    # # img_dst = cv2.line(img_dst, (x0, y0), (x1, y1), color, 1)
    #     im_dst = cv2.circle(im_dst, tuple(pt1), 20, color, -1)
    # # img_dst = cv2.line(img_dst, (x0, y0), (x1, y1), color, 1)
    #     im_out = cv2.circle(im_out, tuple(pt2), 20, color, -1)

    horizontal_concat = np.concatenate((im_dst, im_out), axis=1)
    im_fin=horizontal_concat
    for pt1, pt2 in zip(pts_dst, pts_src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        im_fin = cv2.circle(im_fin, tuple(pt1), 40, color, -1)
        im_fin = cv2.circle(im_fin, tuple((pt2[0]+width, pt2[1])), 40, color, -1)
        im_fin = cv2.line(im_fin, (pt1[0], pt1[1]), (pt2[0]+width, pt2[1]), color,15)
    cv2.imshow("Together", im_fin)
    cv2.imwrite("homography_matrix_fig.jpeg",im_fin)
       # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey(0)

