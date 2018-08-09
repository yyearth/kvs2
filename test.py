#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/4/15 23:04
# @Author  : yy

import numpy as np
import cv2

# from main import ColorTracker, cc, fc, wr,mtx, dist
#
# ct = ColorTracker([255, 232, 7])
# img = cv2.imread('photoS2.jpg')
# w, h = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# # img = cv2.undistort(img, mtx, dist, None, newcameramtx)
# cv2.circle(img, (int(cc[0]), int(cc[1])), 4, (0, 0, 255))
# u, v, iw, ih, x, y = ct.track(img)
# print(u, v, iw, ih, x, y)
# if u != -1:
#     wp = max(iw, ih)
#     d = fc[0] * (wr / wp)
#     dX = x * wr / wp
#     dY = y * wr / wp
#     print(d, dX, dY)
# cv2.imshow('img', img)
# cv2.waitKey()

img = cv2.imread('sample.jpg')
# img = cv2.bilateralFilter(img, 15, 100, 100)
# edges = cv2.Canny(img,100,200)
bgr = [166, 85, 0]
sigma = 1
hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
_color_l = np.array([cv2.add(hsv[0][0], - sigma)[0][0], 100, 100], np.uint8)
_color_u = np.array([cv2.add(hsv[0][0], sigma)[0][0], 255, 255], np.uint8)

img2 = cv2.GaussianBlur(img, (21, 21), 1)
img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, _color_l, _color_u)
m = mask[300:420, 180:340]
# kernel = np.ones((5, 5), np.uint8)
# dilate = cv2.dilate(m, kernel, iterations=1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
cl = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('close.jpg', cl)
cv2.imshow('img', cl)
cv2.imshow('img2', m)
cv2.waitKey()
