#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/4/15 16:41
# @Author  : yy

import numpy as np
import cv2
import socket
import threading
# from multiprocessing import Process, Queue
# from queue import Queue
import xml.etree.ElementTree as ET
import time

fc = (1301.11026, 1298.09944)
cc = (361.16784, 213.44209)
kc = (0.07171, 0.61192, -0.00554, -0.00077, 0.00000)

mtx = np.array([[fc[0],     0, cc[0]],
                [    0, fc[1], cc[1]],
                [    0,     0,    1]])
dist = np.array((0.07171, 0.61192, -0.00554, -0.00077, 0.00000))

wr = 31
Ready = False

def center(img,principle=True):
    h, w = img.shape[:2]
    if principle:
        cv2.line(img, (0, int(cc[1])), (w, int(cc[1])),(0, 255, 0))
        cv2.line(img, (int(cc[0]), 0), (int(cc[0]), h),(0, 255, 0))
    else:
        cv2.line(img, (0, h//2), (w, h//2), (0, 255, 0))
        cv2.line(img, (w//2, 0), (w//2, h), (0, 255, 0))

    return img

def framewrappe(x, y, z, a, b, c):
    data = '<Sensor><frame X="%.6f" Y="%.6f" Z="%.6f" ' \
           'A="%.6f" B="%.6f" C="%.6f"/></Sensor>' % (x, y, z, a, b, c)
    return data


def frameparse(sr):
    root = ET.fromstring(sr)
    if root is None:
        return None
    att = root[0].attrib
    return [float(att['X']), float(att['Y']), float(att['Z']), float(att['A']), float(att['B']), float(att['C'])]


class ColorTracker(object):

    def __init__(self, bgr=None, sigma=5):
        # super().__init__()
        self._sigma = sigma
        self._targetbgr = bgr
        self._color_l = [0, 0, 0]
        self._color_u = [0, 0, 0]
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if bgr is not None:
            self.setColor(bgr)

    def setColor(self, bgr):
        self._targetbgr = bgr
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        self._color_l = np.array([cv2.add(hsv[0][0], - self._sigma)[0][0], 100, 100], np.uint8)
        self._color_u = np.array([cv2.add(hsv[0][0], self._sigma)[0][0], 255, 255], np.uint8)

    def track(self, img):
        sh, sw, _ = img.shape
        x, y = cc[0], cc[1]
        u, v, w, h = -1, -1, 0, 0
        if self._targetbgr is None:
            return u, v, w, h, cc[0], cc[1]

        img2 = cv2.GaussianBlur(img, (21, 21), 1)
        img_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, self._color_l, self._color_u)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 700:
                continue
            # print(cv2.contourArea(cnt))
            ix, iy, iw, ih = cv2.boundingRect(cnt)
            cv2.rectangle(img, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)

        if len(contours) > 0:
            cnt_max = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt_max) < 700:
                return u, v, w, h, cc[0], cc[1]
            u, v, w, h = cv2.boundingRect(cnt_max)
            cv2.rectangle(img, (u, v), (u + w, v + h), (0, 0, 225), 1)
            x = u + w / 2 - cc[0]
            y = v + h / 2 - cc[1]
            # cv2.circle(img, (, ), 2, (0, 0, 255), -1)
        return u, v, w, h, x, y


class SocketThread(threading.Thread):

    def __init__(self, addr):
        super().__init__()
        self.addr = addr
        self.sock = socket.socket()


    def run(self):
        global Ready
        print('connecting...')
        self.sock.connect(self.addr)
        print('connect to KUKA server@', self.addr)
        while True:
            rec = self.sock.recv(1024).decode('ascii')
            Ready = True
            data = frameparse(rec)
            # data[0]  data[1]  data[2]  data[3]  data[4]  data[5]
            #   X        Y        Z        A        B        C
            # print(data)
            # position.put(data)
            # id, idX, idY = target.get()
            # data[1] = data[1] + 5
            # data[2] = data[2] - idY
            # time.sleep(1)
            print(data)
            data = framewrappe(*data).encode('ascii')
            self.sock.send(data)
            Ready = False


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    d, dX, dY = 0, 0, 0
    target = Queue(maxsize=1)
    position = Queue(maxsize=1)
    ct = ColorTracker([255, 232,  7], 10)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    SocketThread(('172.31.1.147', 54600)).start()  # ('172.31.1.147', 54600)
    # while True:
    #     time.sleep(1)
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        u, v, iw, ih, x, y = ct.track(frame)
        center(frame, True)
        if u != -1:
            wp = max(iw, ih)
            d = fc[0]*(wr/wp)
            dX = x*wr/wp
            dY = y*wr/wp
            cv2.putText(frame, 'd:%3.2f dX:%3.2f dY:%3.2f' % (d, dX, dY), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 13:
            if Ready:
                if target.empty():
                    target.put((d, dX, dY))
                else:
                    target.get()
                    target.put((d, dX, dY))
            else: print('busy...')
        elif key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite('snap.jpg', frame)
    cv2.destroyAllWindows()
    cap.release()

