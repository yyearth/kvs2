#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/4/16 9:08
# @Author  : yy

import socket
import time
from main import framewrapper, frameparse

cli = socket.socket()
cli.connect(('172.31.1.147', 54600))
print('connect to KUKA server')
while True:
    rec = cli.recv(1024).decode('ascii')
    po = frameparse(rec)
    print(po)
    po[0] = po[0] + 30
    cm = framewrapper(*po).encode('ascii')
    # cm = '<Sensor><frame X="551.153503" Y="28.755083" Z="1404.805664" ' \
    #      'A="10.193840" B="11.517308" C="-13.239092"/></Sensor>'.encode('ascii')
    cli.send(cm)
    print(cm)



