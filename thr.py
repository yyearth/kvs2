#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/4/19 10:40
# @Author  : yy

import threading
import time
def do(name):
    while True:
        print('running thread:', name)
        time.sleep(1)

t1 = threading.Thread(target=do, args=('sub',))
t1.deamon = False
t1.start()
# t1.join()
print('main thread-----------------')
time.sleep(1)
print('main thread end-------------')

