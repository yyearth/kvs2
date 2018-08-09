#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time    : 2018/4/18 10:33
# @Author  : yy

from multiprocessing import Process,Queue
# from queue import Queue
import time

q = Queue()
class MyProcess(Process):
    def __init__(self):
        super().__init__()
        self._i = 0

    def run(self):
        while True:
            print(q.get())
            print(q.put(str(self._i)+'from process'))
            self._i +=1

i = 0
MyProcess().start()
time.sleep(1)
q.put('0:main process')
print(q.get())
time.sleep(1)
print('main process end')
