#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import threading


class InfoGetter(object):
    '''
    Get Information from rostopic. It reduces delay.
    Note: Code copied from Seongin Na's work. I trust delay reduction 
    '''
    
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, msg):
        # uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = msg
        self._event.set()

    def get_msg(self, timeout=None):
        # blocks until the data is rx'd with optional timeout
        # returns the received message
        # 
        self._event.wait(timeout)
        return self._msg
