# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:29:02 2023

@author: candy
"""

import cv2
import collections
import numpy as np
import sousa
import time
import multiprocessing

def process_frame(frame, sousa_instance):
    # Your frame processing logic here
    player1_next = frame[73:123, 240:260]
    player1_next_next = frame[132:172, 259:274]

    q1.append(player1_next)
    q2.append(player1_next_next)

    if len(q1) == 4:
        flag1 = (np.array_equal(q1[0], q1[2]) == 0) and (np.array_equal(q2[0], q2[2]) == 0)
        flag2 = (np.array_equal(q1[1], q1[3]) == 1) and (np.array_equal(q2[1], q2[3]) == 1)

        if flag1 and flag2:
            sousa_instance.no1()

    sousa_instance.drop()

def main():
    capture = cv2.VideoCapture(1)
    sousa_instance = sousa.sousa()

    if not capture.isOpened():
        print("ビデオファイルを開くとエラーが発生しました")
        return

    frame_buffer = collections.deque(maxlen=4)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())

        if len(frame_buffer) == 4:
            process_frame(frame_buffer[0], sousa_instance)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()