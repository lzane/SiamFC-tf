#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

r"""Generate tracking results for videos using Siamese Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

import cv2


# CURRENT_DIR = osp.dirname(__file__)
# sys.path.append(CURRENT_DIR)
import datetime
from SiameseTracker import SiameseTracker


def preprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return res


def postprocess(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return res


def main():
    # debug = 0 , no log will produce
    # debug = 1 , will produce log file
    tracker = SiameseTracker(debug=1)
    time_per_frame = 0

    if len(sys.argv) <= 1:
        print('[ERROR]: File path error!')
        return

    if sys.argv[1] == "cam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(sys.argv[1])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = preprocess(frame)
        cv2.imshow('frame', postprocess(frame))
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break

    # select ROI and initialize the model
    r = cv2.selectROI(postprocess(frame))
    cv2.destroyWindow("ROI selector")
    print('ROI:', r)
    tracker.set_first_frame(frame, r)

    while True:
        start_time = datetime.datetime.now()
        ret, frame = cap.read()
        frame = preprocess(frame)
        reported_bbox = tracker.track(frame)

        # Display the resulting frame
        # print(reported_bbox)
        cv2.rectangle(frame, (int(reported_bbox[0]), int(reported_bbox[1])),
                      (
                          int(reported_bbox[0]) + int(reported_bbox[2]),
                          int(reported_bbox[1]) + int(reported_bbox[3])),
                      (0, 0, 255), 2)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        time_per_frame = 0.9 * time_per_frame + 0.1 * duration.microseconds
        cv2.putText(frame, 'FPS ' + str(round(1e6 / time_per_frame, 1)),
                    (30, 50), 0, 1, (0, 0, 255), 3)

        cv2.imshow('frame', postprocess(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


main()
