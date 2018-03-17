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

import logging
import os
import os.path as osp
import sys

import cv2
import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import auto_select_gpu, mkdir_p, load_cfgs


def resize(img):
    return img
    # return cv2.resize(img, (800, 600))


def main():
    checkpoint = 'Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'
    os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

    model_config, _, track_config = load_cfgs(checkpoint)
    track_config['log_level'] = 1

    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
    g.finalize()

    if not osp.isdir(track_config['log_dir']):
        logging.info('Creating inference directory: %s', track_config['log_dir'])
        mkdir_p(track_config['log_dir'])

    # video_dirs = []
    # for file_pattern in input_files.split(","):
    #   video_dirs.extend(glob(file_pattern))
    # logging.info("Running tracking on %d videos matching %s", len(video_dirs), input_files)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(graph=g, config=sess_config) as sess:
        restore_fn(sess)

        tracker = Tracker(model, model_config=model_config, track_config=track_config)
        video_name = "Camera"
        video_log_dir = osp.join(track_config['log_dir'], video_name)
        mkdir_p(video_log_dir)
        if len(sys.argv) <= 1:
            print('[ERROR]: File path error!')
            return

        if sys.argv[1] == "camera":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(sys.argv[1])

        # select ROI and initialize the model
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = resize(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('o'):
                break

        # Select ROI
        r = cv2.selectROI(frame)
        first_line = "{},{},{},{}".format(r[0], r[1], r[2], r[3])
        bb = [int(v) for v in first_line.strip().split(',')]
        init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python
        tracker.initialize(sess, init_bb, frame, video_log_dir)

        while True:
            ret, frame = cap.read()
            frame = resize(frame)
            reported_bbox = tracker.track(sess, frame)
            # with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
            #     for region in trajectory:
            #         rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
            #                                           region.width, region.height)
            #         f.write(rect_str)

            # Display the resulting frame
            print(reported_bbox)
            # print(reported_bbox[1])
            cv2.rectangle(frame, (int(reported_bbox[0]), int(reported_bbox[1])),
                          (
                              int(reported_bbox[0]) + int(reported_bbox[2]),
                              int(reported_bbox[1]) + int(reported_bbox[3])),
                          (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


main()
