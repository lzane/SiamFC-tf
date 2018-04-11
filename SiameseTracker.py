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

import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(CURRENT_DIR)

from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import auto_select_gpu, mkdir_p, load_cfgs, rmdir


class SiameseTracker:
    def __init__(self,
                 debug=0,
                 checkpoint='Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained'):
        os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

        # run only on cpu
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_config, _, track_config = load_cfgs(checkpoint)
        track_config['log_level'] = debug

        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)
        g.finalize()
        if not osp.isdir(track_config['log_dir']):
            logging.info('Creating inference directory: %s', track_config['log_dir'])
            mkdir_p(track_config['log_dir'])


        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(graph=g, config=sess_config)
        # sess.run(tf.global_variables_initializer())
        restore_fn(sess)
        tracker = Tracker(model, model_config=model_config, track_config=track_config)
        video_name = "demo"
        video_log_dir = osp.join(track_config['log_dir'], video_name)
        rmdir(video_log_dir)
        mkdir_p(video_log_dir)
        self.tracker = tracker
        self.sess = sess
        self.video_log_dir = video_log_dir
        self.graph = g

    def set_first_frame(self, frame, r):
        first_line = "{},{},{},{}".format(r[0], r[1], r[2], r[3])
        bb = [int(v) for v in first_line.strip().split(',')]
        init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python
        self.tracker.initialize(self.sess, init_bb, frame, self.video_log_dir)

    def track(self, frame):
        reported_bbox = self.tracker.track(self.sess, frame)
        return reported_bbox
