#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Class for tracking using a track model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from utils.infer_utils import convert_bbox_format, Rectangle
from utils.misc_utils import get_center


class TargetState(object):
    """Represent the target state."""

    def __init__(self, bbox, search_pos, scale_idx):
        self.bbox = bbox  # (cx, cy, w, h) in the original image
        self.search_pos = search_pos  # target center position in the search image
        self.scale_idx = scale_idx  # scale index in the searched scales


class Tracker(object):
    """Tracker based on the siamese model."""

    def __init__(self, siamese_model, model_config, track_config):
        self.siamese_model = siamese_model
        self.model_config = model_config
        self.track_config = track_config

        self.num_scales = track_config['num_scales']
        logging.info('track num scales -- {}'.format(self.num_scales))
        scales = np.arange(self.num_scales) - get_center(self.num_scales)
        self.search_factors = [self.track_config['scale_step'] ** x for x in scales]

        self.x_image_size = track_config['x_image_size']  # Search image size
        self.window = None  # Cosine window
        self.log_level = track_config['log_level']

        self.frame2crop_scale = None
        self.original_target_height = None
        self.original_target_width = None
        self.search_center = None
        self.current_target_state = None

    def initialize(self, sess, first_bbox, frame, logdir='/tmp'):
        """Runs tracking on a single image sequence."""
        # Get initial target bounding box and convert to center based
        bbox = convert_bbox_format(first_bbox, 'center-based')

        # Feed in the first frame image to set initial state.
        bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
        input_feed = [frame, bbox_feed]
        self.frame2crop_scale = self.siamese_model.initialize(sess, input_feed)

        # Storing target state
        self.original_target_height = bbox.height
        self.original_target_width = bbox.width
        self.search_center = np.array([get_center(self.x_image_size),
                                       get_center(self.x_image_size)])
        self.current_target_state = TargetState(bbox=bbox,
                                                search_pos=self.search_center,
                                                scale_idx=int(get_center(self.num_scales)))

    def track(self, sess, frame):
        bbox_feed = [self.current_target_state.bbox.y, self.current_target_state.bbox.x,
                     self.current_target_state.bbox.height, self.current_target_state.bbox.width]
        input_feed = [frame, bbox_feed]

        outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
        search_scale_list = outputs['scale_xs']
        response = outputs['response']
        response_size = response.shape[1]

        # Choose the scale whole response map has the highest peak
        if self.num_scales > 1:
            response_max = np.max(response, axis=(1, 2))
            penalties = self.track_config['scale_penalty'] * np.ones((self.num_scales))
            current_scale_idx = int(get_center(self.num_scales))
            penalties[current_scale_idx] = 1.0
            response_penalized = response_max * penalties
            best_scale = np.argmax(response_penalized)
        else:
            best_scale = 0

        response = response[best_scale]

        with np.errstate(all='raise'):  # Raise error if something goes wrong
            response = response - np.min(response)
            response = response / np.sum(response)

        if self.window is None:
            window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                            np.expand_dims(np.hanning(response_size), 0))
            self.window = window / np.sum(window)  # normalize window
        window_influence = self.track_config['window_influence']
        response = (1 - window_influence) * response + window_influence * self.window

        # Find maximum response
        r_max, c_max = np.unravel_index(response.argmax(),
                                        response.shape)

        # Convert from crop-relative coordinates to frame coordinates
        p_coor = np.array([r_max, c_max])
        # displacement from the center in instance final representation ...
        disp_instance_final = p_coor - get_center(response_size)
        # ... in instance feature space ...
        upsample_factor = self.track_config['upsample_factor']
        disp_instance_feat = disp_instance_final / upsample_factor
        # ... Avoid empty position ...
        r_radius = int(response_size / upsample_factor / 2)
        disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
        # ... in instance input ...
        disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
        # ... in instance original crop (in frame coordinates)
        disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
        # Position within frame in frame coordinates
        y = self.current_target_state.bbox.y
        x = self.current_target_state.bbox.x
        y += disp_instance_frame[0]
        x += disp_instance_frame[1]

        # Target scale damping and saturation
        target_scale = self.current_target_state.bbox.height / self.original_target_height
        search_factor = self.search_factors[best_scale]
        scale_damp = self.track_config['scale_damp']  # damping factor for scale update
        target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
        target_scale = np.maximum(0.2, np.minimum(5.0, target_scale))

        # Some book keeping
        height = self.original_target_height * target_scale
        width = self.original_target_width * target_scale
        self.current_target_state.bbox = Rectangle(x, y, width, height)
        self.current_target_state.scale_idx = best_scale
        self.current_target_state.search_pos = self.search_center + disp_instance_input

        assert 0 <= self.current_target_state.search_pos[0] < self.x_image_size, \
            'target position in feature space should be no larger than input image size'
        assert 0 <= self.current_target_state.search_pos[1] < self.x_image_size, \
            'target position in feature space should be no larger than input image size'

        reported_bbox = convert_bbox_format(self.current_target_state.bbox, 'top-left-based')
        return reported_bbox
