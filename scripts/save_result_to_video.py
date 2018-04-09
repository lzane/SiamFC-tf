#! /usr/bin/env python

import os.path as osp
import sys

import cv2
from sacred import Experiment

ex = Experiment()
from matplotlib.pyplot import Rectangle

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))


def readbbox(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
    return bboxs


def create_bbox(bbox, color):
    return Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                     fill=False,  # remove background\n",
                     edgecolor=color)


def set_bbox(artist, bbox):
    artist.set_xy((bbox[0], bbox[1]))
    artist.set_width(bbox[2])
    artist.set_height(bbox[3])


@ex.config
def configs():
    videoname = 'demo'
    runname = 'SiamFC-3s-color-pretrained'
    track_log_dir = 'Logs/SiamFC/track_model_inference/{}/{}'.format(runname, videoname)
    output_path = '../'


@ex.automain
def main(track_log_dir, output_path, videoname):
    track_log_dir = osp.join(track_log_dir)
    te_bboxs = readbbox(osp.join(track_log_dir, 'track_rect.txt'))
    num_frames = len(te_bboxs)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    org_img = cv2.imread(osp.join(track_log_dir, 'image_origin{}.jpg'.format(1)))
    out = cv2.VideoWriter(osp.join(output_path, videoname + '.avi'), fourcc, 24.0, (org_img.shape[1], org_img.shape[0]))
    print(org_img.shape)

    for i in range(1, num_frames):
        org_img = cv2.imread(osp.join(track_log_dir, 'image_origin{}.jpg'.format(i)))
        bbox = te_bboxs[i]
        cv2.rectangle(org_img, (int(bbox[0]), int(bbox[1])),
                      (
                          int(bbox[0]) + int(bbox[2]),
                          int(bbox[1]) + int(bbox[3])),
                      (0, 0, 255), 2)
        cv2.imshow('frame', org_img)
        out.write(org_img)
        cv2.waitKey(1)

    out.release()
    cv2.destroyAllWindows()
