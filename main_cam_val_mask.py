
import modelth_cam_pretrain_new as model
import dataset
from timer import Timer
import sys
import cv2
import numpy as NP
from util import compute_iou
from example_generator import crop_pad_image, make_single_training_example
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as PL
import matplotlib.patches as PA

data = dataset.ImageNetVidDataset('/beegfs/qg323/ILSVRC', 'map_clsloc.txt')
model.create_model(len(data.classes))
chk = 0 if len(sys.argv) <= 1 else int(sys.argv[1]) - 1
if chk != 0:
    print 'Loading /beegfs/qg323/model_mask_%03d' % chk
    model.load('/beegfs/qg323/model_mask_%03d' % chk)

def addbox(ax, b, color='red'):
    ax.add_patch(PA.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], ec=color, fill=False, lw=1))

i = 0
with open('imagenet_labels2') as f:
    loc_clsname = pickle.load(f)
while True:
    imgs, bboxes, cls = data.pick_video(i, train=False)
    if imgs is None:
        break
    elif imgs.shape[0] == 0:
        i += 1
        continue

    prev, _, _, _ = crop_pad_image(bboxes[0], imgs[0], padding=False)
    prev_bbox = bboxes[0]
    for t, (img, bbox) in enumerate(zip(imgs, bboxes)):
        _img, _bbox, _search, _ex, _ey, _ = make_single_training_example(img, prev_bbox, bboxes[t], False, 0, 0, 0, 0)
        _search_width = _img.shape[1]
        _search_height = _img.shape[0]
        _prev = cv2.resize(prev, (224, 224))
        _img = cv2.resize(_img, (224, 224))
        _pred, topk, cam = model.val_func(
                _img[NP.newaxis] / 255.,
                _prev[NP.newaxis] / 255.,
                _bbox[NP.newaxis],
                NP.array([0]),
                NP.array([cls]),
                )
        _pred = _pred[0]
        cam = cam[0]
        topk = topk[0]
        cam_classes = [loc_clsname[_] for _ in topk]

        _unscaled = (_pred + 1) * NP.array([
            1. * _search_height / 2,
            1. * _search_width / 2,
            1. * _search_height / 2,
            1. * _search_width / 2,
            ])
        pred = NP.array([
            max(0, _unscaled[0] - _ey + _search[0]),
            max(0, _unscaled[1] - _ex + _search[1]),
            min(img.shape[0], _unscaled[2] - _ey + _search[0]),
            min(img.shape[1], _unscaled[3] - _ex + _search[1]),
            ])
        print '#', i, t, compute_iou(bbox, pred), bbox, pred

        prev, _, _, _ = crop_pad_image(pred, img, padding=False)
        prev_bbox = pred

        fig, ax = PL.subplots(2, 4, figsize=(16, 8))
        ax[0, 0].imshow(img[:, :, ::-1])
        addbox(ax[0, 0], bbox, 'yellow')
        addbox(ax[0, 0], pred, 'red')
        ax[0, 1].imshow(cam[0])
        ax[0, 1].set_title(cam_classes[0])
        ax[0, 2].imshow(cam[1])
        ax[0, 2].set_title(cam_classes[1])
        ax[1, 0].imshow(cam[2])
        ax[1, 0].set_title(cam_classes[2])
        ax[1, 1].imshow(cam[3])
        ax[1, 1].set_title(cam_classes[3])
        ax[1, 2].imshow(cam[4])
        ax[1, 2].set_title(cam_classes[4])
        ax[0, 3].imshow(_img[:, :, ::-1] / 255.)
        ax[1, 3].imshow(_prev[:, :, ::-1] / 255.)
        PL.savefig('viz-val/%05d-%06d.jpg' % (i, t), bbox_inches='tight')
        PL.close()
    os.system('ffmpeg -y -i viz-val/%05d-%%06d.jpg -framerate 2 viz-val/%05d.avi' % (i, i))
    os.system('rm -rf viz-val/%05d-*.jpg' % i)
    i += 1
