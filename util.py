import numpy as NP
import matplotlib.patches as PA

def compute_iou(a, b):
    # of arbitrary shape, the last dimension has 4 elements
    # Somehow Theano/GPU does not support axis=-1 as in Theano/CPU and numpy.
    a_top, a_left, a_bottom, a_right = map(
            lambda x: a.take(x, axis=a.ndim-1),
            range(4)
            )
    b_top, b_left, b_bottom, b_right = map(
            lambda x: b.take(x, axis=b.ndim-1),
            range(4)
            )
    a_w = NP.maximum(0, a_right - a_left)
    a_h = NP.maximum(0, a_bottom - a_top)
    b_w = NP.maximum(0, b_right - b_left)
    b_h = NP.maximum(0, b_bottom - b_top)
    inter_top = NP.maximum(a_top, b_top)
    inter_left = NP.maximum(a_left, b_left)
    inter_bottom = NP.minimum(a_bottom, b_bottom)
    inter_right = NP.minimum(a_right, b_right)
    inter_w = NP.maximum(0, inter_right - inter_left)
    inter_h = NP.maximum(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    a_area = a_w * a_h
    b_area = b_w * b_h
    union_area = a_area + b_area - inter_area
    return NP.where(
            union_area == 0,
            0,
            inter_area / union_area
            )

def addbox(ax, b, ec):
    ax.add_patch(PA.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], ec=ec, fill=False, lw=1))
