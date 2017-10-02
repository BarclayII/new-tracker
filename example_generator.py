
import numpy as NP
import numpy.random as RNG

class TimeoutError(Exception):
    pass

def bbox_shift(img, tlbr, lambda_scale, lambda_shift, min_scale, max_scale):
    top, left, bottom, right = tlbr
    width = right - left
    height = bottom - top
    cx = (left + right) / 2
    cy = (bottom + top) / 2

    rows = img.shape[0]
    cols = img.shape[1]

    new_width = -1
    trials = 0
    while new_width < 0 or new_width > cols - 1:
        #width_scale_factor = NP.clip(RNG.exponential(1. / lambda_scale) * RNG.choice([-1, 1]), min_scale, max_scale)
        width_scale_factor = RNG.uniform(-0.2, 0.2)
        new_width = NP.clip(width * (1 + width_scale_factor), 1, cols - 1)
        trials += 1
        if trials > 10:
            raise TimeoutError
    new_height = -1
    trials = 0
    while new_height < 0 or new_height > rows - 1:
        #height_scale_factor = NP.clip(RNG.exponential(1. / lambda_scale) * RNG.choice([-1, 1]), min_scale, max_scale)
        height_scale_factor = RNG.uniform(-0.2, 0.2)
        new_height = NP.clip(height * (1 + height_scale_factor), 1, rows - 1)
        trials += 1
        if trials > 10:
            raise TimeoutError

    #new_x_temp = cx + width * RNG.exponential(1. / lambda_shift) * RNG.choice([-1, 1])
    new_x_temp = cx + width * RNG.uniform(-0.5, 0.5)
    new_cx = NP.clip(new_x_temp, new_width / 2, cols - new_width / 2)
    new_y_temp = cy + height * RNG.uniform(-0.5, 0.5)
    #new_y_temp = cy + height * RNG.exponential(1. / lambda_shift) * RNG.choice([-1, 1])
    new_cy = NP.clip(new_y_temp, new_height / 2, rows - new_height / 2)

    return NP.array([
        new_cy - new_height / 2,
        new_cx - new_width / 2,
        new_cy + new_height / 2,
        new_cx + new_width / 2
        ])

def compute_crop_pad_image_loc(tlbr, img, ctx_factor=2, padding=True):
    top, left, bottom, right = tlbr
    width = right - left
    height = bottom - top
    cx = (left + right) / 2
    cy = (bottom + top) / 2

    cols = img.shape[1]
    rows = img.shape[0]

    if not padding:
        ctx_factor = min(
                ctx_factor,
                cx / (width / 2.),
                (cols - cx) / (width / 2.),
                cy / (height / 2.),
                (rows - cy) / (height / 2.),
                )

    output_width = max(1, width * ctx_factor)
    output_height = max(1, height * ctx_factor)

    roi_left = max(0, cx - output_width / 2)
    roi_top = max(0, cy - output_height / 2)
    '''
    left_half = min(output_width / 2, cx)
    right_half = min(output_width / 2, cols - cx)
    roi_width = max(1, left_half + right_half)
    top_half = min(output_height / 2, cy)
    bottom_half = min(output_height / 2, rows - cy)
    roi_height = max(1, top_half + bottom_half)
    '''
    roi_right = min(cols, cx + output_width / 2)
    roi_bottom = min(rows, cy + output_height / 2)

    #return roi_top, roi_left, roi_top + roi_height, roi_left + roi_width, ctx_factor
    return roi_top, roi_left, roi_bottom, roi_right, ctx_factor

def crop_pad_image(tlbr, img, ctx_factor=2, padding=True):
    new_top, new_left, new_bottom, new_right, ctx_factor = compute_crop_pad_image_loc(tlbr, img, ctx_factor, padding)
    cols = img.shape[1]
    rows = img.shape[0]
    cx = (tlbr[..., 1] + tlbr[..., 3]) / 2.
    cy = (tlbr[..., 0] + tlbr[..., 2]) / 2.
    width = tlbr[..., 3] - tlbr[..., 1]
    height = tlbr[..., 2] - tlbr[..., 0]

    roi_left = int(min(new_left, cols - 1))
    roi_top = int(min(new_top, rows - 1))
    roi_width = int(NP.clip(NP.ceil(new_right - new_left), 1, cols))
    roi_height = int(NP.clip(NP.ceil(new_bottom - new_top), 1, rows))

    cropped_image = img[roi_top:roi_top+roi_height, roi_left:roi_left+roi_width]

    output_width = int(max(NP.ceil(width * ctx_factor), 1))
    output_height = int(max(NP.ceil(height * ctx_factor), 1))
    edge_spacing_x = max(0, output_width / 2 - cx)
    edge_spacing_y = max(0, output_height / 2 - cy)
    output_width = max(output_width, roi_width)
    output_height = max(output_height, roi_height)
    edge_spacing_x = int(min(edge_spacing_x, output_width - 1))
    edge_spacing_y = int(min(edge_spacing_y, output_height - 1))

    edge_spacing_x = 0
    edge_spacing_y = 0

    output_img = NP.zeros((roi_height, roi_width, 3))
    output_img[edge_spacing_y : edge_spacing_y + roi_height, edge_spacing_x : edge_spacing_x + roi_width] = cropped_image
    roi_tlbr = NP.array((roi_top, roi_left, roi_top + roi_height, roi_left + roi_width))
    return output_img, roi_tlbr, edge_spacing_x, edge_spacing_y

def make_single_training_example(img, bbox, gt, synth, lambda_scale, lambda_shift, min_scale, max_scale, scale_factor=2, scale_off=-1, random=False, img_fake=None):
    _random = False
    if synth:
        gt_width = gt[3] - gt[1]
        gt_height = gt[2] - gt[0]
        shifted_bbox = bbox_shift(img, gt, lambda_scale, lambda_shift, min_scale, max_scale)
    else:
        shifted_bbox = bbox
    fake = synth and random and RNG.randint(2) == 0
    search_img, search_bbox, edge_spacing_x, edge_spacing_y = crop_pad_image(shifted_bbox, img_fake if fake else img, padding=True)

    recentered_bbox = NP.array([
        gt[0] - search_bbox[0] + edge_spacing_y,
        gt[1] - search_bbox[1] + edge_spacing_x,
        gt[2] - search_bbox[0] + edge_spacing_y,
        gt[3] - search_bbox[1] + edge_spacing_x,
        ])
    search_width = search_img.shape[1]
    search_height = search_img.shape[0]

    scaled_bbox = recentered_bbox * NP.array([
        1. * scale_factor / search_height,
        1. * scale_factor / search_width,
        1. * scale_factor / search_height,
        1. * scale_factor / search_width,
        ]) + scale_off

    return search_img, scaled_bbox, search_bbox, edge_spacing_x, edge_spacing_y, _random
