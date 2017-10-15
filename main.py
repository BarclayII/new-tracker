
import dataset
from modelth_cam_rnn import Model, tovar, tonumpy, compute_iou, compute_dev
import torch as T
import os
import matplotlib
if os.getenv('APPDEBUG', None):
    matplotlib.use('Agg')
import matplotlib.pyplot as PL
import sh
from util import addbox
import argparse
import sys
import numpy as NP

NP.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--train_iter', type=int, default=1000, help='training iterations per epoch')
parser.add_argument('--epochs', type=int, default=5, help='# of epochs')
parser.add_argument('--valid_size', type=int, default=10, help='validation set size')
parser.add_argument('-k', type=int, default=5, help='number of activated CAMs')
parser.add_argument('--ilsvrc', type=str, help='location of ImageNet VID dataset')
parser.add_argument('--rnn', action='store_true')

args = parser.parse_args()

sh.mkdir('-p', 'viz-val')

model = Model(args.k, args.rnn)
if not os.getenv('NOCUDA', None):
    model = model.cuda()
data = dataset.ImageNetVidDataset(args.ilsvrc, 'map_vid.txt')

opt = T.optim.Adam(p for p in model.parameters() if p.requires_grad)

valid_set = []

epoch = 0
valid_size = args.valid_size
train_iter = args.train_iter

for _ in range(valid_size):
    result = dataset.prepare_batch(1, 1, data, 15, 5, -0.4, 0.4, resize=None, swapdims=False, train=False,
            random=False, target_resize=False)
    px, pp, b, cls = result
    px = px[0]
    pp = pp[0]
    b = b[0]
    valid_set.append((px, pp, b))

def figure_title(cls, pi):
    cls = cls.split(',')[0]
    return '%s/%.5f' % (cls, pi)

for epoch in range(args.epochs):
    model.train()
    for i in range(train_iter):
        result = dataset.prepare_batch(1, 1, data, 15, 5, -0.4, 0.4, resize=None, swapdims=False, train=True,
                random=False, target_resize=False)

        px, pp, b, cls = result
        px = px[0]
        pp = pp[0]
        b = b[0]

        px = tovar(px).permute(0, 3, 1, 2)
        pp = tovar(pp).permute(2, 0, 1)
        b = tovar(b)
        b_list, b_internal, b_internal_gt, loss, b_gt = model.forward(px, pp, b)
        iou = compute_iou(b_list.data, b_gt.data).mean()
        d_pos, d_size, d_w_scale, d_h_scale = compute_dev(b_gt.data, b_list.data)
        opt.zero_grad()
        loss.backward()
        grad_norm = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            grad_norm += p.grad.data.norm(2) ** 2
        grad_norm = NP.sqrt(grad_norm)
        opt.step()
        print 'Training', epoch, i, tonumpy(loss), iou, d_pos.mean(), d_pos.std(), d_size.mean(), d_size.std(), d_w_scale.mean(), d_h_scale.mean(), grad_norm

    model.train()
    iou_sum = 0
    d_pos_sum = 0
    d_size_sum = 0
    d_w_scale_sum = 0
    d_h_scale_sum = 0
    for i in range(valid_size):
        px, pp, b = valid_set[i]
        x = px
        p = pp
        px = tovar(px).permute(0, 3, 1, 2)
        pp = tovar(pp).permute(2, 0, 1)
        b = tovar(b)

        b_list, b_internal, b_internal_gt, loss, b_gt = model.forward(px, pp, b, 1)
        iou = compute_iou(b_list.data, b_gt.data).mean()
        d_pos, d_size, d_w_scale, d_h_scale = compute_dev(b_gt.data, b_list.data)
        iou_sum += iou
        d_pos_sum += d_pos.mean()
        d_size_sum += d_size.mean()
        d_w_scale_sum += d_w_scale.mean()
        d_h_scale_sum += d_h_scale.mean()
        print 'Validation', epoch, i, tonumpy(loss), iou, d_pos.mean(), d_pos.std(), d_size.mean(), d_size.std(), d_w_scale.mean(), d_h_scale.mean()
        b, b_list, b_internal, b_internal_gt = tonumpy(b, b_list, b_internal, b_internal_gt)

        for t in range(5):
            fig, ax = PL.subplots(2, 4)
            ax[0][0].imshow(px[t, :, :, ::-1])
            addbox(ax[0][0], b[t], 'red')
            addbox(ax[0][0], b_list[0, t], 'yellow')
            ax[0][1].imshow(p[:, :, ::-1])
            ax[0][2].imshow(model.p_t_list[t][0].transpose(1, 2, 0)[:, :, ::-1])
            addbox(ax[0][2], (b_internal_gt[0, t] + 1) * 112, 'red')
            addbox(ax[0][2], (b_internal[0, t] + 1) * 112, 'yellow')
            ax[0][2].set_title(figure_title(model.cls_t_0_tops[0, 0], model.pi_t_0_tops[0, 0]), fontsize=6)
            # We always show top-5 CAMs here regardless of the actual k value
            ax[0][3].imshow(model.m_t_topk[t][0, 0])
            ax[0][3].set_title(figure_title(model.cls_t_topk[t][0, 0], model.pi_t_topk[t][0, 0]), fontsize=6)
            ax[1][0].imshow(model.m_t_topk[t][0, 1])
            ax[1][0].set_title(figure_title(model.cls_t_topk[t][0, 1], model.pi_t_topk[t][0, 1]), fontsize=6)
            ax[1][1].imshow(model.m_t_topk[t][0, 2])
            ax[1][1].set_title(figure_title(model.cls_t_topk[t][0, 2], model.pi_t_topk[t][0, 2]), fontsize=6)
            ax[1][2].imshow(model.m_t_topk[t][0, 3])
            ax[1][2].set_title(figure_title(model.cls_t_topk[t][0, 3], model.pi_t_topk[t][0, 3]), fontsize=6)
            ax[1][3].imshow(model.m_t_topk[t][0, 4])
            ax[1][3].set_title(figure_title(model.cls_t_topk[t][0, 4], model.pi_t_topk[t][0, 4]), fontsize=6)
            if os.getenv('APPDEBUG', None):
                PL.show()
            else:
                PL.savefig('viz-val/%05d-%05d-%d.png' % (epoch, i, t))
            PL.close()
    print 'Average validation IOU:', iou_sum / valid_size
    print 'Average center deviation:', d_pos_sum / valid_size
    print 'Average log size deviation:', d_size_sum / valid_size
    print 'Average x-axis deviation / width:', d_w_scale_sum / valid_size
    print 'Average y-axis deviation / height:', d_h_scale_sum / valid_size
    T.save(model, '/beegfs/qg323/model-%03d' % epoch)
