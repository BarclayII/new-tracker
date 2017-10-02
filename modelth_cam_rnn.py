
import torch as T
import torch.nn as NN
import torch.nn.functional as F
from torchvision.models import squeezenet1_1

import pickle
import os
import matplotlib
if not os.getenv('APPDEBUG', None):
    matplotlib.use('Agg')
import matplotlib.pyplot as PL
import numpy as NP
import numpy.random as RNG

from util import addbox

def tovar(*arrs, **kwargs):
    tensors = [(T.FloatTensor(NP.array(a).astype('float32')) if not T.is_tensor(a) and not isinstance(a, T.autograd.Variable) else a.float())
               for a in arrs]
    if len(arrs) == 1:
        return T.autograd.Variable(tensors[0]).cuda()
    else:
        return [T.autograd.Variable(t).cuda() for t in tensors]

def tonumpy(*vars_):
    result = []
    for v in vars_:
        if isinstance(v, T.autograd.Variable):
            result.append(v.data.cpu().numpy())
        elif T.is_tensor(v):
            result.append(v.cpu().numpy())
        else:
            result.append(v)
    return result

def safe_log_sigmoid(x, boundary=-80):
    x_bound = (x <= boundary).float()
    x_safe = x * (1 - x_bound)
    y = F.logsigmoid(x_safe)
    return y * (1 - x_bound) + x * x_bound

def compute_iou(a, b):
    a_top = a[:, 0]
    a_left = a[:, 1]
    a_bottom = a[:, 2]
    a_right = a[:, 3]
    b_top = b[:, 0]
    b_left = b[:, 1]
    b_bottom = b[:, 2]
    b_right = b[:, 3]
    a_w = T.abs(a_right - a_left)
    a_h = T.abs(a_bottom - a_top)
    b_w = T.abs(b_right - b_left)
    b_h = T.abs(b_bottom - b_top)
    inter_top = T.max(a_top, b_top)
    inter_left = T.max(a_left, b_left)
    inter_bottom = T.min(a_bottom, b_bottom)
    inter_right = T.min(a_right, b_right)
    inter_w = T.abs(inter_right - inter_left)
    inter_h = T.abs(inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    a_area = a_w * a_h
    b_area = b_w * b_h
    union_area = a_area + b_area - inter_area

    inter_area1 = inter_area * (1 - (union_area == 0).float())
    union_area1 = union_area + (union_area == 0).float()
    return inter_area1 / union_area1

def compute_acc(cls_out, cls_gt, c):
    cls_pred = cls_out.max(1)[1]
    cls_gt = cls_gt.long()
    acc = ((cls_pred == cls_gt).float() * (1 - c)).sum() / (1 - c).sum()
    return acc

def crop_bilinear(x, b, size):
    pre_chan_size = x.size()[:-3]
    nsamples = NP.prod(pre_chan_size)
    nchan, xrow, xcol = x.size()[-3:]
    crow, ccol = size

    x = x.view(nsamples, nchan, xrow, xcol)
    b = b.view(nsamples, 4)

    cx = (b[:, 1] + b[:, 3]) / 2.
    cy = (b[:, 0] + b[:, 2]) / 2.
    w = T.abs(b[:, 3] - b[:, 1])
    h = T.abs(b[:, 2] - b[:, 0])
    dx = w / (ccol - 1.)
    dy = h / (crow - 1.)

    cx = cx.unsqueeze(1)
    cy = cy.unsqueeze(1)
    dx = dx.unsqueeze(1)
    dy = dy.unsqueeze(1)

    ca = tovar(T.arange(0., ccol * 1.))
    cb = tovar(T.arange(0., crow * 1.))

    mx = cx + dx * (ca.unsqueeze(0) - (ccol - 1.) / 2.)
    my = cy + dy * (cb.unsqueeze(0) - (crow - 1.) / 2.)

    a = tovar(T.arange(0., xcol * 1.))
    b = tovar(T.arange(0., xrow * 1.))

    ax = T.clamp(1 - T.abs(a.view(1, -1, 1) - mx.unsqueeze(1)), min=0)
    ax = ax.unsqueeze(1).repeat(1, nchan, 1, 1).view(-1, xcol, ccol)
    by = T.clamp(1 - T.abs(b.view(1, -1, 1) - my.unsqueeze(1)), min=0)
    by = by.unsqueeze(1).repeat(1, nchan, 1, 1).view(-1, xrow, crow)

    bilin = T.bmm(by.permute(0, 2, 1), T.bmm(x.view(-1, xrow, xcol), ax))

    return bilin.view(*(list(pre_chan_size) + [nchan, crow, ccol]))


class GlobalAvgPool2d(NN.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return x.mean(3, keepdim=True).mean(2, keepdim=True)


class Model(NN.Module):
    def __init__(self, k):
        super(Model, self).__init__()
        self.sq = squeezenet1_1(pretrained=True)
        for p in self.sq.parameters():
            p.requires_grad = False
        self.classifier = NN.Sequential(
                *(list(self.sq.classifier.children())[:-1] + [GlobalAvgPool2d()]))
        self.k = k

        self.masking = NN.Sequential(
                NN.Conv2d(1000 + 512, 256, 3, padding=1),
                NN.Conv2d(256, 256, 3, padding=1),
                NN.ReLU(),
                NN.Conv2d(256, 128, 3, padding=1),
                NN.Conv2d(128, 128, 3, padding=1),
                NN.ReLU(),
                NN.Conv2d(128, 64, 3, padding=1),
                NN.Conv2d(64, 64, 3, padding=1),
                NN.ReLU(),
                )
        self.b_decision = NN.Sequential(
                NN.Linear(64 * 13 * 13, 2000),
                NN.ReLU(),
                NN.Linear(2000, 2000),
                NN.ReLU(),
                NN.Linear(2000, 4),
                )
        self.cam_picker = NN.LSTM(1000, 1000)

        with open('imagenet_labels2', 'rb') as f:
            self.labels = pickle.load(f)

    def train(self, mode=True):
        super(Model, self).train(mode)
        self.sq.eval()

    def generate_search_area(self, b, size):
        batch_size = b.size()[0]
        cx = (b[:, 1] + b[:, 3]) / 2
        cy = (b[:, 0] + b[:, 2]) / 2
        w = T.abs(b[:, 3] - b[:, 1])
        h = T.abs(b[:, 2] - b[:, 0])
        rows, cols = size

        new_w = w * (tovar(RNG.uniform(1.5, 2, (batch_size,))) if self.training else 2)
        new_h = h * (tovar(RNG.uniform(1.5, 2, (batch_size,))) if self.training else 2)

        cx = cx + w * (tovar(RNG.uniform(-0.1, 0.1, (batch_size,))) if self.training else 0)
        cy = cy + h * (tovar(RNG.uniform(-0.1, 0.1, (batch_size,))) if self.training else 0)

        top = T.clamp(cy - new_h / 2, 0, rows)
        left = T.clamp(cx - new_w / 2, 0, cols)
        bottom = T.clamp(cy + new_h / 2, 0, rows)
        right = T.clamp(cx + new_w / 2, 0, cols)

        return T.stack([top, left, bottom, right], 1)

    def scale_to(self, b, b_context):
        cx_context = (b_context[:, 1] + b_context[:, 3]) / 2
        cy_context = (b_context[:, 0] + b_context[:, 2]) / 2
        w_context = T.abs(b_context[:, 3] - b_context[:, 1])
        h_context = T.abs(b_context[:, 2] - b_context[:, 0])

        b = b - T.stack([cy_context, cx_context, cy_context, cx_context], 1)
        b = b / (T.stack([h_context, w_context, h_context, w_context], 1) / 2)

        return b

    def scale_from(self, b, b_context):
        cx_context = (b_context[:, 1] + b_context[:, 3]) / 2
        cy_context = (b_context[:, 0] + b_context[:, 2]) / 2
        w_context = T.abs(b_context[:, 3] - b_context[:, 1])
        h_context = T.abs(b_context[:, 2] - b_context[:, 0])

        b = b * T.stack([h_context, w_context, h_context, w_context], 1) / 2
        b = b + T.stack([cy_context, cx_context, cy_context, cx_context], 1)

        return b

    def pi_mix(self, pi_old, pi_new, *args, **kwargs):
        return pi_new

    def cam(self, phi):
        batch_size = phi.size()[0]
        params = list(self.sq.parameters())
        w = T.squeeze(params[-2])

        w = w.unsqueeze(0).expand(batch_size, 1000, 512)
        return w.bmm(phi.view(batch_size, 512, -1)).view(batch_size, 1000, 13, 13)

    def get_top_classes(self, pi):
        pi_tops, pi_indices = pi.topk(self.k, 1, sorted=True)
        pi_tops = F.softmax(pi_tops)
        if not self.training:
            pi_indices_np = tonumpy(pi_indices[0])
            cls_t_tops = NP.array([[self.labels[i] for i in s] for s in pi_indices_np])
        else:
            cls_t_tops = None
        return pi_tops, pi_indices, cls_t_tops

    def forward(self, x, target, b, batch_size=16):
        '''
        x: (nframes, nchannels, rows, cols)
        target: (nchannels, rows', cols')
        b: (nframes, 4) ground truth bbox unscaled
        '''

        mean = T.autograd.Variable(T.Tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1))
        std = T.autograd.Variable(T.Tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1))
        nframes, nchannels, rows, cols = x.size()
        phi = self.sq.features((target.unsqueeze(0) - mean) / std)
        pi_t_0 = pi_t = self.classifier(phi).squeeze(3).squeeze(2)
        self.pi_t_0_tops, self.pi_t_0_indices, self.cls_t_0_tops = tonumpy(*self.get_top_classes(pi_t_0))
        b_t_1 = b[0].unsqueeze(0).expand(batch_size, 4)
        b_list = []
        loss = 0
        self.m_t_topk = []
        self.pi_t_topk = []
        self.cls_t_topk = []
        self.p_t_list = []
        self.s_t_list = []

        for t in range(nframes):
            x_t = x[t]

            s_t = self.generate_search_area(b_t_1, (rows, cols))
            p_t = crop_bilinear(
                    x_t.unsqueeze(0).expand(batch_size, nchannels, rows, cols).contiguous(),
                    s_t,
                    (224, 224),
                    )
            phi_t = self.sq.features((p_t - mean) / std)
            pi_t_new = self.classifier(phi_t).squeeze(3).squeeze(2)
            pi_t = self.pi_mix(pi_t, pi_t_new)

            m_t_all = self.cam(phi_t)
            pi_t_tops, pi_t_indices, cls_t_tops = self.get_top_classes(pi_t)
            pi_t_trunc = T.autograd.Variable(T.zeros(*pi_t.size())).cuda()
            pi_t_trunc = pi_t_trunc.scatter_(1, pi_t_indices, pi_t_tops)
            m_t = m_t_all * pi_t_trunc[:, :, NP.newaxis, NP.newaxis].expand(batch_size, 1000, 13, 13)

            if not self.training:
                m_t_indices_0 = T.arange(0, batch_size).cuda().long().unsqueeze(1).expand(batch_size, self.k)
                m_t_indices_0 = m_t_indices_0.contiguous().view(-1)
                m_t_indices_1 = pi_t_indices.view(-1).data
                m_t_gathered = m_t[m_t_indices_0, m_t_indices_1].view(batch_size, self.k, 13, 13)
            else:
                m_t_gathered = None

            b_input = self.masking(T.cat([m_t, phi_t], 1))
            b_t_1 = self.scale_from(self.b_decision(b_input.view(batch_size, -1)), s_t)
            b_list.append(b_t_1)

            loss += T.abs(b_t_1 - b[t]).mean()

            self.m_t_topk.append(tonumpy(m_t_gathered))
            self.pi_t_topk.append(tonumpy(pi_t_tops))
            self.cls_t_topk.append(tonumpy(cls_t_tops))
            self.p_t_list.append(tonumpy(p_t))
            self.s_t_list.append(tonumpy(s_t))

        return T.stack(b_list, 1), loss
