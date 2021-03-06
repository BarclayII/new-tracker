
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torch.nn.init as INIT
from torchvision.models import squeezenet1_0

import pickle
import os
import matplotlib
if not os.getenv('APPDEBUG', None):
    matplotlib.use('Agg')
nocuda = bool(os.getenv('NOCUDA', None))
import matplotlib.pyplot as PL
import numpy as NP
import numpy.random as RNG

from util import addbox

def tovar(*arrs, **kwargs):
    tensors = [(T.FloatTensor(NP.array(a).astype('float32')) if not T.is_tensor(a) and not isinstance(a, T.autograd.Variable) else a.float())
               for a in arrs]
    vars_ = [T.autograd.Variable(t) for t in tensors]
    if not nocuda:
        vars_ = [v.cuda() for v in vars_]
    if len(arrs) == 1:
        return vars_[0]
    else:
        return vars_

def tonumpy(*vars_):
    result = []
    for v in vars_:
        if isinstance(v, T.autograd.Variable):
            result.append(v.data.cpu().numpy())
        elif T.is_tensor(v):
            result.append(v.cpu().numpy())
        else:
            result.append(v)
    if len(vars_) == 1:
        return result[0]
    else:
        return result

def safe_log_sigmoid(x, boundary=-80):
    x_bound = (x <= boundary).float()
    x_safe = x * (1 - x_bound)
    y = F.logsigmoid(x_safe)
    return y * (1 - x_bound) + x * x_bound

def compute_iou(a, b):
    a_top = a[..., 0]
    a_left = a[..., 1]
    a_bottom = a[..., 2]
    a_right = a[..., 3]
    b_top = b[..., 0]
    b_left = b[..., 1]
    b_bottom = b[..., 2]
    b_right = b[..., 3]
    a_w = T.abs(a_right - a_left)
    a_h = T.abs(a_bottom - a_top)
    b_w = T.abs(b_right - b_left)
    b_h = T.abs(b_bottom - b_top)
    inter_top = T.max(T.min(a_top, a_bottom), T.min(b_top, b_bottom))
    inter_left = T.max(T.min(a_left, a_right), T.min(b_left, b_right))
    inter_bottom = T.min(T.max(a_bottom, a_top), T.max(b_bottom, b_top))
    inter_right = T.min(T.max(a_right, a_left), T.max(b_right, b_left))
    inter_w = T.abs(inter_right - inter_left)
    inter_h = T.abs(inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    a_area = a_w * a_h
    b_area = b_w * b_h
    union_area = a_area + b_area - inter_area

    inter_area1 = inter_area * (1 - (union_area == 0).float())
    union_area1 = union_area + (union_area == 0).float()
    return inter_area1 / union_area1

def compute_dev(a, b):
    a_cx = (a[..., 1] + a[..., 3]) / 2
    a_cy = (a[..., 0] + a[..., 2]) / 2
    a_w = T.abs(a[..., 3] - a[..., 1])
    a_h = T.abs(a[..., 2] - a[..., 0])
    b_cx = (b[..., 1] + b[..., 3]) / 2
    b_cy = (b[..., 0] + b[..., 2]) / 2
    b_w = T.abs(b[..., 3] - b[..., 1])
    b_h = T.abs(b[..., 2] - b[..., 0])

    d_cx = a_cx - b_cx
    d_cy = a_cy - b_cy
    d_pos = T.sqrt(d_cx ** 2 + d_cy ** 2)
    # The point is to make the size ratio of 1/1.1 and 1.1 to have the same metric
    d_size = T.abs(a_w.log() + a_h.log() - b_w.log() - b_h.log())
    d_w_scale = T.abs(d_cx) / a_w
    d_h_scale = T.abs(d_cy) / a_h

    return d_pos, d_size, d_w_scale, d_h_scale

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

def init_lstm(lstm):
    for name, param in lstm.named_parameters():
        if name.startswith('weight_ih'):
            INIT.xavier_uniform(param.data)
        elif name.startswith('weight_hh'):
            INIT.orthogonal(param.data)
        elif name.startswith('bias'):
            INIT.constant(param.data, 0)

def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            INIT.xavier_uniform(param.data)
        elif name.find('bias') != -1:
            INIT.constant(param.data, 0.1)


class GlobalAvgPool2d(NN.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return x.mean(3, keepdim=True).mean(2, keepdim=True)


class Model(NN.Module):
    def __init__(self, k, rnn_enabled=False):
        super(Model, self).__init__()
        self.k = k
        self.rnn_enabled = rnn_enabled

        self.cam_compressor = NN.Sequential(
                NN.Conv2d(1000, 256, 1),
                )
        self.feat_compressor = NN.Sequential(
                NN.Conv2d(512, 256, 1),
                )

        self.masking = NN.Sequential(
                NN.Conv2d(256 * 2, 256, 3, padding=1),
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
        self.cam_picker_rnn = NN.LSTMCell(1000, 1000)
        self.cam_picker_feedfwd = NN.Sequential(
                NN.Linear(1000, 1000),
                NN.LogSoftmax(),
                )

        with open('imagenet_labels2', 'rb') as f:
            self.labels = pickle.load(f)

        self.teaching = True

        for m in self.children():
            if isinstance(m, NN.LSTM):
                init_lstm(m)
            elif isinstance(m, NN.Module):
                init_weights(m)

        self.sq = squeezenet1_0(pretrained=True)
        for p in self.sq.parameters():
            p.requires_grad = False
        self.classifier = NN.Sequential(
                *(list(self.sq.classifier.children())[:-1] + [GlobalAvgPool2d()]))

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

        new_w = w * (tovar(RNG.uniform(1.5, 2.5, (batch_size,))) if self.training else 2)
        new_h = h * (tovar(RNG.uniform(1.5, 2.5, (batch_size,))) if self.training else 2)

        w_shift = (new_w - w) / 2
        h_shift = (new_h - h) / 2

        cx = cx + (tovar(RNG.uniform(-1, 1, (batch_size,))) if self.training else 0) * w_shift
        cy = cy + (tovar(RNG.uniform(-1, 1, (batch_size,))) if self.training else 0) * h_shift

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

    def pi_mix(self, pi_new, state, *args, **kwargs):
        if self.rnn_enabled:
            h, c = self.cam_picker_rnn(pi_new, state)
            pi_new = pi_new + self.cam_picker_feedfwd(h)
            return pi_new, (h, c)
        else:
            return pi_new, state

    def cam(self, phi):
        batch_size = phi.size()[0]
        params = list(self.sq.parameters())
        w = T.squeeze(params[-2])

        w = w.unsqueeze(0).expand(batch_size, 1000, 512)
        return w.bmm(phi.view(batch_size, 512, -1)).view(batch_size, 1000, 13, 13)

    @property
    def visualize(self):
        #return not self.training
        return True

    def get_top_classes(self, pi):
        pi_tops, pi_indices = F.softmax(pi).topk(self.k, 1, sorted=True)
        if self.visualize:
            pi_indices_np = tonumpy(pi_indices)
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

        mean = tovar(T.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        std = tovar(T.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        nframes, nchannels, rows, cols = x.size()
        phi = self.sq.features((target.unsqueeze(0) - mean) / std)
        pi_t = self.classifier(phi).squeeze(3).squeeze(2)

        lstm_state = (
                tovar(T.zeros(batch_size, 1000)),
                tovar(T.zeros(batch_size, 1000)),
                )
        pi_t, lstm_state = self.pi_mix(pi_t.expand(batch_size, 1000), lstm_state)
        pi_t_0 = pi_t

        self.pi_t_0_tops, self.pi_t_0_indices, self.cls_t_0_tops = tonumpy(*self.get_top_classes(pi_t_0))
        b_t_1 = b[0].unsqueeze(0).expand(batch_size, 4)
        b_list = []
        b_internal_list = []
        b_internal_gt_list = []
        loss = 0
        self.m_t_topk = []
        self.pi_t_topk = []
        self.cls_t_topk = []
        self.p_t_list = []
        self.s_t_list = []

        for t in range(nframes):
            x_t = x[t]
            if self.teaching:
                b_t_1 = b[max(t-1, 0)].unsqueeze(0).expand(batch_size, 4)

            s_t = self.generate_search_area(b_t_1, (rows, cols))
            p_t = crop_bilinear(
                    x_t.unsqueeze(0).expand(batch_size, nchannels, rows, cols).contiguous(),
                    s_t,
                    (224, 224),
                    )
            phi_t = self.sq.features((p_t - mean) / std)
            pi_t_new = self.classifier(phi_t).squeeze(3).squeeze(2)
            pi_t, lstm_state = self.pi_mix(pi_t_new, lstm_state)

            m_t_all = self.cam(phi_t)
            pi_t_tops, pi_t_indices, cls_t_tops = self.get_top_classes(pi_t)
            pi_t_trunc = tovar(T.zeros(*pi_t.size()))
            pi_t_trunc = pi_t_trunc.scatter_(1, pi_t_indices, pi_t_tops)
            m_t = m_t_all * pi_t_trunc[:, :, NP.newaxis, NP.newaxis].expand(batch_size, 1000, 13, 13)

            if self.visualize:
                m_t_indices_0 = tovar(T.arange(0, batch_size)).data
                m_t_indices_0 = m_t_indices_0.long().unsqueeze(1).expand(batch_size, self.k)
                m_t_indices_0 = m_t_indices_0.contiguous().view(-1)
                m_t_indices_1 = pi_t_indices.view(-1).data
                m_t_gathered = m_t[m_t_indices_0, m_t_indices_1].view(batch_size, self.k, 13, 13)
            else:
                m_t_gathered = None

            m_t_compressed = self.cam_compressor(m_t)
            phi_t_compressed = self.feat_compressor(phi_t)
            b_input = self.masking(T.cat([m_t_compressed, phi_t_compressed], 1))
            b_hat = self.b_decision(b_input.view(batch_size, -1))
            b_t_1 = self.scale_from(b_hat, s_t)
            b_list.append(b_t_1)
            b_internal_list.append(b_hat)
            b_internal_gt_list.append(self.scale_to(b[t].unsqueeze(0), s_t))

            loss += (T.abs(b_hat - self.scale_to(b[t].unsqueeze(0), s_t))).mean()

            self.m_t_topk.append(tonumpy(m_t_gathered))
            self.pi_t_topk.append(tonumpy(pi_t_tops))
            self.cls_t_topk.append(tonumpy(cls_t_tops))
            self.p_t_list.append(tonumpy(p_t))
            self.s_t_list.append(tonumpy(s_t))

        b_hat = T.stack(b_list, 1)
        b_internal = T.stack(b_internal_list, 1)
        b_internal_gt = T.stack(b_internal_gt_list, 1)
        b_gt = b.unsqueeze(0).expand(batch_size, nframes, 4)
        return b_hat, b_internal, b_internal_gt, loss, b_gt
