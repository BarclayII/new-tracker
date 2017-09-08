
import torch as T
import torch.nn as NN
import torch.nn.functional as F
import torchvision.models as TVM
import torchvision as TV
import numpy as NP
import scipy.misc as SPM

import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as PL
import matplotlib.patches as PA

def tovar(*arrs, **kwargs):
    tensors = [(T.FloatTensor(NP.array(a).astype('float32')) if not T.is_tensor(a) and not isinstance(a, T.autograd.Variable) else a.float())
               for a in arrs]
    return [T.autograd.Variable(t).cuda() for t in tensors]

def tonumpy(*vars_):
    return [v.data.cpu().numpy() for v in vars_]

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
    a_w = T.clamp(a_right - a_left, min=0)
    a_h = T.clamp(a_bottom - a_top, min=0)
    b_w = T.clamp(b_right - b_left, min=0)
    b_h = T.clamp(b_bottom - b_top, min=0)
    inter_top = T.max(a_top, b_top)
    inter_left = T.max(a_left, b_left)
    inter_bottom = T.min(a_bottom, b_bottom)
    inter_right = T.min(a_right, b_right)
    inter_w = T.clamp(inter_right - inter_left, min=0)
    inter_h = T.clamp(inter_bottom - inter_top, min=0)
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

class Model(NN.Module):
    def __init__(self, k):
        super(Model, self).__init__()
        self.sq = TVM.squeezenet1_1(pretrained=True)
        self.k = k

        self.b_decision = NN.Sequential(
                NN.Linear(self.k * 13 * 13, 200),
                NN.ReLU(),
                NN.Linear(200, 200),
                NN.ReLU(),
                NN.Linear(200, 4),
                )
        self.c_decision = NN.Sequential(
            #NN.Linear(512 * 13 * 13 * 2, 2000),
            NN.Linear(512 * 4, 2000),
            NN.ReLU(),
            NN.Linear(2000, 2000),
            NN.ReLU(),
            NN.Linear(2000, 1),
            )

    def forward(self, x, p, b_gt=None, c_gt=None):
        batch_size = x.size()[0]

        p_out = self.sq.features(p)
        p_cls = self.sq.classifier(p_out).squeeze(3).squeeze(2)

        params = list(self.sq.parameters())
        w = T.squeeze(params[-2])
        top_k_indices = T.sort(p_cls, 1, descending=True)[1][:, :self.k]
        
        w = w.unsqueeze(0).expand(batch_size, 1000, 512)
        top_k_indices_expand = top_k_indices.unsqueeze(2).expand(batch_size, self.k, 512)
        w = w.gather(1, top_k_indices_expand)

        x_out = self.sq.features(x)
        _, _, height, width = x_out.size()
        x_cam = w.bmm(x_out.view(batch_size, 512, -1)).view(batch_size, self.k, 13, 13)

        # x has black padding and its scale is different from p.  I wonder
        # if I can use a convnet to do confidence regression (probably using
        # global max-pooling?)
        p_mean = p_out.mean(3).mean(2)
        x_mean = x_out.mean(3).mean(2)
        p_max = p_out.max(3)[0].max(2)[0]
        x_max = x_out.max(3)[0].max(2)[0]
        '''
        c_in = T.cat([
            p_out.view(batch_size, -1),
            x_out.view(batch_size, -1),
            ], 1)
        '''
        c_in = T.cat([p_mean, x_mean, p_max, x_max], 1)
        c_out = safe_log_sigmoid(self.c_decision(c_in))
        b_out = self.b_decision(x_cam.view(batch_size, -1))

        if not (b_gt is None or c_gt is None):
            c_gt = c_gt.unsqueeze(1)
            c_loss = F.binary_cross_entropy_with_logits(c_out, c_gt)
            b_loss = ((b_gt - b_out) ** 2 * (1 - c_gt)).sum()
        else:
            b_loss = c_loss = None

        return b_out, c_out, x_cam, b_loss, c_loss, top_k_indices

def norm_batch(x, mean, std):
    mean = T.FloatTensor(mean).cuda().view(1, 3, 1, 1)
    std = T.FloatTensor(std).cuda().view(1, 3, 1, 1)
    return (T.FloatTensor(x).cuda() - mean) / std

def load(path):
    global model
    model = T.load(path)

def val_func(x, p, b, c, cls):
    model.eval()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    x = norm_batch(x.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
    p = norm_batch(p.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
    x, p, b, c = tovar(x, p, b, c)
    b_out, c_out, x_cam, b_l, c_l, topk = model.forward(x, p, b, c)
    return tonumpy(b_out, topk, x_cam)

model = None
def create_model(n_classes):
    global model
    model = Model(5).cuda()

def train_on(datagen, epochs, steps_per_epoch, val, path, orig):
    global model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    with open('imagenet_labels2') as f:
        loc_clsname = pickle.load(f)

    if not os.path.exists(path + '-train'):
        os.mkdir(path + '-train')
    if not os.path.exists(path + '-valid'):
        os.mkdir(path + '-valid')
    #opt = T.optim.SGD(model.parameters(), lr=1e-5)
    opt = T.optim.Adam(model.parameters())
    #'''
    for p in model.sq.parameters():
        p.requires_grad = False
    #'''
    for epoch in range(epochs):

        model.train()
        for step in range(steps_per_epoch):
            (x, p), (b, c, cls) = datagen.next()
            orig_x = x
            orig_p = p
            x = norm_batch(x.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
            p = norm_batch(p.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
            x, p, b, c = tovar(x, p, b, c)
            opt.zero_grad()
            b_out, c_out, x_cam, b_l, c_l, topk = model.forward(x, p, b, c)
            iou = compute_iou(b_out, b)
            all_l = b_l + c_l
            all_l.backward()

            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.data
                anynan = (g != g).long().sum()
                anybig = (g.abs() > 1e+5).long().sum()
                assert anynan == 0
                assert anybig == 0
            opt.step()
            iou, all_l, b_l, c_l, c = tonumpy(iou, all_l, b_l, c_l, c)
            print '#', epoch, step, iou.mean(), all_l, b_l, c_l

        model.eval()
        (x, p), (bt, ct, cls) = val[:5]
        x = norm_batch(x.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
        p = norm_batch(p.transpose([0, 3, 1, 2])[:, ::-1], mean, std)
        val_batch_size = 16
        avg_iou = 0
        avg_acc = 0
        orig_x, orig_p, clsname, data = orig
        for _id in range(0, x.shape[0], val_batch_size):
            _x = x[_id:_id + val_batch_size]
            _p = p[_id:_id + val_batch_size]
            _bt = bt[_id:_id + val_batch_size]
            _ct = ct[_id:_id + val_batch_size]
            _clsgt = cls[_id:_id + val_batch_size]

            _x, _p, _bt, _ct = tovar(_x, _p, _bt, _ct)
            _b, _c, cam, _, _, topk = model.forward(_x, _p)

            vgg_cls_out = model.sq(_p).data.squeeze().cpu().numpy()
            vgg_cls_result = [loc_clsname[NP.argmax(_)] for _ in vgg_cls_out]

            iou = compute_iou(_b, _bt)
            _b, _c, cam, iou, topk, _bt = tonumpy(_b, _c, cam, iou, topk, _bt)
            cam = cam.transpose([0, 2, 3, 1])
            if _id > 80:
                continue

            for i in range(val_batch_size):
                fig, ax = PL.subplots(2, 4, figsize=(8, 4))
                ax[0, 0].imshow(orig_x[_id + i][:, :, ::-1])
                ax[0, 1].imshow(orig_p[_id + i][:, :, ::-1])
                b = (_b[i] + 1) / 2. * 224
                B = (_bt[i] + 1) / 2. * 224
                ax[0, 0].add_patch(
                        PA.Rectangle(
                            (b[1], b[0]),
                            b[3] - b[1],
                            b[2] - b[0],
                            ec='red',
                            fill=False,
                            lw=1
                            )
                        )
                ax[0, 0].add_patch(
                        PA.Rectangle(
                            (B[1], B[0]),
                            B[3] - B[1],
                            B[2] - B[0],
                            ec='yellow',
                            fill=False,
                            lw=1
                            )
                        )
                ax[0, 2].imshow(vgg_cls_out[i:i+1])
                ax[0, 2].set_title(
                        clsname[i] + '/' +
                        vgg_cls_result[i],
                        fontsize=8,
                        )

                cam_classes = [loc_clsname[_] for _ in topk[i]]
                ax[0, 3].imshow(cam[i, :, :, 0])
                ax[0, 3].set_title(cam_classes[0], fontsize=8)
                [ax[1, _].imshow(cam[i, :, :, _+1]) for _ in range(4)]
                [ax[1, _].set_title(cam_classes[_ + 1], fontsize=6) for _ in range(4)]
                if not os.path.exists(path):
                    os.mkdir(path)
                fig.savefig('%s/%05d-%02d.png' % (path, epoch, i + _id))
                PL.close(fig)
            avg_iou = ((avg_iou * (_id // val_batch_size)) + iou.mean()) / (_id // val_batch_size + 1)
        print '@', epoch, avg_iou, avg_acc
        T.save(model, path % epoch)
