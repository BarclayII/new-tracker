
import dataset
from modelth_cam_rnn import Model, tovar
import torch as T
import os
import matplotlib
if os.getenv('APPDEBUG', None):
    matplotlib.use('Agg')
import matplotlib.pyplot as PL

model = Model(5).cuda()
data = dataset.ImageNetVidDataset('/beegfs/qg323/ILSVRC', 'map_vid.txt')

opt = T.optim.Adam(p for p in model.parameters() if p.requires_grad)

valid_set = []

epoch = 0
valid_size = 5
train_iter = 5

for _ in range(valid_size):
    result = dataset.prepare_batch(1, 1, data, 15, 5, -0.4, 0.4, resize=None, swapdims=False, train=False,
            random=False, target_resize=False)
    px, pp, b, cls = result
    px = px[0]
    pp = pp[0]
    b = b[0]
    valid_set.append((px, pp, b))

model.train()
for _ in range(train_iter):
    result = dataset.prepare_batch(1, 1, data, 15, 5, -0.4, 0.4, resize=None, swapdims=False, train=True,
            random=False, target_resize=False)

    px, pp, b, cls = result
    px = px[0]
    pp = pp[0]
    b = b[0]

    px = tovar(px).permute(0, 3, 1, 2)
    pp = tovar(pp).permute(2, 0, 1)
    b = tovar(b)
    _, loss = model.forward(px, pp, b)
    opt.zero_grad()
    loss.backward()
    opt.step()

def figure_title(cls, pi):
    cls = cls.split(',')[0]
    return '%s/%.5f' % (cls, pi)

model.eval()
for i in range(valid_size):
    px, pp, b = valid_set[i]
    x = px[0]
    p = pp
    px = tovar(px).permute(0, 3, 1, 2)
    pp = tovar(pp).permute(2, 0, 1)
    b = tovar(b)

    b_list, loss = model.forward(px, pp, b, 1)

    for t in range(5):
        fig, ax = PL.subplots(2, 4)
        ax[0][0].imshow(x[:, :, ::-1])
        ax[0][1].imshow(p[:, :, ::-1])
        ax[0][2].imshow(model.p_t_list[t][0].transpose(1, 2, 0)[:, :, ::-1])
        ax[0][2].set_title(figure_title(model.cls_t_0_tops[0, 0], model.pi_t_0_tops[0, 0]), fontsize=6)
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
