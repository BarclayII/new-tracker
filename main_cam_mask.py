
import modelth_cam_pretrain_new as model
import dataset
from timer import Timer
import sys
import numpy as NP
import os
import pickle

os.system("taskset -p 0xffffffff %d" % os.getpid())

data = dataset.ImageNetVidDataset('/beegfs/qg323/ILSVRC', 'map_vid.txt')
model.create_model(len(data.classes))
chk = 0 if len(sys.argv) <= 1 else int(sys.argv[1]) - 1
if chk != 0:
    model.load('/scratch/qg323/model_mask%d' % chk)

def gen(data, train, return_prenorm=False):
    #data.spinup()
    while True:
        result = dataset.prepare_batch(
                4, 4, data, 15, 5, -0.4, 0.4,
                resize=224, swapdims=False,
                train=train, random=train,
                )
        #(px, pp), (b, c, cls) = data.prepare_batch()
        px, pp, b, c, cls = result
        x = px
        p = pp
        if return_prenorm:
            yield ([x, p], [b, c, cls], [px, pp])
        else:
            yield ([x, p], [b, c, cls])

valid_gen = gen(data, False, True)
val_x = []
val_p = []
val_b = []
val_c = []
val_cls = []
val_orig_x = []
val_orig_p = []
val_clsname = []
for i in range(100):
    print 'Generating validation %d' % i
    (x, p), (b, c, cls), (px, pp) = valid_gen.next()
    val_x.extend(x)
    val_p.extend(p)
    val_b.extend(b)
    val_c.extend(c)
    val_cls.extend(cls)
    val_orig_x.extend(px)
    val_orig_p.extend(pp)
    val_clsname.extend([data.clsdic[data.classes[_cls]] for _cls in cls])
ad.terminate()
val_x = NP.array(val_x)
val_p = NP.array(val_p)
val_b = NP.array(val_b)
val_c = NP.array(val_c)
val_cls = NP.array(val_cls)
val_orig_x = NP.array(val_orig_x)
val_orig_p = NP.array(val_orig_p)

train_gen = gen(data, True, False)
print 'Training'
try:
    model.train_on(train_gen, 30, 2000,
                   ([val_x, val_p], [val_b, val_c, val_cls]),
                   '/beegfs/qg323/model_mask_%03d',
                   [val_orig_x, val_orig_p, val_clsname, data])
except:
    raise
finally:
    ad.terminate()
