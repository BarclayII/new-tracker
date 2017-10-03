
import tarfile
import lxml.etree as ETREE
import numpy as NP
import numpy.random as RNG
import cv2
import os
import matplotlib
if not os.getenv('APPDEBUG', None):
    matplotlib.use('Agg')
import matplotlib.pyplot as PL

from util import addbox
from example_generator import *

class DictList(dict):
    def append_to(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)

class VideoDataset(object):
    def __init__(self):
        pass

    def pick(self, train=True):
        pass

_parser = ETREE.XMLParser(remove_blank_text=True)

class ALOVDataset(VideoDataset):
    def __init__(self, dir_):
        super(ALOVDataset, self).__init__()
        self.dir_ = dir_
        self.imagedata = dir_ + '/imagedata++'
        self.annodata = dir_ + '/alov300++_rectangleAnnotation_full'

        self.videos = []

        for rootdir, subdirs, _ in os.walk(self.annodata):
            for category in subdirs:
                _, _, files = os.walk(os.path.join(self.annodata, category)).next()
                video_dirs = [category + '/' + f[:-4] for f in files]
                self.videos.extend(video_dirs)

    def pick(self, train=True):
        pass

    def pick_video(self, i, train=True):
        video = self.videos[i]
        imagedir = self.imagedata + '/' + video
        annofile = self.annodata + '/' + video + '.ann'
        imgs = []
        nfiles = len(os.walk(imagedir).next()[2])

        for t in range(nfiles):
            imgs.append(cv2.imread(imagedir + '/%08d.jpg' % (t + 1)))

        imgs = NP.array(imgs)
        bboxes = []
        annos = []
        with open(annofile) as a:
            for l in a:
                x = [float(_) for _ in l.strip().split()]
                x[0] = int(x[0])
                annos.append(x)

        def _get_bbox(anno):
            anno_t, x0, y0, x1, y1, x2, y2, x3, y3 = anno
            return anno_t, min(y0, y1, y2, y3), min(x0, x1, x2, x3), max(y0, y1, y2, y3), max(x0, x1, x2, x3)
        anno_0, ymin, xmin, ymax, xmax = _get_bbox(annos[0])
        anno_1, _, _, _, _ = _get_bbox(annos[-1])
        bboxes.append([ymin, xmin, ymax, xmax])
        for p in range(1, len(annos)):
            anno_t, ymin, xmin, ymax, xmax = _get_bbox(annos[p - 1])
            anno_t1, ymin1, xmin1, ymax1, xmax1 = _get_bbox(annos[p])
            xmaxs = NP.linspace(xmax, xmax1, anno_t1 - anno_t + 1)[1:]
            ymins = NP.linspace(ymin, ymin1, anno_t1 - anno_t + 1)[1:]
            xmins = NP.linspace(xmin, xmin1, anno_t1 - anno_t + 1)[1:]
            ymaxs = NP.linspace(ymax, ymax1, anno_t1 - anno_t + 1)[1:]
            bboxes.extend(zip(ymins, xmins, ymaxs, xmaxs))

        bboxes = NP.array(bboxes)
        imgs = imgs[anno_0-1:anno_1]
        return imgs, bboxes, 0

class ImageNetVidDatasetBase(VideoDataset):
    def __init__(self):
        self.train_annotations = DictList()
        self.val_annotations = DictList()
        self.train_data = DictList()
        self.val_data = DictList()

    def _addfile(self, file_):
        nl = file_.split('/')
        dirname = '/'.join(nl[:-1])
        if nl[-6] == 'Data' and nl[-4] == 'train':
            self.train_data.append_to(dirname, file_)
        elif nl[-6] == 'Annotations' and nl[-4] == 'train':
            self.train_annotations.append_to(dirname, file_)
        elif nl[-5] == 'Data' and nl[-3] == 'val':
            self.val_data.append_to(dirname, file_)
        elif nl[-5] == 'Annotations' and nl[-3] == 'val':
            self.val_annotations.append_to(dirname, file_)

    def _check_integrity(self):
        print 'Checking integrity...'

        train_classes = set()
        val_classes = set()
        for cls, dl in zip((train_classes, val_classes), (self.train_annotations, self.val_annotations)):
            for dirname in dl:
                for filename in dl[dirname]:
                    doc = ETREE.parse(filename, _parser)
                    root = doc.getroot()
                    objs = root.findall('object')
                    cls |= set(obj.find('name').text for obj in objs)
        self.classes = list(train_classes | val_classes)
        print 'Number of classes: %d' % len(self.classes)

        self.train_annotations = {k: len(self.train_annotations[k]) for k in self.train_annotations}
        self.val_annotations = {k: len(self.val_annotations[k]) for k in self.val_annotations}
        self.train_data = {k: len(self.train_data[k]) for k in self.train_data}
        self.val_data = {k: len(self.val_data[k]) for k in self.val_data}

        self.clsdic = {}
        with open(self.dic) as f:
            for l in f:
                cls, _, name = l.strip().split()
                self.clsdic[cls] = name

    def _load_file(self, file_):
        return NotImplemented

    def pick(self, train=True, frames=5):
        anno_set = self.train_annotations if train else self.val_annotations
        anno_dir = RNG.choice(anno_set.keys())
        anno_file = anno_dir + '/000000.xml'

        anno = self._load_file(anno_file)
        anno_root = ETREE.parse(anno, parser=_parser).getroot()
        folder = anno_root.find('folder').text
        data_dir = self._getpath('Data/VID/%s/%s' % ('train' if train else 'val', folder))

        while True:
            anno_idx = RNG.choice(anno_set[anno_dir] - frames + 1)
            trackids_list = []
            anno_cur_list = []
            anno_cur_root_list = []
            objs_list = []
            data_file_list = []

            for i in range(frames):
                idx = anno_idx + i
                anno_file = anno_dir + '/%06d.xml' % idx
                data_file = data_dir + '/%06d.JPEG' % idx
                anno_cur = self._load_file(anno_file)
                anno_cur_root = ETREE.parse(anno_cur, parser=_parser).getroot()
                objs = anno_cur_root.findall('object')
                trackids = {obj.find("trackid").text: obj for obj in objs}

                trackids_list.append(trackids)
                anno_cur_list.append(anno_cur)
                anno_cur_root_list.append(anno_cur_root)
                objs_list.append(objs)
                data_file_list.append(data_file)

            trackids = reduce(lambda a, b: a & b, [set(_.keys()) for _ in trackids_list], set(trackids_list[0].keys()))
            if len(trackids) != 0:
                break
            else:
                for f in anno_cur_list:
                    f.close()

        trackid = RNG.choice(list(trackids))
        objs = [_[trackid] for _ in trackids_list]
        cls_idx = self.classes.index(objs[0].find('name').text)

        order = ['ymin', 'xmin', 'ymax', 'xmax']
        bboxes = [NP.array([int(obj.find('bndbox').find(tag).text) for tag in order]) for obj in objs]
        imgs = [cv2.imread(f) for f in data_file_list]

        #data_prev = self._load_file(data_prev_file)
        #data_next = self._load_file(data_next_file)

        anno.close()
        [f.close() for f in anno_cur_list]
        #data_prev.close()
        #data_next.close()

        return imgs, bboxes, cls_idx

    def pick_video(self, i, train=True):
        anno_set = self.train_annotations if train else self.val_annotations
        if i >= len(anno_set.keys()):
            return None, None
        anno_dir = anno_set.keys()[i]
        anno_file = anno_dir + '/000000.xml'

        anno = self._load_file(anno_file)
        anno_root = ETREE.parse(anno, parser=_parser).getroot()
        obj = anno_root.find('object')
        if obj is None:
            return NP.array([]), NP.array([]), NP.array([])
        trackid = obj.find('trackid').text
        cls_idx = self.classes.index(obj.find('name').text)
        folder = anno_root.find('folder').text
        data_dir = self._getpath('Data/VID/%s/%s' % ('train' if train else 'val', folder))
        anno.close()

        imgs = []
        bboxes = []
        order = ['ymin', 'xmin', 'ymax', 'xmax']
        for anno_idx in range(anno_set[anno_dir]):
            anno_file = anno_dir + '/%06d.xml' % anno_idx
            anno = self._load_file(anno_file)
            anno_root = ETREE.parse(anno_file, parser=_parser).getroot()
            anno.close()
            objs = anno_root.findall('object')
            if len(objs) == 0:
                break
            obj = [o for o in objs if o.find('trackid').text == trackid]
            if len(obj) > 0:
                obj = obj[0]
                bbox = NP.array([int(obj.find('bndbox').find(tag).text) for tag in order])
                data_file = data_dir + '/%06d.JPEG' % anno_idx
                img = cv2.imread(data_file)

                bboxes.append(bbox)
                imgs.append(img)
            else:
                break

        return NP.array(imgs), NP.array(bboxes), cls_idx


class ImageNetVidDataset(ImageNetVidDatasetBase):
    def __init__(self, dir_, dic):
        super(ImageNetVidDataset, self).__init__()
        self.dir_ = dir_
        self.dic = dic
        print 'Scanning directory...'

        for rootdir, subdirs, files in os.walk(dir_):
            for f in files:
                self._addfile(rootdir + '/' + f)

        self._check_integrity()

    def _load_file(self, file_):
        return open(file_)

    def _getpath(self, path):
        return self.dir_ + '/' + path


class ImageNetVidTarDataset(ImageNetVidDatasetBase):
    def __init__(self, tarball):
        super(ImageNetVidTarDataset, self).__init__()
        self.tarball = tarfile.open(tarball)
        print 'Loading archive index...'
        names = self.tarball.getnames()

        for n in names:
            self._addfile(n)

        self._check_integrity()

    def _load_file(self, file_):
        return self.tarball.extractfile(file_)

    def _getpath(self, path):
        return "ILSVRC/" + path


def prepare_batch(num_img, num_ext, dataset, lambda_scale, lambda_shift, min_scale, max_scale, train=True, resize=None,
                  swapdims=False, random=False, target_resize=True):
    images = []
    targets = []
    scaled_bboxes = []
    cls_indices = []
    negatives = []

    for j in range(num_img):
        imgs, bboxes, cls = dataset.pick()
        prev_img = imgs[0]
        prev_bbox = bboxes[0]

        target_pad, _, _, _ = crop_pad_image(prev_bbox, prev_img, padding=False, ctx_factor=1)
        target_rows, target_cols, _ = target_pad.shape
        target_scale = min(1, target_rows / 400., target_cols / 400.)
        if target_scale != 1:
            target_rows = int(target_rows / target_scale)
            target_cols = int(target_cols / target_scale)
            target_pad = cv2.resize(target_pad, (target_cols, target_rows))

        for i in range(num_ext):
            focus_imgs = []
            scaled_bbox = []
            negs = []
            for j in range(len(imgs)):
                next_img = imgs[j]
                next_bbox = bboxes[j]
                while True:
                    try:
                        focus_img, scaled_bbox_gt, _, _, _, negative = make_single_training_example(
                                next_img, prev_bbox, next_bbox, True, lambda_scale, lambda_shift, min_scale, max_scale, random=random
                                )
                        #fig, ax = PL.subplots()
                        #ax.imshow(focus_img[:, :, ::-1].astype('uint8'))
                        #scale_back = NP.array([
                        #    focus_img.shape[0],
                        #    focus_img.shape[1],
                        #    focus_img.shape[0],
                        #    focus_img.shape[1],
                        #    ])
                        #addbox(ax, (scaled_bbox_gt + 1) * scale_back / 2., 'red')
                        #PL.show()
                        break
                    except TimeoutError:
                        pass
                focus_imgs.append(next_img)
                scaled_bbox.append(next_bbox)
            images.append(focus_imgs)
            targets.append(target_pad)
            scaled_bboxes.append(scaled_bbox)
            cls_indices.append(cls)

    if resize:
        for i in range(num_img * num_ext):
            for t in range(len(images[i])):
                images[i][t] = cv2.resize(images[i][t], (resize, resize))
        images = NP.array(images)
        if swapdims:
            images = images.transpose(0, 1, 4, 2, 3)

    if target_resize and resize:
        for i in range(num_img * num_ext):
            targets[i] = cv2.resize(targets[i], (resize, resize))
        targets = NP.array(targets)
        if swapdims:
            targets = targets.transpose(0, 3, 1, 2)

    return NP.array(images) / 255., NP.array(targets) / 255., NP.array(scaled_bboxes), \
              NP.array(cls_indices, dtype='int32')
