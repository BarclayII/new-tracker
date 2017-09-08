
import tarfile
import lxml.etree as ETREE
import numpy as NP
import numpy.random as RNG
import cv2
import os

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
        return imgs, bboxes

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
        assert train_classes == val_classes
        self.classes = list(train_classes)
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

    def pick(self, train=True):
        anno_set = self.train_annotations if train else self.val_annotations
        anno_dir = RNG.choice(anno_set.keys())
        anno_file = anno_dir + '/000000.xml'

        anno = self._load_file(anno_file)
        anno_root = ETREE.parse(anno, parser=_parser).getroot()
        folder = anno_root.find('folder').text
        data_dir = self._getpath('Data/VID/%s/%s' % ('train' if train else 'val', folder))

        while True:
            anno_idx = RNG.choice(anno_set[anno_dir] - 1)

            anno_prev_file = anno_dir + '/%06d.xml' % anno_idx
            anno_next_file = anno_dir + '/%06d.xml' % (anno_idx + 1)
            data_prev_file = data_dir + '/%06d.JPEG' % anno_idx
            data_next_file = data_dir + '/%06d.JPEG' % (anno_idx + 1)

            anno_prev = self._load_file(anno_prev_file)
            anno_next = self._load_file(anno_next_file)
            anno_prev_root = ETREE.parse(anno_prev, parser=_parser).getroot()
            anno_next_root = ETREE.parse(anno_next, parser=_parser).getroot()

            prev_objs = anno_prev_root.findall('object')
            next_objs = anno_next_root.findall('object')
            prev_trackids = {obj.find('trackid').text: obj for obj in prev_objs}
            next_trackids = {obj.find('trackid').text: obj for obj in next_objs}
            trackids = set(prev_trackids.keys()) & set(next_trackids.keys())
            if len(trackids) != 0:
                break

        trackid = RNG.choice(list(trackids))
        prev_obj = prev_trackids[trackid]
        cls_idx = self.classes.index(prev_obj.find('name').text)
        next_obj = next_trackids[trackid]

        order = ['ymin', 'xmin', 'ymax', 'xmax']
        prev_bbox = NP.array([int(prev_obj.find('bndbox').find(tag).text) for tag in order])
        next_bbox = NP.array([int(next_obj.find('bndbox').find(tag).text) for tag in order])

        #data_prev = self._load_file(data_prev_file)
        #data_next = self._load_file(data_next_file)
        prev_img = cv2.imread(data_prev_file)
        next_img = cv2.imread(data_next_file)

        anno.close()
        anno_prev.close()
        anno_next.close()
        #data_prev.close()
        #data_next.close()

        return prev_img, next_img, prev_bbox, next_bbox, cls_idx

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
                  swapdims=False, random=False):
    images = []
    targets = []
    scaled_bboxes = []
    cls_indices = []
    negatives = []

    for j in range(num_img):
        prev_img, next_img, prev_bbox, next_bbox, cls = dataset.pick()
        fake_img, _, _, _, _ = dataset.pick()

        target_pad, _, _, _ = crop_pad_image(prev_bbox, prev_img, padding=False)

        for i in range(num_ext):
            while True:
                try:
                    focus_img, scaled_bbox_gt, _, _, _, negative = make_single_training_example(
                            next_img, prev_bbox, next_bbox, i != 0, lambda_scale, lambda_shift, min_scale, max_scale, random=random, img_fake=fake_img
                            )
                    break
                except TimeoutError:
                    pass
            images.append(focus_img)
            targets.append(target_pad)
            scaled_bboxes.append(scaled_bbox_gt)
            cls_indices.append(cls)
            negatives.append(negative)

    if resize is not None:
        for i in range(num_img * num_ext):
            images[i] = cv2.resize(images[i], (resize, resize))
            targets[i] = cv2.resize(targets[i], (resize, resize))
        images = NP.array(images)
        targets = NP.array(targets)
        if swapdims:
            images = images.transpose(0, 3, 1, 2)
            targets = targets.transpose(0, 3, 1, 2)

    return NP.array(images) / 255., NP.array(targets) / 255., NP.array(scaled_bboxes), \
              NP.array(negatives, dtype='float32'), NP.array(cls_indices, dtype='int32')
