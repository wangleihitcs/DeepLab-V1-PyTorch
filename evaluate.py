import os
import numpy as np
from PIL import Image

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

if __name__ == '__main__':
    mIOU = IOUMetric(num_classes=21)
    root_dir = '/home/ubuntu/workshops/datasets/voc12/VOCdevkit/VOC2012/'

    pred_dir = './exp/labels'
    gt_dir = root_dir + 'SegmentationClass/'
    ids = [i.strip() for i in open(root_dir + 'ImageSets/Segmentation/val.txt') if not i.strip() == '']

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
            (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
            (0.5, 0.75, 0), (0, 0.25, 0.5)]
    values = [i for i in range(21)]
    color2val = dict(zip(colormap, values))
    # print(color2val)

    import time
    st = time.time()
    for ind, img_id in enumerate(ids):
        img_path = os.path.join(gt_dir, img_id+'.png')
        pred_img_path = os.path.join(pred_dir, img_id+'.png')

        gt = Image.open(img_path)
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.int32)   # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        
        pred = Image.open(pred_img_path)
        pred = pred.crop((0, 0, w, h))
        pred = np.array(pred, dtype=np.int32)   # shape = [h, w]
        mIOU.add_batch(pred, gt)
        # print(img_id, ind)

    acc, acc_cls, iou, miou, fwavacc = mIOU.evaluate()
    print(acc, acc_cls, iou, miou, fwavacc)
    print('mIOU = %s, time = %s s' % (miou, str(time.time() - st)))
