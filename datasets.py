import numpy as np
import torch
from PIL import Image, ImageDraw
import os
import numpy as np
import cv2
import random

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, split='train_aug', crop_size=321):
        self.root = './VOCdevkit/VOC2012/'
        self.ann_dir_path = os.path.join(self.root, 'Annotations')
        self.image_dir_path = os.path.join(self.root, 'JPEGImages')
        self.label_dir_path = os.path.join(self.root, 'SegmentationClassAug')
        self.id_path = os.path.join('./list', split + '.txt')

        self.image_ids = [i.strip() for i in open(self.id_path) if not i.strip() == ' ']
        print('%s datasets num = %s' % (split, self.__len__()))

        self.mean_bgr = np.array((104.008, 116.669, 122.675))
        self.split = split
        self.crop_size = crop_size
        self.ignore_label = 255
        self.base_size = None
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        self.is_augment = True
        self.is_flip = True
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir_path, image_id + '.jpg')
        label_path = os.path.join(self.label_dir_path, image_id + '.png')
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        if self.is_augment:
            image, label = self._augmentation(image, label)
        
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)
    
    def _augmentation(self, image, label):
        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        # print(bbox)

        if self.split.__contains__('train') and self.is_flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

if __name__ == "__main__":
    dataset = VOCDataset()
    id, image, label = dataset.__getitem__(0)