import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from PIL import Image
import time
from time import strftime, localtime
import cv2
import argparse

from datasets import VOCDataset
from nets import vgg
from utils import crf, losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

batch_size = 30 # 30 for "step", 10 for 'poly'
lr = 1e-3
weight_decay = 5e-4
num_max_iters = 20000 # 6000 for "step", 20000 for 'poly'
num_update_iters = 10 # 4000 for "step", 10 for 'poly'
num_save_iters = 1000
num_print_iters = 100
init_model_path = './data/deeplab_largeFOV.pth'
log_path = './exp/log.txt'
model_path_save = './exp/model_last_'
root_dir_path = './VOCdevkit/VOC2012'
pred_dir_path = './exp/labels/'

def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] != 'features.38':
                    # print(m[0], m[1])
                    yield m[1].weight

    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] != 'features.38':
                    yield m[1].bias
    if key == '10x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    yield m[1].weight
    if key == '20x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0] == 'features.38':
                    yield m[1].bias

def train():
    model = vgg.VGG16_LargeFOV()
    model.load_state_dict(torch.load(init_model_path))
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        params = [
            {
                'params': get_params(model, '1x'),
                'lr': lr,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '2x'),
                'lr': lr * 2,
                'weight_decay': 0
            },
            {
                'params': get_params(model, '10x'),
                'lr': lr * 10,
                'weight_decay': weight_decay
            },
            {
                'params': get_params(model, '20x'),
                'lr': lr * 20,
                'weight_decay': 0
            },
        ],
        momentum = 0.9,
    )
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) # for val mIoU = 69.6

    print('Set data...')
    train_loader = torch.utils.data.DataLoader(
        VOCDataset(split='train_aug', crop_size=321, is_scale=False, is_flip=True),
        # VOCDataset(split='train_aug', crop_size=321, is_scale=True, is_flip=True), # for val mIoU = 69.6
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Learning rate policy
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])

    print('Start train...')
    iters = 0
    log_file = open(log_path, 'w')
    loss_iters, accuracy_iters = [], []
    for epoch in range(1, 100):
        for iter_id, batch in enumerate(train_loader):
            loss_seg, accuracy = losses.build_metrics(model, batch, device)
            optimizer.zero_grad()
            loss_seg.backward()
            optimizer.step()

            loss_iters.append(float(loss_seg.cpu()))
            accuracy_iters.append(float(accuracy))

            iters += 1
            if iters % num_print_iters == 0:
                cur_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
                log_str = 'iters:{:4}, loss:{:6,.4f}, accuracy:{:5,.4}'.format(iters, np.mean(loss_iters), np.mean(accuracy_iters))
                print(log_str)
                log_file.write(cur_time + ' ' + log_str + '\n')
                log_file.flush()
                loss_iters = []
                accuracy_iters = []
            
            if iters % num_save_iters == 0:
                torch.save(model.state_dict(), model_path_save + str(iters) + '.pth')
            
            # step
            # if iters == num_update_iters or iters == num_update_iters + 1000:
            #     for group in optimizer.param_groups:
            #         group["lr"] *= 0.1
            
            # poly
            for group in optimizer.param_groups:
                group["lr"] = group['initial_lr'] * (1 - float(iters) / num_max_iters) ** 0.9

            if iters == num_max_iters:
                exit()


def test(model_path_test, use_crf):
    batch_size = 2
    is_post_process = use_crf
    crop_size = 513
    model = vgg.VGG16_LargeFOV(input_size=crop_size, split='test')
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(model_path_test))
    model.eval()
    model = model.to(device)
    val_loader = torch.utils.data.DataLoader(
        VOCDataset(split='val', crop_size=crop_size, label_dir_path='SegmentationClassAug', is_scale=False, is_flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # DenseCRF
    post_processor = crf.DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=5,         # 4, 5
    )

    img_dir_path = root_dir_path + '/JPEGImages/'
    # class palette for test
    palette = []
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21] = np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128]], dtype='uint8').flatten()
    times = 0.0
    index = 0
    loss_iters, accuracy_iters = [], []
    CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)
    for iter_id, batch in enumerate(val_loader):
        image_ids, images, labels = batch
        images = images.to(device)
        labels = losses.resize_labels(labels, size=(crop_size, crop_size)).to(device)
        logits = model(images)
        probs = nn.functional.softmax(logits, dim=1) # shape = [batch_size, C, H, W]

        outputs = torch.argmax(probs, dim=1) # shape = [batch_size, H, W]
        
        loss_seg = CEL(logits, labels)
        accuracy = float(torch.eq(outputs, labels).sum().cpu()) / (len(image_ids) * logits.shape[2] * logits.shape[3])
        loss_iters.append(float(loss_seg.cpu()))
        accuracy_iters.append(float(accuracy))

        for i in range(len(image_ids)):
            if is_post_process:
                raw_image = cv2.imread(img_dir_path + image_ids[i] + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
                h, w = raw_image.shape[:2]
                pad_h = max(513 - h, 0)
                pad_w = max(513 - w, 0)
                pad_kwargs = {
                    "top": 0,
                    "bottom": pad_h,
                    "left": 0,
                    "right": pad_w,
                    "borderType": cv2.BORDER_CONSTANT,
                }
                raw_image = cv2.copyMakeBorder(raw_image, value=[0, 0, 0], **pad_kwargs)
                raw_image = raw_image.astype(np.uint8)
                start_time = time.time()
                prob = post_processor(raw_image, probs[i].detach().cpu().numpy())
                times += time.time() - start_time
                output = np.argmax(prob, axis=0).astype(np.uint8)
                img_label = Image.fromarray(output)
            else:
                output = np.array(outputs[i].cpu(), dtype=np.uint8)
                img_label = Image.fromarray(output)
            img_label.putpalette(palette)
            img_label.save(pred_dir_path + image_ids[i] + '.png')

            accuracy = float(torch.eq(outputs[i], labels[i]).sum().cpu()) / (logits.shape[2] * logits.shape[3])
            index += 1
            if index % 200 == 0:
                print(image_ids[i], float('%.4f' % accuracy), index)
    if is_post_process:
        print('dense crf time = %s' % (times / index))
    print('val loss = %s, acc = %s' % (np.mean(loss_iters), np.mean(accuracy_iters)))
    print(model_path_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='test', help='train or test model')
    parser.add_argument('--model_path_test', default='./exp/model_last_20000.pth', help='test model path')
    parser.add_argument('--use_crf', default=False, action='store_true', help='use crf or not')
    args = parser.parse_args()

    if args.type == 'train':
        train()
    else:
        test(args.model_path_test, args.use_crf)

