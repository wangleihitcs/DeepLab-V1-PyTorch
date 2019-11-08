import torch
import torch.nn as nn
from PIL import Image
import numpy as np


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def build_metrics(model, batch, device):
    CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)

    image_ids, images, labels = batch
    labels = resize_labels(labels, size=(41, 41)).to(device)
    logits = model(images.to(device))

    loss_seg = CEL(logits, labels)

    preds = torch.argmax(logits, dim=1)
    accuracy = float(torch.eq(preds, labels).sum().cpu()) / (len(image_ids) * logits.shape[2] * logits.shape[3])

    return loss_seg, accuracy
                
