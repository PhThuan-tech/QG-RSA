import os
import numpy as np
import torch
import torch.nn as nn
import logging

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def log_count_parameter(param_groups):
    all_params = []
    for g in param_groups:
        all_params += list(g['params'])

    all_count = sum(p.numel() for p in all_params)
    trainable_count = sum(p.numel() for p in all_params if p.requires_grad)

    logging.info('All params: {}'.format(all_count))
    logging.info('Trainable params: {}'.format(trainable_count))

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10, task_sizes=None):
    assert len(y_pred) == len(y_true), "Data length error."
    if len(y_true) == 0:
        raise ValueError("Cannot compute accuracy for an empty target array.")
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy.  ``task_sizes`` is authoritative because the final task
    # need not equal the nominal increment (and fixtures may use uneven tasks).
    if task_sizes is None:
        max_class = int(np.max(y_true))
        boundaries = [
            (class_id, min(class_id + int(increment), max_class + 1))
            for class_id in range(0, max_class + 1, int(increment))
        ]
    else:
        boundaries = []
        lower = 0
        for size in task_sizes:
            size = int(size)
            if size <= 0:
                raise ValueError("task_sizes must contain positive integers.")
            boundaries.append((lower, lower + size))
            lower += size

    for class_id, upper in boundaries:
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < upper)
        )[0]
        if len(idxes) == 0:
            continue
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(upper - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

class BaseAttention(nn.Module):

    def __init__(self):
        super(BaseAttention, self).__init__()

    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x + x

class AutoencoderSigmoid(BaseAttention):
    def __init__(self, input_dims=768, code_dims=384):
        super(AutoencoderSigmoid, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.GELU(),
            nn.Linear(64, code_dims),
            nn.GELU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, 64),
            nn.GELU(),
            nn.Linear(64, input_dims),
            nn.Sigmoid())
