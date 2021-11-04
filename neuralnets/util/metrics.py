import torchmetrics as tm
import torch
from torch import nn
import torch
import torchmetrics as tm
from torch import nn
import torch.nn.functional as f
from neuralnets.util.torch.metrics import iou

########################################
# ---------- Wrapped Metrics --------- #
########################################


class Accuracy(object):
    """
    Wrapper around accuracy function

    """

    def __init__(self):
        self.softmax = nn.Softmax(dim=-1)
        self.metric_func = tm.Accuracy().to('cuda:4')

    def __call__(self, preds, labels):
        # pr = list(torch.argmax(self.softmax(preds), dim=-1).detach().cpu().numpy())
        # la = list(labels[:, 0, ...].detach().cpu().numpy())
        # a = [(p, l) for p, l in zip(pr, la)]
        # print(a)
        # print('preds ', torch.argmax(self.softmax(preds), dim=-1).shape)
        # print('targets ', labels[:, 0, ...].shape)
        return self.metric_func(torch.argmax(self.softmax(preds), dim=-1), labels[:, 0, ...])


class MSE(object):
    """
    Wrapper around skimage's mean_squared_error function
    """

    def __init__(self):
        self.metric_func = tm.MeanSquaredError().cuda()

    def __call__(self, *args):
        return self.metric_func(*args)


class mIoU(object):
    """
    Wrapper around mIoU metric function
    """

    def __init__(self, num_classes):
        self.metric_func = tm.IoU(num_classes, ignore_index=255)
        # self.num_classes = num_classes

    def __call__(self, preds, labels):
        # print(preds.shape)
        # print(labels.shape)
        # print(torch.unique(labels))
        # exit()
        # mask = labels != 255
        # ious = torch.stack([iou(labels[:, 0] == c, preds[:, c]) for c in range(self.num_classes)])

        return self.metric_func(preds, labels)
