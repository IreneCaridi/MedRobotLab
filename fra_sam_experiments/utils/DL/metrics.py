
from .callbacks import BaseCallback

import torch
import torchmetrics
import numpy as np


class BaseMetric(BaseCallback):
    def __init__(self, num_classes=2, device="gpu"):
        """
        :param
            --num_classes: number of classes
            --device: device "cpu" or "gpu"
        """
        super().__init__()
        if device == "gpu" and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = "cpu"

        self.num_classes = num_classes

        self.t_value = 0.0
        self.v_value = 0.0
        self.t_value_mean = 0.0
        self.v_value_mean = 0.0

    def on_train_batch_end(self, output=None, target=None, batch=None):
        if self.device == "cpu":
            output = output.to("cpu")
            target = target.to("cpu")
        batch_value = self.metric(output, target)
        self.t_value = self.metric.compute()  # value computed along every batch
        self.t_value_mean = self.t_value.mean()

    def on_val_start(self):
        self.metric.reset()

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if self.device == "cpu":
            output = output.to("cpu")
            target = target.to("cpu")
        batch_value = self.metric(output, target)
        self.v_value = self.metric.compute()  # value computed along every batch
        self.v_value_mean = self.v_value.mean()

    def on_epoch_end(self, epoch=None):
        self.t_value = 0.0
        self.v_value = 0.0
        self.t_value_mean = 0.0
        self.v_value_mean = 0.0
        self.metric.reset()


class Accuracy(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes,
                                                           top_k=top_k, average=None).to(self.device)


class Precision(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Precision(task="multiclass", num_classes=num_classes,
                                                            top_k=top_k, average=None).to(self.device)


class Recall(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --top_k: Number of highest probability or logit score predictions considered to find the correct label.
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", top_k=1, thresh=0.5):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.Recall(task="multiclass", num_classes=num_classes,
                                                             top_k=top_k, average=None).to(self.device)


class AUC(BaseMetric):
    """
    :param
        --num_classes: number of classes
        --device: device "cpu" or "gpu"
        --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
    """
    def __init__(self, num_classes=2, device="gpu", thresh=None):
        super().__init__(num_classes, device)

        self.metric = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_classes,
                                                        average=None, thresholds=thresh).to(self.device)


class Metrics(BaseCallback):
    def __init__(self, loss_fn, num_classes=2, device="cpu", top_k=1, thresh=0.5):
        """

            wrapper for metrics to be used in the model

        """

        # self.A = Accuracy(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        # self.P = Precision(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        # self.R = Recall(num_classes=num_classes, device=device, top_k=top_k, thresh=thresh)
        # self.AuC = AUC(num_classes=num_classes, device=device, thresh=None)
        # self.metrics = [self.A, self.P, self.R, self.AuC]
        self.loss_fn = loss_fn
        self.metrics = []
        self.dict = self.build_metrics_dict()
        self.num_classes = num_classes

    def on_train_batch_end(self, output=None, target=None, batch=None):
        for obj in self.metrics:
            obj.on_train_batch_end(output, target, batch)

    def on_train_end(self, num_batches=None):
        for k in self.loss_fn.running_dict.keys():
            if k == 'dice':
                self.dict["train_" + k] = [self.loss_fn.running_dict[k][i] / num_batches
                                           for i in self.loss_fn.running_dict[k].shape[0]]
            else:
                self.dict["train_" + k] = [self.loss_fn.running_dict[k] / num_batches]  # all scalars

        for obj in self.metrics:
            name = "train_" + obj.__class__.__name__
            self.dict[name] = [x for x in obj.t_value.to("cpu").numpy().astype(np.float16)]

    def on_val_start(self):
        for obj in self.metrics:
            obj.on_val_start()

    def on_val_end(self, num_batches=None, epoch=None):
        for k in self.loss_fn.running_dict.keys():
            if k == 'dice':
                self.dict["val_" + k] = [self.loss_fn.running_dict[k][i] / num_batches
                                         for i in self.loss_fn.running_dict[k].shape[0]]
            else:
                self.dict["val_" + k] = [self.loss_fn.running_dict[k] / num_batches]  # all scalars

        for obj in self.metrics:
            name = "val_" + obj.__class__.__name__
            self.dict[name] = [x for x in obj.v_value.to("cpu").numpy().astype(np.float16)]

    def on_val_batch_end(self, output=None, target=None, batch=None):
        for obj in self.metrics:
            obj.on_val_batch_end(output, target, batch)

    def on_epoch_end(self, epoch=None):
        for obj in self.metrics:
            obj.on_epoch_end(epoch)

    def build_metrics_dict(self):

        names = [obj.__class__.__name__ for obj in self.metrics]
        names += [k for k in self.loss_fn.running_dict.keys()]
        keys = ["train_"+name for name in names] + ["val_"+name for name in names]

        return {key: None for key in keys}





#
# def plot_roc_curve(fpr, tpr, thresholds):
#     plt.figure(figsize=(8, 8))
#     plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
#
#     # Annotate points for the thresholds used in the ROC calculation
#     for threshold in thresholds:
#         idx = (thresholds >= threshold).nonzero()[0].max()
#         plt.annotate(f'{threshold:.2f}', (fpr[idx], tpr[idx]), textcoords="offset points", xytext=(0, 10),
#                      ha='center', fontsize=8, color='red')
#
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def plot_pr_curve(precision, recall, thresholds):
#     plt.figure(figsize=(8, 8))
#     plt.plot(recall, precision, color='green', lw=2, label='PR curve')
#
#     # Annotate points for every threshold value
#     for threshold in thresholds:
#         idx = (thresholds >= threshold).nonzero()[0].max()
#         plt.annotate(f'Threshold = {threshold:.2f}', (recall[idx], precision[idx]),
#                      textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')
#
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
