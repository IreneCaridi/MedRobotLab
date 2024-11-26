import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
import torchmetrics
import wandb

from .callbacks import BaseCallback


class LogsHolder(BaseCallback):
    """
        :param
            --metrics = metrics object
    """
    def __init__(self, metrics, test=False, wandb=False):
        super().__init__()
        self.metrics = metrics
        self.dict = self.build_metrics_dict()
        self.test = test
        self.wandb = wandb

    def build_metrics_dict(self):
        d = {key: [] for key in self.metrics.dict}
        dummy = d.copy()
        for key in dummy:
            if "loss" not in key:
                for i in range(self.metrics.num_classes):
                    d[key+f"_{i}"] = []
        return d

    def on_epoch_end(self, epoch=None):
        if not self.test:
            for key in self.metrics.dict:
                if "loss" in key:
                    self.dict[key].append(self.metrics.dict[key][0])
                else:
                    for i in range(len(self.metrics.dict[key])):
                        self.dict[key+f"_{i}"].append(self.metrics.dict[key][i])
                    flat = [item for item in self.metrics.dict[key]]
                    self.dict[key].append(np.float16(sum(flat)/len(flat)))  # mean value
        else:
            for key in self.metrics.dict:
                if 'train' not in key:
                    if "loss" in key:
                        pass
                    else:
                        for i in range(len(self.metrics.dict[key])):
                            self.dict[key+f"_{i}"].append(self.metrics.dict[key][i])
                        flat = [item for item in self.metrics.dict[key]]
                        self.dict[key].append(np.float16(sum(flat)/len(flat)))  # mean value
        if self.wandb:
            dd = {k: self.dict[k][-1] for k in self.dict.keys()}
            dd['epoch'] = epoch
            wandb.log(dd)


class SaveCSV(BaseCallback):
    """
        :param
            --logs = LogsHolder object
    """
    def __init__(self, logs, save_path, name="results.csv", test=False):
        super().__init__()
        self.save_path = save_path
        self.logs = logs
        self.file_name = name
        self.build_dict()
        self.test = test

    def on_epoch_start(self, epoch=None, max_epoch=None):
        for key in self.dict:
            self.dict[key] = []

    def on_epoch_end(self, epoch=None):
        self.update_dict()
        self.write_csv()

    def build_dict(self):
        # dictionary with only last metrics
        setattr(self, "dict", {key: [] for key in self.logs.dict})

    def update_dict(self):
        if not self.test:
            for key in self.dict:
                self.dict[key].append(self.logs.dict[key][-1])
        else:
            for key in self.dict:
                if 'train' not in key and 'loss' not in key:
                    self.dict[key].append(self.logs.dict[key][-1])

    def write_csv(self):
        if self.test:
            df = pd.DataFrame({key: self.dict[key] for key in self.dict if 'train' not in key and 'loss' not in key})
        else:
            df = pd.DataFrame(self.dict)
        csv_path = self.save_path / self.file_name
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)


class SaveFigures(BaseCallback):
    def __init__(self, logs, save_path, name="metrics.png"):
        self.save_path = save_path
        self.logs = logs
        self.name = name

    def on_end(self):
        self.save_metrics()

    def save_metrics(self):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 11))

        self.plot_metric("val_loss", axes[0])
        # self.plot_metric("Accuracy", axes[0, 1])
        # self.plot_metric("Precision", axes[0, 2])
        self.plot_metric("train_loss", axes[1])
        # self.plot_metric("AUC", axes[1, 1])
        # self.plot_metric("Recall", axes[1, 2])

        fig.tight_layout()
        plt.savefig(self.save_path / self.name, dpi=96)
        plt.close()

    def plot_metric(self, metric, ax):
        if "loss" not in metric:
            for key in self.logs.dict:
                if metric in key and "train_" not in key:
                    if key[-1].isdigit():
                        lab = key[-1]
                    else:
                        lab = "mean"
                    ax.plot(range(len(self.logs.dict[key])), self.logs.dict[key], label=lab)
            ax.legend()
        else:
            ax.plot(range(len(self.logs.dict[metric])), self.logs.dict[metric])
        ax.set_title(metric)


class LrLogger(BaseCallback):
    def __init__(self, opt, save_path, wandb=False):
        super().__init__()
        self.opt = opt
        self.save_path = save_path
        self.log = []
        self.last_epoch = 0
        self.wandb = wandb

    def on_epoch_end(self, epoch=None):
        self.log.append(self.opt.param_groups[0]["lr"])
        self.last_epoch = len(self.log)
        if self.wandb:
            wandb.log({'epoch': epoch, 'lr': self.log[-1]})

    def on_end(self):
        self.save()

    def save(self, name="lr.png"):
        plt.figure(figsize=(20, 11))
        plt.plot(range(self.last_epoch), self.log, marker='*')
        plt.xlabel('Epoch')
        plt.title('lr Scheduler')
        plt.savefig(self.save_path / name, dpi=96)
        plt.close()


class ROClogger(BaseCallback):
    """
       :param
           --num_classes: number of classes
           --device: device "cpu" or "gpu"
           --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
       """

    def __init__(self, save_path, num_classes=2, device='gpu', thresh=None, name='ROC.png'):
        super().__init__()

        if device == "gpu" and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = "cpu"

        self.save_path = save_path

        self.name = name

        self.run = False
        self.num_classes = num_classes

        self.metric = torchmetrics.classification.ROC(task="multiclass", num_classes=num_classes,
                                                      thresholds=thresh).to(self.device)

        self.fpr = None
        self.tpr = None
        self.thresh = None

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if batch == 0:
            setattr(self, "preds", output)
            setattr(self, "labs", target)
        else:
            self.preds = torch.cat((self.preds, output), dim=0)
            self.labs = torch.cat((self.labs, target), dim=0)

    def on_val_end(self, metrics=None, epoch=None):
        self.fpr, self.tpr, self.thresh = self.metric(self.preds, self.labs)
        self.metric.reset()
        self.preds = 0
        self.labs = 0

    def on_end(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 11))
        if self.num_classes == 2:
            self.metric.plot(curve=(self.fpr[1], self.tpr[1], self.thresh[1]), score=True, ax=ax)
        else:
            self.metric.plot(curve=(self.fpr, self.tpr, self.thresh), score=True, ax=ax)
        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

        plt.tight_layout()
        plt.savefig(self.save_path / self.name, dpi=96)
        plt.close()

    def on_epoch_start(self, epoch=None, max_epoch=None):
        pass


class PRClogger(BaseCallback):
    """
       :param
           --num_classes: number of classes
           --device: device "cpu" or "gpu"
           --thresh: Threshold for transforming probability to binary {0,1} predictions (ONLY if binary)
       """

    def __init__(self, save_path, num_classes=2, device='gpu', thresh=None, name='PRC.png'):
        super().__init__()

        if device == "gpu" and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = "cpu"

        self.name = name

        self.save_path = save_path
        self.run = False
        self.num_classes = num_classes

        self.metric = torchmetrics.classification.PrecisionRecallCurve(task="multiclass", num_classes=num_classes,
                                                                       thresholds=thresh).to(self.device)

        self.metric_best_P = torchmetrics.classification.PrecisionAtFixedRecall(task='multiclass', min_recall=0.5,
                                                                                num_classes=num_classes).to(self.device)
        self.metric_best_R = torchmetrics.classification.RecallAtFixedPrecision(task='multiclass', min_precision=0.5,
                                                                                num_classes=num_classes).to(self.device)

        self.P = None
        self.R = None
        self.thresh = None  # REMEMBER TO SWAP ORDER [::-1]
        self.best_P = None
        self.best_R = None
        self.best_P_th = None
        self.best_R_th = None

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if batch == 0:
            setattr(self, "preds", output)
            setattr(self, "labs", target)
        else:
            self.preds = torch.cat((self.preds, output), dim=0)
            self.labs = torch.cat((self.labs, target), dim=0)

    def on_val_end(self, metrics=None, epoch=None):
        self.P, self.R, self.thresh = self.metric(self.preds, self.labs)
        self.metric.reset()
        self.best_P, self.best_P_th = self.metric_best_P(self.preds, self.labs)
        self.best_R, self.best_R_th = self.metric_best_R(self.preds, self.labs)
        self.metric_best_P.reset()
        self.metric_best_R.reset()
        self.preds = 0
        self.labs = 0

    def on_end(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 11))
        if self.num_classes == 2:
            self.metric.plot(curve=(self.R[1], self.P[1], self.thresh[::-1][1]), score=True, ax=ax)
        else:
            self.metric.plot(curve=(self.R, self.P, self.thresh[::-1]), score=True, ax=ax)
        text_content = self.get_text()
        ax.text(1.01, 0.97, text_content, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        plt.tight_layout()
        plt.savefig(self.save_path / self.name, dpi=96)
        plt.close()

    def on_epoch_start(self, epoch=None, max_epoch=None):
        pass

    def get_text(self):
        sep = '\n'
        rows = ['BEST METRICS']
        if self.num_classes != 2:
            for i, ((P, th_P), (R, th_R)) in enumerate(zip(zip(self.best_P, self.best_P_th), zip(self.best_R, self.best_R_th))):
                rows.append(f'class {i}: P = {P :.2f} at {th_P :.2f}; R = {R :.2f} at {th_R :.2f}')
        else:
            rows.append(f'class {1}: P = {self.best_P[1] :.2f} at {self.best_P_th[1] :.2f}; R = {self.best_R[1] :.2f} at {self.best_R_th[1] :.2f}')

        return sep.join(rows)


class ConfusionMatrixLogger(BaseCallback):
    """
       :param
           --num_classes: number of classes
           --device: device "cpu" or "gpu"
       """

    def __init__(self, save_path, num_classes=2, device='gpu', name='ConfusionMatrix.png'):
        super().__init__()

        if device == "gpu" and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = "cpu"

        self.name = name
        self.save_path = save_path

        self.num_classes = num_classes

        self.metric = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)

        self.cm = None

    def on_val_batch_end(self, output=None, target=None, batch=None):
        if batch == 0:
            setattr(self, "preds", output)
            setattr(self, "labs", target)
        else:
            self.preds = torch.cat((self.preds, output), dim=0)
            self.labs = torch.cat((self.labs, target), dim=0)

    def on_val_end(self, metrics=None, epoch=None):
        self.cm = self.metric(self.preds, self.labs)
        self.metric.reset()
        self.preds = 0
        self.labs = 0

    def on_end(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 11))
        self.metric.plot(val=self.cm, ax=ax)
        plt.savefig(self.save_path / self.name, dpi=96)
        plt.close()

    def on_epoch_start(self, epoch=None, max_epoch=None):
        pass


class Loggers(BaseCallback):
    def __init__(self, metrics, opt, save_path, test, wandb):
        super().__init__()
        self.logs = LogsHolder(metrics, test=test, wandb=wandb)
        self.csv = SaveCSV(self.logs, save_path, test=test)
        self.lr = LrLogger(opt, save_path, wandb=wandb)
        self.figure_saver = SaveFigures(self.logs, save_path)
        # self.ROC = ROClogger(save_path, num_classes=metrics.num_classes, device=device)
        # self.PRC = PRClogger(save_path, num_classes=metrics.num_classes, device=device)
        # self.cm = ConfusionMatrixLogger(save_path, metrics.num_classes, device)
        self.build_list()

    def on_epoch_end(self, epoch=None):
        for obj in self.list:
            obj.on_epoch_end(epoch)

    def on_val_batch_end(self, output=None, target=None, batch=None):
        for obj in self.list:
            obj.on_val_batch_end(output, target, batch)

    def on_val_end(self, metrics=None, epoch=None):
        for obj in self.list:
            obj.on_val_end(metrics, epoch)

    def on_epoch_start(self, epoch=None, max_epoch=None):
        for obj in self.list:
            obj.on_epoch_start(epoch, max_epoch)

    def on_end(self):
        for obj in self.list:
            obj.on_end()

    def build_list(self):
        setattr(self, "list", [value for _, value in vars(self).items()])


def log_confidence_score(indexes, scores, save_path):

    scores_np = scores.numpy()

    # all together
    save_boxplot([scores_np], save_path / 'confidence.png')

    N = scores_np[np.where(indexes == 0)]
    V = scores_np[np.where(indexes == 1)]
    S = scores_np[np.where(indexes == 2)]

    save_scores(scores_np, N, V, S, save_path)

    div_data = [d for d in [N, V, S] if d.size != 0]
    # divided
    save_boxplot(div_data, save_path / 'divided_confidence.png')


def save_boxplot(values: list, save_path):

    names = [apply_map(l) for l in range(len(values))]
    data = [d for d in values]

    plt.figure(figsize=(20, 11))
    plt.boxplot(data, vert=True, patch_artist=True, labels=names)
    plt.title('Boxplot')
    plt.savefig(save_path, dpi=96)


def apply_map(x):
    if x == 0:
        return 'N'
    if x == 1:
        return 'V'
    if x == 2:
        return 'S'


def save_scores(scores_np, N, V, S, save_path):
    np.save(save_path / 'scores.npy', scores_np)
    df = pd.DataFrame({})
    df['mean_conf'] = np.mean(scores_np)
    df['mean_0_conf'] = np.mean(N)
    df['mean_1_conf'] = np.mean(V)
    np.save(save_path / 'scores_0.npy', N)
    np.save(save_path / 'scores_1.npy', V)
    if S.size != 0:
        np.save(save_path / 'scores_2.npy', S)
        df['mean_2_conf'] = np.mean(S)

    df.to_csv(save_path / 'results.csv', index=False)


def save_predictions(pred, save_path):
    pred = pred.numpy()
    np.save(save_path / 'predictions.npy', pred)

