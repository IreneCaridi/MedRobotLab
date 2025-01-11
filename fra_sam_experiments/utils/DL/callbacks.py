import torch
# import torchvision
# import numpy as np
# from matplotlib import pyplot as plt
# from pathlib import Path
import os
# import pandas
#
# from utils import increment_path

from .. import my_logger

# ----------------------------------------------------------------------------------------------------------------------
# BASE CALLBACK CLASS
# ----------------------------------------------------------------------------------------------------------------------


class BaseCallback:

    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self, metrics = None):
        pass

    def on_val_start(self):
        pass

    def on_val_end(self, metrics=None, epoch=None):
        pass

    def on_train_batch_start(self, batch=None):
        pass

    def on_train_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_val_batch_start(self, batch=None):
        pass

    def on_val_batch_end(self, output=None, target=None, batch=None):
        pass

    def on_epoch_start(self, epoch=None, max_epoch=None):
        pass

    def on_epoch_end(self, epoch=None):
        pass

# ----------------------------------------------------------------------------------------------------------------------
# CALLBACK SUBCLASSES
# ----------------------------------------------------------------------------------------------------------------------


class EarlyStopping(BaseCallback):

    def __init__(self, patience=30, monitor="val_loss", mode='min'):
        super().__init__()
        self.mode = mode
        self.monitor = monitor
        if mode == 'max':
            self.best_fitness = 0.0
        if mode == 'min':
            self.best_fitness = 1000.0
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
        self.stop = False

    def on_val_end(self, metrics=None, epoch=None):
        fitness = metrics[self.monitor][0]
        if self.mode == "min":
            if fitness <= self.best_fitness:
                self.best_epoch = epoch
                self.best_fitness = fitness
        elif self.mode == "max":
            if fitness >= self.best_fitness:
                self.best_epoch = epoch
                self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        self.stop = delta >= self.patience  # stop training if patience exceeded

    def on_epoch_end(self, epoch=None):
        if self.stop:
            raise StopIteration


class Saver(BaseCallback):
    def __init__(self, model, save_path, save_best=True,  monitor="val_loss", mode='min'):

        self.model = model
        self.monitor = monitor
        self.mode = mode
        if mode == "min":
            self.best_fitness = 5.  # validation
        elif mode == "max":
            self.best_fitness = 0.0  # validation
        else:
            raise TypeError("mode not recognized, use ['min', 'max']")
        self.best_epoch = 0
        self.save_path = save_path
        self.save_best = save_best
        if not os.path.isdir(self.save_path / "weights"):
            os.mkdir(self.save_path / "weights")

    def on_val_end(self, metrics=None, epoch=None):
        if self.save_best:
            fitness = metrics[self.monitor][0]
            if self.mode == "min":
                if fitness <= self.best_fitness:
                    self.save(fitness, epoch)
            elif self.mode == "max":
                if fitness >= self.best_fitness:
                    self.save(fitness, epoch)
        else:
            pass

    def on_end(self):
        torch.save(self.model, self.save_path / f"weights/last.pt")
        my_logger.info(f"model saved to {self.save_path}")

    def save(self, fitness, epoch, name="best"):
        self.best_fitness = fitness
        if os.path.isfile(self.save_path / "weights" / f"{name}_{self.best_epoch}.pt"):
            os.remove(self.save_path / "weights" / f"{name}_{self.best_epoch}.pt")
        torch.save(self.model, self.save_path / "weights" / f"{name}_{epoch}.pt")
        my_logger.info("saved best")
        self.best_epoch = epoch


class Callbacks(BaseCallback):
    def __init__(self, callbacks_list: list):
        """
            wrapper for all callbacks

        args:
            - callbacks_list = list containing all the callbacks objects to be used
        """

        self.list = callbacks_list

    def on_train_batch_end(self, output=None, target=None, batch=None):
        for obj in self.list:
            obj.on_train_batch_end(output, target, batch)

    def on_val_start(self):
        for obj in self.list:
            obj.on_val_start()

    def on_val_end(self, metrics=None, epoch=None):
        for obj in self.list:
            obj.on_val_end(metrics, epoch)

    def on_val_batch_end(self, output=None, target=None, batch=None):
        for obj in self.list:
            obj.on_val_batch_end(output, target, batch)

    def on_epoch_end(self, epoch=None):
        for obj in self.list:
            obj.on_epoch_end(epoch)

    def on_end(self):
        for obj in self.list:
            obj.on_end()





