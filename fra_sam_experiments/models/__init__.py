import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys
import logging

# SETTING GLOBAL VARIABLES
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def check_load_model(model, backbone_weights, info_logger):
    if not backbone_weights:
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, str) and Path(model).suffix == ".pt" or ".pth":
            info_logger.info(f"Loading weights from {model}")
            return torch.load(model, map_location=torch.device('cpu'))
        else:
            raise TypeError("model not recognised")
    else:
        # I'm loading only the weights from the backbone

        assert isinstance(model, nn.Module)  # check that the model is something to load weights to

        info_logger.info(f'loading weights from {backbone_weights}')
        old = torch.load(backbone_weights, map_location=torch.device('cpu'))

        new_state_dict = {k: model.state_dict()[k] for k in model.state_dict()}
        for k in old.state_dict():
            if k in new_state_dict:  # to load normally I'd say...
                new_state_dict[k] = old.state_dict()[k]
            elif'encoder.'+k in new_state_dict:  # to load after pre distillation from sam
                new_state_dict['encoder.'+k] = old.state_dict()[k]
            else:
                continue

        info_logger.info(model.load_state_dict(new_state_dict, strict=False))

        return model


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='cpu', callbacks=None, loss_fn=None, optimizer=None, sched=None,
                 metrics=None, loggers=None, AMP=True, freeze=None, info_log=None, is_det=False):
        super().__init__()
        """
        :param
            --model: complete Torch model to train/test
            --loaders: tuple with the Torch data_loaders like (train,val,test)
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
            --loggers: loggers instance
            --AMP: Automatic Mixed Precision 
            --freeze: list containing names of layers to freeze
            --sequences: to handle windowed input sequences
            --is_det: flag whether you are using a detection head
        """

        assert isinstance(info_log, logging.Logger), 'provided info_log is not a logger'
        self.my_logger = info_log

        self.freeze = freeze

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError("model not recognised")

        self.train_loader, self.val_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                self.my_logger.info('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        self.my_logger.info(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.callbacks = callbacks

        self.metrics = metrics
        self.metrics1 = None

        self.loggers = loggers

        if loss_fn:
            self.loss_fun = loss_fn.to(self.device)
            self.opt = optimizer
            self.sched = sched

        if AMP and "cuda" in self.device:
            self.my_logger.info("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False

        if is_det:
            self.task = 'bbox'
        else:
            self.task = 'seg'

        total_params = sum(p.numel() for p in self.model.parameters())
        self.my_logger.info(f"'{self.model.name}' - Total parameters: {total_params / 1e6:.2f}M")
        if self.freeze:
            self.my_logger.info('freezing backbone...')

    def train_one_epoch(self, epoch_index, tot_epochs):
        self.loss_fun.reset()

        # initializing progress bar
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)

            inputs, labs, poly_box = self.unpack_data(data)

            self.opt.zero_grad()

            if self.AMP:
                with autocast():

                    outputs, loss = self.compute_loss(inputs, labs, poly_box)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                outputs, loss = self.compute_loss(inputs, labs, poly_box)

                loss.backward()
                self.opt.step()

            del inputs

            current_loss = self.loss_fun.get_current_value(batch)

            with torch.no_grad():
                if self.task == 'bbox':
                    labs = poly_box

                # computing training metrics
                self.metrics.on_train_batch_end(outputs, labs, batch)
                # # calling callbacks
                # self.callbacks.on_train_batch_end(outputs, labs, batch)

                self.update_pbar(pbar_loader, current_loss, epoch_index, tot_epochs, gpu_used, mode='train')

            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(batch + 1)

    def val_loop(self, epoch):
        self.loss_fun.reset()

        # resetting metrics for validation
        self.metrics.on_val_start()

        # calling callbacks
        self.callbacks.on_val_start()

        # initializing progress bar
        description = f'Validation'
        pbar_loader = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # Disable gradient computation and reduce memory consumption, and model to evaluation mode.
        self.model.eval()
        with torch.no_grad():
            for batch, data in pbar_loader:
                torch.cuda.empty_cache()  # Clear GPU memory

                inputs, labels, poly_box = self.unpack_data(data)

                outputs, _ = self.compute_loss(inputs, labels, poly_box)

                current_loss = self.loss_fun.get_current_value(batch)

                if self.device != "cpu":
                    torch.cuda.synchronize()

                if self.task == 'bbox':
                    labels = poly_box

                # computing metrics on batch
                self.metrics.on_val_batch_end(outputs, labels, batch)
                # # calling callbacks
                # self.callbacks.on_val_batch_end(outputs, labels, batch)
                # # updating roc and prc
                # self.loggers.on_val_batch_end(outputs, labels, batch)

                self.update_pbar(pbar_loader, current_loss, mode='val')

        if outputs is not None:
            # updating metrics dict
            self.metrics.on_val_end(batch + 1)

            # updating loggers (roc, prc)
            self.loggers.on_val_end()
            # calling callbacks
            self.callbacks.on_val_end(self.metrics.dict, epoch)

    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            self.metrics.on_epoch_start()

            self.loggers.on_epoch_start(epoch=epoch, max_epoch=num_epochs)

            # self.model.train(True)
            self.check_freeze()  # freezing specific layers (if needed)

            # for name, param in self.model.named_parameters():
            #     self.my_logger.info(name, param.requires_grad)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)

            # validation
            self.val_loop(epoch)

            # logging results
            self.loggers.on_epoch_end(epoch)
            # updating lr scheduler
            if self.sched:
                self.sched.step()
            #resetting metrics
            self.metrics.on_epoch_end()

            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                self.my_logger.info(f"early stopping at epoch {epoch}")
                break

        # self.loggers.on_epoch_end(0)

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model)
        self.callbacks.on_end()

    def check_freeze(self):
        if self.freeze:
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            self.model.train(True)

    def update_pbar(self, pbar, current_loss, epoch_index=None, tot_epochs=None, gpu_used=None, mode='train' ):
        if mode == 'train':
            if self.task == 'seg':
                # updating pbar
                A = self.metrics.A.t_value_mean
                P = self.metrics.P.t_value_mean
                R = self.metrics.R.t_value_mean
                # AUC = self.metrics.AuC.t_value[1]
                dice = self.metrics.Dice.t_value

                pbar.set_description(
                    f'Epoch {epoch_index}/{tot_epochs - 1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                    f'train_loss: {current_loss:.4f}, A: {A :.2f}, P: {P :.2f}, R: {R :.2f}, Dice: {dice: .2f}')
            elif self.task == 'bbox':
                mAP50 = self.metrics.mAP50.t_value_mean
                mAP50_95 = self.metrics.mAP50_95.t_value_mean
                IoU = self.metrics.IoU.t_value_mean
                pbar.set_description(
                    f'Epoch {epoch_index}/{tot_epochs - 1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                    f'train_loss: {current_loss:.4f}, IoU: {IoU: .2f}, mAP50: {mAP50}, mAP50_95: {mAP50_95:.2f}')
            else:
                raise NotImplementedError
        elif mode == 'val':
            if self.task == 'seg':
                # # updating pbar
                A = self.metrics.A.v_value_mean
                P = self.metrics.P.v_value_mean
                R = self.metrics.R.v_value_mean
                # AUC = self.metrics.AuC.v_value[1]
                dice = self.metrics.Dice.v_value
                description = f'Validation: val_loss: {current_loss:.4f}, A: {A :.2f}, ' \
                              f'P: {P :.2f}, R: {R :.2f}, Dice: {dice :.2f}'

                # description = f'Validation: val_loss: {current_loss:.4f}'
                pbar.set_description(description)
            elif self.task == 'bbox':
                mAP50 = self.metrics.mAP50.v_value_mean
                mAP50_95 = self.metrics.mAP50_95.v_value_mean
                IoU = self.metrics.IoU.v_value_mean
                pbar.set_description(f'Validation: val_loss: {current_loss:.4f},  IoU: {IoU: .2f}, mAP50: {mAP50:.2f}, '
                                     f'mAP50_95: {mAP50_95:.2f}')
            else:
                raise NotImplementedError

    def unpack_data(self, data):
        inputs, labels, poly_box = data
        inputs = inputs.to(self.device)

        if self.task == 'seg':
            labels = labels.to(self.device)
        elif self.task == 'bbox':
            for ann in poly_box:
                ann['bboxes'] = ann['bboxes'].to(self.device)
                ann['labels'] = ann['labels'].to(self.device)
        else:
            raise NotImplementedError

        return inputs, labels, poly_box

    def compute_loss(self, inputs, labs, poly_box):
        if self.task == 'seg':
            outputs = self.model(inputs)
            loss = self.loss_fun(outputs, labs)
            return outputs, loss
        elif self.task == 'bbox':
            x_enc, x_raw, x_pred = self.model(inputs)
            loss_dict = self.model.get_loss(x_raw, poly_box)
            loss = self.loss_fun(loss_dict)
            # x_pred is [Instancedata{scores:[...], labels[...], bboxes:[...]}...]
            return  x_pred, loss
        else:
            raise NotImplementedError


class DistillatioModels(nn.Module):
    def __init__(self, student, teacher, loaders, info_log, device='cpu', callbacks=None, loss_fn=None, optimizer=None,
                 sched=None, metrics=None, loggers=None, AMP=True, as_encoder=True):
        super().__init__()
        """
        :param
            --model: complete Torch model to train/test
            --loaders: tuple with the Torch data_loaders like (train,val,test)
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
            --loggers: loggers instance
            --AMP: Automatic Mixed Precision 
            --freeze: list containing names of layers to freeze
            --sequences: to handle windowed input sequences
        """

        if isinstance(student, nn.Module):
            self.student = student
        else:
            raise TypeError("student not recognised")
        if isinstance(teacher, nn.Module):
            self.teacher = teacher
        else:
            raise TypeError("teacher not recognised")

        self.my_logger = info_log

        self.train_loader, self.val_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                self.my_logger.info('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        assert isinstance(info_log, logging.Logger), 'provided info_log is not a logger'
        self.my_logger = info_log

        self.values_to_find = torch.tensor([0, 1, 2])

        self.my_logger.info(f"loading models to device: {self.device}")
        self.student.to(self.device)
        self.teacher.to(self.device)  # I'm moving dynamically in train_one_epoch

        # freezing the whole teacher.
        self.teacher.eval()

        self.callbacks = callbacks

        self.metrics = metrics
        self.loggers = loggers

        if loss_fn:
            self.loss_fun = loss_fn.to(self.device)
            self.opt = optimizer
            self.sched = sched
        else:
            raise AttributeError('how the hell can I train without a loss... (loss_fn not defined)')

        if AMP and "cuda" in self.device:
            self.my_logger.info("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False

        self.encoder_only_teacher = as_encoder
        total_params = sum(p.numel() for p in self.student.parameters())
        self.my_logger.info(f"'{self.student.name}' - Total parameters: {total_params / 1e6:.2f}M")

    def train_one_epoch(self, epoch_index, tot_epochs):
        self.loss_fun.reset()

        # initializing progress bar
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if self.encoder_only_teacher:
                inputs = data

                sam_in = [x.numpy() for x in inputs.permute(0, 2, 3, 1)]  # SAM wants list of np.array
                with torch.no_grad():
                    labs = self.teacher(sam_in)
            else:
                inputs, labs, polys = data
                labs = labs.to(self.device)

            inputs = inputs.to(self.device)

            self.opt.zero_grad()

            if self.AMP:
                with autocast():

                    student_out = self.student(inputs)

                    loss = self.loss_fun(student_out, labs)  # for later also another input for labs I guess..

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                student_out = self.student(inputs)

                loss = self.loss_fun(student_out, labs)
                loss.backward()
                self.opt.step()

            # self.student.to('cpu')

            del inputs, sam_in

            current_loss = self.loss_fun.get_current_value(batch)

            with torch.no_grad():
                # computing training metrics
                self.metrics.on_train_batch_end(student_out.float(), labs, batch)
                # calling callbacks
                self.callbacks.on_train_batch_end(student_out.float(), labs, batch)

            # updating pbar
            if self.encoder_only_teacher:
                pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                            f'train_loss (mse): {current_loss:.4f}')
            else:
                raise AttributeError('training with decoder not yet implementeeeeeeddddddd.....')

            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(batch + 1)

    def val_loop(self, epoch):
        self.loss_fun.reset()

        #resetting metrics for validation
        self.metrics.on_val_start()

        # calling callbacks
        self.callbacks.on_val_start()

        # initializing progress bar
        description = f'Validation'
        pbar_loader = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # Disable gradient computation and reduce memory consumption, and model to evaluation mode.
        self.student.eval()
        with torch.no_grad():
            for batch, data in pbar_loader:
                torch.cuda.empty_cache()  # Clear GPU memory

                if self.encoder_only_teacher:
                    inputs = data

                    sam_in = [x.numpy() for x in inputs.permute(0, 2, 3, 1)]
                    labels = self.teacher(sam_in)
                else:
                    inputs, labels, polys = data
                    labels = labels.to(self.device)

                inputs = inputs.to(self.device)

                # self.teacher.to(self.device)

                # self.teacher.to('cpu')

                # self.student.to(self.device)
                student_out = self.student(inputs)
                # self.student.to('cpu')

                del inputs, sam_in

                _ = self.loss_fun(student_out, labels)

                current_loss = self.loss_fun.get_current_value(batch)

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(student_out.float(), labels, batch)
                # calling callbacks
                self.callbacks.on_val_batch_end(student_out, labels, batch)
                # updating roc and prc
                self.loggers.on_val_batch_end(student_out, labels, batch)

                # updating pbar
                if self.encoder_only_teacher:
                    description = f'Validation: val_loss (mse): {current_loss:.4f}'
                else:
                    raise AttributeError('not yet implemented the detectorrrr')
                pbar_loader.set_description(description)

        if student_out is not None:
            # updating metrics dict
            self.metrics.on_val_end(batch + 1)

            # updating loggers (roc, prc)
            self.loggers.on_val_end()
            # calling callbacks
            self.callbacks.on_val_end(self.metrics.dict, epoch)

    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            self.metrics.on_epoch_start()
            self.loggers.on_epoch_start(epoch=epoch, max_epoch=num_epochs)

            # setting the student to trainable
            self.student.train(True)

            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)

            # validation
            self.val_loop(epoch)

            # logging results
            self.loggers.on_epoch_end(epoch)
            # updating lr scheduler
            if self.sched:
                self.sched.step()

            #resetting metrics
            self.metrics.on_epoch_end()

            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                self.my_logger.info(f"early stopping at epoch {epoch}")
                break

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model) #
        self.callbacks.on_end()


class ModelTest(nn.Module):
    def __init__(self, model, loader, device='cpu', metrics=None, loggers=None, info_log=None):
        super().__init__()
        """
        :param
            --model: complete Torch model to test
            --loader: test loader
            --device: str for gpu or cpu
            --metrics: metrics instance for computing metrics callbacks
            --loggers: loggers instance
            --is_det: flag whether you are using a detection head
        """

        assert isinstance(info_log, logging.Logger), 'provided info_log is not a logger'
        self.my_logger = info_log

        if isinstance(model, nn.Module):
            self.model = model
            self.model.eval()
        else:
            raise TypeError("model not recognised")

        self.test_loader = loader
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                self.my_logger.info('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        self.my_logger.info(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.metrics = metrics

        self.loggers = loggers

        total_params = sum(p.numel() for p in self.model.parameters())
        self.my_logger.info(f"'{self.model.name}' - Total parameters: {total_params / 1e6:.2f}M")

    def test_loop(self):

        # initializing progress bar
        description = 'Test'
        pbar_loader = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        self.metrics.on_epoch_start()
        self.loggers.on_epoch_start(epoch=0, max_epoch=0)

        with torch.no_grad():
            for batch, data in pbar_loader:
                torch.cuda.empty_cache()  # Clear GPU memory
                gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)

                inputs, labs, poly_box = self.unpack_data(data)

                outputs = self.model(inputs)

                # computing training metrics
                self.metrics.on_val_batch_end(outputs, labs, batch)
                self.loggers.on_val_batch_end(outputs, inputs, poly_box)

                self.update_pbar(pbar_loader, gpu_used)

                if self.device != "cpu":
                    torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_val_end(batch + 1)

        # logging results
        self.loggers.on_epoch_end(0)

        # logging metrics images
        self.loggers.on_end()

    def update_pbar(self, pbar, gpu_used=None):
        # updating pbar
        A = self.metrics.A.t_value_mean
        P = self.metrics.P.t_value_mean
        R = self.metrics.R.t_value_mean
        # AUC = self.metrics.AuC.t_value[1]
        dice = self.metrics.Dice.t_value

        pbar.set_description(f'GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                             f'A: {A :.2f}, P: {P :.2f}, R: {R :.2f}, Dice: {dice: .2f}')

    def unpack_data(self, data):
        inputs, labels, poly_box = data
        inputs = inputs.to(self.device)

        labels = labels.to(self.device)

        return inputs, labels, poly_box


# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(str(Path(parent_dir) / 'sam2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# same in image handling becouse i'm having problem with imports....
def bbox_from_poly(masks_batch, return_dict=False):
    """
    Retrieves bounding boxes for each give polygonals.

    args:
        - masks_batch: list of masks polygons lists (masks polygons in same format as above [[(polys, id)...]...] )
        - return_dict: if True it returns a list of dicts (1 per img) where each key is a class containing list of top_left
                       and bottom_right corners of bboxes. Else it returns a list like [[(bbox, id)...]...] )
                       where bbox is a tuple containing xyxy coord of bbox
                       (NOTE it is different from polys, here 1 tuple x box)
    Returns:
        bboxes_list:
    """

    bboxes_list = []
    for masks_instance in masks_batch:
        bbox_dict = {}
        for masks, class_id in masks_instance:
            bboxes = []

            for polygon in masks:

                min_x, min_y = np.min(polygon, axis=0)
                max_x, max_y = np.max(polygon, axis=0)

                bboxes.append((min_x, min_y, max_x, max_y))
                bbox_dict[class_id] = bboxes

        # sorting classes for consistency
        if return_dict:
            bboxes_list.append({k: bbox_dict[k] for k in sorted(bbox_dict.keys())})
        else:
            bb = []
            for k in sorted(bbox_dict.keys()):
                bb += [(x, k) for x in bbox_dict[k]]
            bboxes_list.append(bb)
    return bboxes_list



class SAM2handler(nn.Module):
    def __init__(self, sam2_checkpoint, model_cfg, info_log, as_encoder=False):
        super().__init__()

        assert isinstance(info_log, logging.Logger), 'provided info_log is not a logger'
        my_logger = info_log
        my_logger.info(f'loading {Path(model_cfg).stem}...')

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device('cpu'))

        self.predictor = SAM2ImagePredictor(sam2_model)
        self.as_encoder = as_encoder
        self.model = self.predictor.model  # to account for the SAM2(nn.Model) being in predictor.model
        my_logger.info('SAM loaded...')

    def forward(self, x, y=None):
        self.predictor.set_image_batch(x)
        if self.as_encoder:
            return self.predictor.get_image_embedding()
        else:
            _, polys = y

            prompt_batch, ids = self.get_prompts(polys)
            # think about how to use scores for better loss computing
            # (maybe if the SAM score is low i can low yhe distillation loss and vice-versa idk...)
            logits, scores, _ = self.predictor.predict_batch(point_coords_batch=None, point_labels_batch=None,
                                                             box_batch=prompt_batch, multimask_output=False,
                                                             return_logits=True)
            # logits list [len N, [len bboxes, [np.array HxW]]]

    # to min_max the logits for stacking to comparable class volumes
    # (if I have to do semantic seg I need 4 output channels, one for each class plus bkg)
    def min_max(self, x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def get_prompts(self, polys):
        """

        args:
            - polys: list of masks polygons lists (masks polygons in format [[(polys, id)...]...] )
        returns:
            - prompt_batch: list of np.array containing bbox xyxy for each instance for each batch element (SAM shape)
            - ids: list of class id per bbox (for alignment)
        """

        bboxes = bbox_from_poly(polys)  # bboxes as list [[(bboxes, id)...]...]
        prompt_batch = []
        ids = []  # list with relative classes id of boxes in batch (for re-alinging output for loss computing)
        for i in bboxes:
            prompt_batch.append(np.array([[x[0], x[1], x[2], x[3]] for x, _ in i]))
            ids.append([l for _, l in i])

        return prompt_batch, ids

    # input_boxes = np.array([
    #     [75, 275, 1725, 850],
    #     [425, 600, 700, 875],
    #     [1375, 550, 1650, 800],
    #     [1240, 675, 1400, 750],
    # ])


