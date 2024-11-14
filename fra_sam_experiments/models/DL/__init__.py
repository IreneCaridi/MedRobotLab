import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import sys

# SETTING GLOBAL VARIABLES
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


# ----------------------------------------------------------------------------------------------------------------------
# GENERAL MODEL CLASS FOR HANDLING TRAINING, VALIDATION AND INFERENCE
# ----------------------------------------------------------------------------------------------------------------------


class ModelClass(nn.Module):
    def __init__(self, model, loaders, device='cpu', callbacks=None, loss_fn=None, optimizer=None, sched=None,
                 metrics=None, loggers=None, AMP=True, freeze=None, sequences=False, bi_head=False):
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

        self.freeze = freeze
        self.seq = sequences
        self.bi_head = bi_head

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError("model not recognised")

        self.train_loader, self.val_loader, self.test_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                print('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        self.values_to_find = torch.tensor([0, 1, 2])

        print(f"loading model to device={self.device}")
        self.model.to(self.device)

        self.callbacks = callbacks

        if self.bi_head:
            self.metrics = metrics[1]  # so I plot all head to pbar
            self.metrics1 = metrics[0]  # binary head
        else:
            self.metrics = metrics
            self.metrics1 = None
        self.loggers = loggers

        if loss_fn:
            self.loss_fun = loss_fn.to(self.device)
            self.opt = optimizer
            self.sched = sched

        if AMP and "cuda" in self.device:
            print("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False


    def train_one_epoch(self,epoch_index,tot_epochs):
        running_loss = 0.
        last_loss = 0.

        # initializing progress bar
        description = 'Training'
        pbar_loader = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)

        # train over batches
        for batch, data in pbar_loader:
            torch.cuda.empty_cache()  # Clear GPU memory
            gpu_used = torch.cuda.max_memory_allocated() / (1024 ** 3)

            if self.seq:
                inputs, labs = data
                inputs, labs = self.get_seq_input(inputs, labs)
            else:
                inputs, labs = data
                inputs = inputs.to(self.device)
                labs = labs.to(self.device)

            self.opt.zero_grad()

            if self.AMP:
                with autocast():
                    outputs = self.model(inputs)
                    if self.bi_head:
                        loss, outputs, labs = self.handle_both(outputs, labs)
                    else:
                        loss = self.loss_fun(outputs, labs)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                outputs = self.model(inputs)
                if self.bi_head:
                    loss, outputs, labs = self.handle_both(outputs, labs)
                else:
                    loss = self.loss_fun(outputs, labs)
                loss.backward()
                self.opt.step()

            del inputs

            running_loss += loss.item()
            last_loss = running_loss #/ self.train_loader.batch_size  # loss per batch
            running_loss = 0.

            with torch.no_grad():
                # computing training metrics
                if self.bi_head:
                    if outputs[1] is not None:
                        self.metrics.on_train_batch_end(outputs[1].float(), labs[1], batch)
                        # calling callbacks
                        self.callbacks.on_train_batch_end(outputs[1].float(), labs[1], batch)
                    self.metrics1.on_train_batch_end(outputs[0].float(), labs[0], batch)
                else:
                    self.metrics.on_train_batch_end(outputs.float(), labs, batch)
                    # calling callbacks
                    self.callbacks.on_train_batch_end(outputs.float(), labs, batch)

            # updating pbar
            if self.metrics.num_classes != 2:
                A = self.metrics.A.t_value_mean
                P = self.metrics.P.t_value_mean
                R = self.metrics.R.t_value_mean
                AUC = self.metrics.AuC.t_value_mean
            else:
                A = self.metrics.A.t_value_mean
                P = self.metrics.P.t_value[1]
                R = self.metrics.R.t_value[1]
                AUC = self.metrics.AuC.t_value[1]

            pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                        f'train_loss: {last_loss:.4f}, A: {A :.2f}, P: {P :.2f}, R: {R :.2f}, AUC: {AUC :.2f}')
            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(last_loss)
        if self.bi_head:
            self.metrics1.on_train_end(last_loss)

    def val_loop(self, epoch):
        running_loss = 0.0
        last_loss = 0.0

        #resetting metrics for validation
        self.metrics.on_val_start()
        if self.bi_head:
            self.metrics1.on_val_start()
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

                if self.seq:
                    inputs, labels = data
                    inputs, labels = self.get_seq_input(inputs, labels)
                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                outputs = self.model(inputs)
                if self.bi_head:
                    loss, outputs, labels = self.handle_both(outputs, labels)
                else:
                    loss = self.loss_fun(outputs, labels)

                running_loss += loss.item()
                last_loss = running_loss #/ self.val_loader.batch_size  # loss per batch
                running_loss = 0.0

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                if self.bi_head:
                    if outputs[1] is not None:
                        self.metrics.on_val_batch_end(outputs[1].float(), labels[1], batch)
                        # calling callbacks
                        self.callbacks.on_val_batch_end(outputs[1].float(), labels[1], batch)
                    self.metrics1.on_val_batch_end(outputs[0].float(), labels[0], batch)
                    self.loggers.on_val_batch_end(outputs[0], labels[0], batch)
                else:
                    self.metrics.on_val_batch_end(outputs.float(), labels, batch)
                    # calling callbacks
                    self.callbacks.on_val_batch_end(outputs, labels, batch)
                    # updating roc and prc
                    self.loggers.on_val_batch_end(outputs, labels, batch)

                # updating pbar
                if self.metrics.num_classes != 2:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value_mean
                    R = self.metrics.R.v_value_mean
                    AUC = self.metrics.AuC.v_value_mean
                else:
                    A = self.metrics.A.v_value_mean
                    P = self.metrics.P.v_value[1]
                    R = self.metrics.R.v_value[1]
                    AUC = self.metrics.AuC.v_value[1]
                description = f'Validation: val_loss: {last_loss:.4f}, val_A: {A :.2f}, ' \
                              f'val_P: {P :.2f}, val_R: {R :.2f}, val_AUC: {AUC :.2f}'
                pbar_loader.set_description(description)

        if outputs is not None:
            # updating metrics dict
            self.metrics.on_val_end(last_loss)
            if self.bi_head:
                self.metrics1.on_val_end(last_loss)
            # updating loggers (roc, prc)
            self.loggers.on_val_end()
            # calling callbacks
            self.callbacks.on_val_end(self.metrics.dict, epoch)

    def train_loop(self, num_epochs):

        for epoch in range(num_epochs):

            torch.cuda.empty_cache()

            self.metrics.on_epoch_start()
            if self.bi_head:
                self.metrics1.on_epoch_start()
            self.loggers.on_epoch_start(epoch=epoch, max_epoch=num_epochs)

            # self.model.train(True)
            self.check_freeze()  # freezing specific layers (if needed)

            # for name, param in self.model.named_parameters():
            #     print(name, param.requires_grad)

            # reshuffle for subsampling
            self.train_loader.dataset.build()
            # 1 epoch train
            self.train_one_epoch(epoch, num_epochs)

            # reshuffle for subsampling
            self.val_loader.dataset.build()
            # validation
            self.val_loop(epoch)

            # logging results
            self.loggers.on_epoch_end()
            # updating lr scheduler
            if self.sched:
                self.sched.step()
            #resetting metrics
            self.metrics.on_epoch_end()
            if self.bi_head:
                self.metrics1.on_epoch_end()
            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                print(f"early stopping at epoch {epoch}")
                break

        # self.loggers.on_epoch_end(0)

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model)
        self.callbacks.on_end()

    def inference(self, return_preds=False):
        self.model.train(False)
        outputs = []

        model = nn.Sequential(self.model,
                              nn.Softmax(1))

        # self.loggers.on_epoch_start(0, 1)

        # resetting metrics for validation
        # self.metrics.on_val_start()

        # initilialize progress bar
        description = f'Test'

        pbar_loader = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=description, unit='batch',
                           bar_format=TQDM_BAR_FORMAT)
        with torch.no_grad():
            for batch, data in pbar_loader:

                if self.seq:
                    inputs, labels = data
                    inputs, labels = self.get_seq_input(inputs, labels)
                else:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    #labels = labels.to(self.device)

                output = model(inputs)
                pred = output.to('cpu')
                outputs.append(pred)

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                # self.metrics.on_val_batch_end(output, labels, batch)
                # self.loggers.on_val_batch_end(output, labels, batch)

                # updating pbar
                # if self.metrics.num_classes != 2:
                #     A = self.metrics.A.v_value_mean
                #     P = self.metrics.P.v_value_mean
                #     R = self.metrics.R.v_value_mean
                #     AUC = self.metrics.AuC.v_value_mean
                # else:
                #     A = self.metrics.A.v_value_mean
                #     P = self.metrics.P.v_value[1]
                #     R = self.metrics.R.v_value[1]
                #     AUC = self.metrics.AuC.v_value[1]
                # description = f'item: {batch}/{len(self.test_loader)-1}, A: {A :.2f}, ' \
                #               f'P: {P :.2f}, R: {R :.2f}, AUC: {AUC :.2f}'
                # pbar_loader.set_description(description)

        # print TEST RESULTS
        # resetting metrics
        # self.metrics.on_val_end(0)
        # self.loggers.on_val_end()
        # self.loggers.on_epoch_end()
        # self.metrics.on_epoch_end()
        #
        # self.loggers.on_end()

        if return_preds:
            return outputs

    def check_freeze(self):
        if self.freeze:
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in self.freeze):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            self.model.train(True)

    def get_seq_input(self, inputs, labels):
        labels = labels
        indices = torch.where(labels == self.values_to_find)

        # on_idx = torch.where(labels == torch.tensor([7]))
        # on = on_idx[1] // 16

        inputs = (inputs.to(self.device), labels.to(self.device))
        labs = indices[2].to(self.device)
        return inputs, labs

    def handle_both(self, outputs, labels):
        o1, o2_pack = outputs
        o2, idx = o2_pack
        lab1 = labels[:, 0]
        lab2 = labels[:, 1]

        # print(labels.shape, lab1.shape, lab2.shape)
        loss1 = self.loss_fun(o1, lab1)
        if o2 is not None:
            loss2 = self.loss_fun(o2, lab2[idx])

            return (loss1+loss2)/2, (o1, o2), (lab1, lab2[idx])
        else:
            return loss1, (o1, o2), (lab1, lab2[idx])


def check_load_model(model, backbone_weights):
    if not backbone_weights:
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, str) and Path(model).suffix == ".pt" or ".pth":
            return torch.load(model, map_location=torch.device('cpu'))
        else:
            raise TypeError("model not recognised")
    else:
        # I'm loading only the weights from the backbone

        assert isinstance(model, nn.Module)  # check that the model is something to load weights to

        old = torch.load(backbone_weights)
        filtered_state_dict = {k: old.state_dict()[k] for k in old.state_dict() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)

        return model


class DistillatioModels(nn.Module):
    def __init__(self, student, teacher, loaders, device='cpu', callbacks=None, loss_fn=None, optimizer=None, sched=None,
                 metrics=None, loggers=None, AMP=True, as_encoder=True):
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

        self.train_loader, self.val_loader = loaders
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_properties = torch.cuda.get_device_properties(self.device)
                self.gpu_mem = gpu_properties.total_memory / (1024 ** 3)
            else:
                print('no gpu found')
                self.gpu_mem = 0
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_mem = 0

        self.values_to_find = torch.tensor([0, 1, 2])

        print(f"loading models to device: {self.device}")
        self.student.to(self.device)
        # self.teacher.to(self.device)  # I'm moving dynamically in train_one_epoch

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
            print("eneabling Automatic Mixed Precision (AMP)")
            self.AMP = True
            self.scaler = GradScaler()
        else:
            self.AMP = False

        self.encoder_only_teacher = as_encoder

    def train_one_epoch(self,epoch_index,tot_epochs):
        running_loss = 0.
        last_loss = 0.

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
            else:
                inputs, labs = data
                labs = labs.to(self.device)

            sam_in = [x.numpy() for x in inputs.permute(0, 2, 3, 1)]
            inputs = inputs.to(self.device)

            self.opt.zero_grad()

            # self.teacher.to(self.device)
            with torch.no_grad():
                teacher_out = self.teacher(sam_in)
            # self.teacher.to('cpu')

            # self.student.to(self.device)
            if self.AMP:
                with autocast():

                    student_out = self.student(inputs)

                    loss = self.loss_fun(student_out, teacher_out)  # for later also another input for labs I guess..

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
            else:
                student_out = self.student(inputs)

                loss = self.loss_fun(student_out, teacher_out)
                loss.backward()
                self.opt.step()

            # self.student.to('cpu')

            del inputs, sam_in

            running_loss += loss.item()
            last_loss = running_loss #/ self.train_loader.batch_size  # loss per batch
            running_loss = 0.

            with torch.no_grad():
                # computing training metrics
                self.metrics.on_train_batch_end(student_out.float(), teacher_out, batch)
                # calling callbacks
                self.callbacks.on_train_batch_end(student_out.float(), teacher_out, batch)

            # updating pbar
            if self.encoder_only_teacher:
                pbar_loader.set_description(f'Epoch {epoch_index}/{tot_epochs-1}, GPU_mem: {gpu_used:.2f}/{self.gpu_mem:.2f}, '
                                            f'train_loss (mse): {last_loss:.4f}')
            else:
                raise AttributeError('training with decoder not yet implementeeeeeeddddddd.....')

            if self.device != "cpu":
                torch.cuda.synchronize()

        # updating dictionary
        self.metrics.on_train_end(last_loss)

    def val_loop(self, epoch):
        running_loss = 0.0
        last_loss = 0.0

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
                else:
                    inputs, labels = data
                    labels = labels.to(self.device)

                sam_in = [x.numpy() for x in inputs.permute(0, 2, 3, 1)]
                inputs = inputs.to(self.device)

                # self.teacher.to(self.device)
                teacher_out = self.teacher(sam_in)
                # self.teacher.to('cpu')

                # self.student.to(self.device)
                student_out = self.student(inputs)
                # self.student.to('cpu')

                del inputs, sam_in

                loss = self.loss_fun(student_out, teacher_out)

                running_loss += loss.item()
                last_loss = running_loss #/ self.val_loader.batch_size  # loss per batch
                running_loss = 0.0

                if self.device != "cpu":
                    torch.cuda.synchronize()

                # computing metrics on batch
                self.metrics.on_val_batch_end(student_out.float(), teacher_out, batch)
                # calling callbacks
                self.callbacks.on_val_batch_end(student_out, teacher_out, batch)
                # updating roc and prc
                self.loggers.on_val_batch_end(student_out, teacher_out, batch)

                # updating pbar
                if self.encoder_only_teacher:
                    description = f'Validation: val_loss (mse): {last_loss:.4f}'
                else:
                    raise AttributeError('not yet implemented the detectorrrr')
                pbar_loader.set_description(description)

        if student_out is not None:
            # updating metrics dict
            self.metrics.on_val_end(last_loss)

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
            self.loggers.on_epoch_end()
            # updating lr scheduler
            if self.sched:
                self.sched.step()

            #resetting metrics
            self.metrics.on_epoch_end()

            # calling callbacks
            try:
                self.callbacks.on_epoch_end(epoch)
            except StopIteration:  # (early stopping)
                print(f"early stopping at epoch {epoch}")
                break

        # self.loggers.on_epoch_end(0)

        # logging metrics images
        self.loggers.on_end()
        # calling callbacks (saving last model) #
        self.callbacks.on_end()


# placing myself in sam2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(str(Path(parent_dir) / 'sam2'))


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2handler(nn.Module):
    def __init__(self, sam2_checkpoint, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", as_encoder=False):
        super().__init__()

        print(f'loading {Path(model_cfg).stem}...')
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=torch.device('cpu'))

        self.predictor = SAM2ImagePredictor(sam2_model)
        self.as_encoder = as_encoder
        self.model = self.predictor.model  # to account for the SAM2(nn.Model) being in predictor.model
        print('SAM loaded...')

    def forward(self, x):
        self.predictor.set_image_batch(x)
        if self.as_encoder:
            return self.predictor.get_image_embedding()
        else:
            # to be done:
            # points_batch, label_batch = self.get_prompts()
            # return self.predictor.predict_batch(points_batch, label_batch, multimask_output=False)
            pass

    def get_prompts(self):
        pass
        # fare cose
