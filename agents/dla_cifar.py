#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 09:37
# @Author  : yangqiang
# @File    : dla_cifar.py
import torch
from torch import nn
import torch.optim as optim

from agents.base import BaseAgent
# from graphs.models.dla import DLA
from graphs.models.cifar_cuda_convnet import Net
from data_loader.cifar10 import Cifar10DataLoader
from utils.torch_utils import select_device, seed_everything

import shutil, os
from easyPlog import Plog

from sklearn.metrics import confusion_matrix


class DLAAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # 日志
        self.logger = Plog(os.path.join(self.config.log_dir, self.config.log_name), msgOnly=False, stream=True)

        # seed everything
        seed_everything(self.config.seed)

        # define device
        self.device = select_device()

        # define models
        self.model = Net().to(self.device)

        # define data_loader
        self.data_loader = Cifar10DataLoader(config=self.config)

        # define loss
        self.loss = nn.CrossEntropyLoss().to(self.device)

        # define optimizer and schedule
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        # self.best_metric = 0
        self.best_valid_mean_iu = 0

        # Model Loading from the latest checkpoint if not found start from scratch.
        # self.load_checkpoint("data/dla_cifar/ckpt/best-0.7403.pt")
        # Summary Writer
        # self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        ckpt = torch.load(file_name, map_location="cpu")
        self.model.load_state_dict(ckpt['state_dict'])
        # self.optimizer.load_state_dict(ckpt['optimizer'])

    def save_checkpoint(self, filename="model.pt", is_best=False):
        # Save the state
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(state, f"{self.config.checkpoint_dir}/{filename}")
        # if is_best:
        #     shutil.copyfile(f"{self.config.checkpoint_dir}/{filename}", f"{self.config.checkpoint_dir}/best.pt")

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            mean_acc = self.validate()
            self.current_epoch += 1
            is_best = mean_acc > self.best_valid_mean_iu
            if is_best:
                self.best_valid_mean_iu = mean_acc
            # self.save_checkpoint(filename=self.config.checkpoint_name, is_best=is_best)
        self.logger.info(f"best acc: {self.best_valid_mean_iu}")

    def train_one_epoch(self):
        """
        One epoch of training
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} lr:{}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            self.current_iteration += 1
        self.scheduler.step()

    def validate(self):
        """
        One cycle of model validation
        """
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, target in self.data_loader.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                y_true.extend(target.cpu().tolist())
                output = output.cpu().max(1, keepdim=True)[1]
                y_pred.extend(output.squeeze().tolist())

        cm = confusion_matrix(y_true, y_pred)
        mean_acc = cm.diagonal().sum()/10000
        self.logger.info(f"\n{cm}")
        self.logger.info(f"acc of 10 classes: {cm.diagonal()/1000}")
        self.logger.info(f"mean acc: {mean_acc}")

        return mean_acc

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass

    def inference(self, img):
        pass
