import os
import time
import shutil
import torch
from torch import optim
from trainer import Trainer
import numpy as np
import itertools
from model import ManifoldEncoderS
from model import ManifoldEncoderX
from model import ManifoldEncoderY
from dataloader import get_dataloader
from dataloader import get_pair_data_loader
from dataloader import  get_visualizeloader

class Args(object):
    def __init__(self):
        self.experiment_id = "ManifoldWarp" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)

        self.epoch = 300
        self.batch_size = 8

        self.gpu_mode = torch.cuda.is_available()
        self.verbose = False

        # model & optimizer
        self.modelS = ManifoldEncoderS()
        self.modelX = ManifoldEncoderX()
        self.modelY = ManifoldEncoderY()

        # self.pretrainS = 'snapshot/ManifoldWarp05171518/models/funcmanifold_S_best.pkl'
        # self.pretrainX = 'snapshot/ManifoldWarp05171518/models/funcmanifold_X_best.pkl'
        # self.pretrainY = 'snapshot/ManifoldWarp05171518/models/funcmanifold_Y_best.pkl'
        self.pretrainS = ''
        self.pretrainX = ''
        self.pretrainY = ''
        self.parameterS = self.modelS.parameters()
        self.parameterX = self.modelX.parameters()
        self.parameterY = self.modelY.parameters()
        self.optimizerS = optim.Adam(self.parameterS, lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-6)
        self.optimizerX = optim.Adam(self.parameterX, lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-6)
        self.optimizerY = optim.Adam(self.parameterY, lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-6)
        self.schedulerS = optim.lr_scheduler.ExponentialLR(self.optimizerS, gamma=0.5)
        self.schedulerX = optim.lr_scheduler.ExponentialLR(self.optimizerX, gamma=0.5)
        self.schedulerY = optim.lr_scheduler.ExponentialLR(self.optimizerY, gamma=0.5)
        self.scheduler_interval = 10

        self.train_loaderS = get_dataloader(split='S')
        self.train_loaderX = get_dataloader(split='X')
        self.train_loaderY = get_dataloader(split='Y')
        #self.visualize_loaderS = get_visualizeloader(split='S')
        #self.visualize_loaderX = get_visualizeloader(split='X')
        #self.visualize_loaderY = get_visualizeloader(split='Y')
        self.train_pair = get_pair_data_loader(b_size=self.batch_size)

        # snapshot
        self.snapshot_interval = 10
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    args = Args()
    trainer = Trainer(args)
    trainer.train()



