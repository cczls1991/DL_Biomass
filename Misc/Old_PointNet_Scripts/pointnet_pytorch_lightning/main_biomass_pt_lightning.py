#Notes:
        #Updated script to include intensity as a predictor
        #Updated to use 10_000 points instead of 4_000

#Get modules
import os

import numpy as np
from time import time

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.loader import DataLoader

from pn2_regressor_pt_lightning import Net
from datamodule_pt_lightning import PointCloudsInFiles
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import R2Score

if __name__ == '__main__':

    # Record start time
    start = time()

    # Set up tensorboard summary writer
    writer = SummaryWriter(comment="_10000_points_attempt_1")

    # Set additional variables to be used for each point in LAS
    use_columns = None #['intensity']

    # Numer of features is determined by the number of used columns
    if use_columns is None:
        num_features = 0
    else:
        num_features = len(use_columns)

    #Set number of epochs
    n_epochs = 200

    #Set the number of points to be used per plot
    n_points = 10_000

    #Import data
    train_dataset = PointCloudsInFiles(r'/Romeo_Data/train',
                                       '*.las',
                                       max_points=n_points, use_columns=use_columns)
    test_dataset = PointCloudsInFiles(r'/Romeo_Data/test',
                                      '*.las',
                                      max_points=n_points, use_columns=use_columns)

    print(f"We have {len(train_dataset)} training samples")

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
                             num_workers=0)


    #Set model
    model = Net(num_features=num_features)

    #Set up trainer
    trainer = pl.Trainer(max_epochs = 5)

    #Run training loop
    trainer.fit(model, train_loader, test_loader)

    print("Training finished, all ok")