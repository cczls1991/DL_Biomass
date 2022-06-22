#Notes:
        #Updated script to include intensity as a predictor
        #Updated to use 10_000 points instead of 4_000

#Get modules  
import os

import numpy as np
from time import time

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from pn2_regressor_biomass_adapted import Net
from pointcloud_dataset_biomass_adapted import PointCloudsInFiles
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import R2Score

if __name__ == '__main__':

    # Record start time
    start = time()

    # Set up tensorboard summary writer
    writer = SummaryWriter(comment="_10000_points_lr_0.0001_batch_size_1")

    # Set additional variables to be used for each point in LAS
    use_columns = None #['intensity']

    # Numer of features is determined by the number of used columns
    if use_columns is None:
        num_features = 0
    else:
        num_features = len(use_columns)

    #Set number of epochs
    n_epochs = 100

    #Set the number of points to be used per plot
    n_points = 10_000

    #Import data
    train_dataset = PointCloudsInFiles(r'/Romeo_Data/train',
                                       '*.las',
                                       max_points=n_points, use_columns=use_columns)
    val_dataset = PointCloudsInFiles(r'/Romeo_Data/val',
                                      '*.las',
                                     max_points=n_points, use_columns=use_columns)

    print(f"We have {len(train_dataset)} training samples")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                             num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = Net(num_features=num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #Define function for training model
    def train():
        model.train()

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out[:, 0], data.y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} ')

    @torch.no_grad()
    def val(loader, ep_id):
        model.eval()
        losses = []
        r2scores = []
        for idx, data in enumerate(loader):

            #Send data to cuda and apply model
            data = data.to(device)
            outs = model(data)

            #Get MSE and use as loss function
            loss = F.mse_loss(outs[:, 0], data.y)
            losses.append(float(loss.to("cpu")))

            #Get R^2 as well, just for info (not used as loss)
            r2score = R2Score().to(device)
            r2score = r2score(outs[:, 0], data.y)
            r2scores.append(float(r2score.to("cpu")))

        return float(np.mean(losses)),  float(np.mean(r2scores))

    for epoch in range(1, n_epochs):
        model_path = rf'/DL_Development/Models/latest.model'
        if os.path.exists(model_path):
            model = torch.load(model_path)
        train()
        mse, r2 = val(val_loader, epoch)

        # Save details for epoch to summary report
        writer.add_scalar("MSE", mse, epoch)  # Save the Mean Squared Error (MSE)
        writer.add_scalar("Mean R2", r2, epoch)  # Save the mean R-Squared (R2) value
        torch.save(model, model_path)
        print(f'Epoch: {epoch:02d}, Mean val MSE: {mse:.4f}')

    print(f'Script run time: {round(time()/3600 - start/3600, 2)}', "hours over", n_epochs, "epochs")

    # Call flush() method to make sure that all pending events have been written to disk.
    writer.flush()
    #Close summary writer
    writer.close()

