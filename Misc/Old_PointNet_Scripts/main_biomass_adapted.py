import os

import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from pn2_regressor_biomass_adapted import Net
from pointcloud_dataset_biomass_adapted import PointCloudsInFiles
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import R2Score

#Set up tensorboard summary writer
writer = SummaryWriter(comment="_Initial_PointNet_Runs")

#Read in the target data
if __name__ == '__main__':

    train_dataset = PointCloudsInFiles(r'D:\Romeo_Data\train',
                                       '*.las',
                                        max_points=4_000, use_columns=None)
    test_dataset = PointCloudsInFiles(r'D:\Romeo_Data\test',
                                      '*.las',
                                        max_points=4_000, use_columns=None)

    print(f"We have {len(train_dataset)} training samples")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = Net(num_features=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    def test(loader, ep_id):
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

    for epoch in range(1, 10):
        model_path = rf'D:\Sync\DL_Development\Models\latest.model'
        if os.path.exists(model_path):
            model = torch.load(model_path)
        train()
        mse, r2 = test(test_loader, epoch)

        torch.save(model, model_path)

        # Save details for epoch to summary report
        writer.add_scalar("MSE", mse, epoch)  # Save the Mean Squared Error (MSE)
        writer.add_scalar("Mean R2", r2, epoch)  # Save the mean R-Squared (R2) value

        print(f'Epoch: {epoch:02d}, Mean test MSE: {mse:.4f}')

    # Call flush() method to make sure that all pending events have been written to disk.
    writer.flush()
    #Close summary writer
    writer.close()

