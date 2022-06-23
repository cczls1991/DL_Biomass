import os
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from pn2_regressor_biomass_adapted_V2 import Net
from pointcloud_dataset_biomass_adapted_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles

from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import glob
import pandas as pd


if __name__ == '__main__':

    #SETUP PARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    num_points = 10_000
    augment = True
    num_augs = 2
    batch_size = 13
    num_epochs = 100
    writer = SummaryWriter(comment="_10000_pts_n_lr_0.1_n_weight.decay_0.01_n_batch.size_12_n_num.augs_2_w_doubled_radius")
    train_dataset_path = r'D:\Sync\Romeo_Data\train'
    val_dataset_path = r'D:\Sync\Romeo_Data\val'

    #Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=len(use_columns)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)

    #Set device
    print(f"Using {device} device.")

    #Get training and val datasets
    train_dataset = PointCloudsInFiles(train_dataset_path,'*.las', max_points=num_points, use_columns=use_columns, filter_ground=True, filter_height=1.3)
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns, filter_ground=True, filter_height=1.3)

    #Augment training data
    if augment is True:
        for i in range(num_augs):
            aug_trainset = AugmentPointCloudsInFiles(
                train_dataset_path,
                "*.las",
                max_points=num_points,
                use_columns=use_columns,
            )

            # Concat training and augmented training datasets
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
            print(f"Adding {i+1} augmentation of {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

    #Set up pytorch training and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#Define training function
    def train():
        model.train()
        loss_list = []
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)[:, 0]
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} ')
                loss_list.append(loss.detach().to("cpu").numpy())
        print(f'mean loss this epoch: {np.mean(loss_list)}')
        return np.mean(loss_list)

    @torch.no_grad()
    def val(loader, ep_id):
        model.eval()
        losses = []
        for idx, data in enumerate(loader):
            data = data.to(device)
            outs = model(data)[:, 0]
            loss = F.mse_loss(outs, data.y)
            print(data.y.to('cpu').numpy() - outs.to('cpu').numpy())
            losses.append(float(loss.to("cpu")))
        return float(np.mean(losses))


    for epoch in range(0, num_epochs):
        model_path = model_path
        if os.path.exists(model_path):
            model = torch.load(model_path)
        train_mse = train()
        val_mse = val(val_loader, epoch)
        writer.add_scalar("Training MSE", train_mse, epoch)  # Save to tensorboard summary
        writer.add_scalar("Validation MSE", val_mse, epoch)  # Save to tensorboard summary
        torch.save(model, model_path)
        with open(model_path.replace('.model', '.csv'), 'a') as f:
            f.write(
            f'{epoch}, {train_mse}, {val_mse}\n'
            )
        print(f'Epoch: {epoch:02d}, Mean val MSE: {val_mse:.4f}')




    # Call flush() method to make sure that all pending events have been written to disk.
    writer.flush()
    #Close summary writer
    writer.close()



    #Plot the change in training and validation MSE --------------------------------------------------

    #Load most recent set of training data
    folder_path = r'/DL_Development/Models'
    file_type = r'\*.csv'
    files = glob.glob(folder_path + file_type)
    training_data = max(files, key=os.path.getctime)
    training_data = pd.read_csv(training_data, sep=",", header=None)
    training_data.columns = ['epoch', 'train_mse', 'val_mse']

    #Plot the change in training and validation mse over time
    fig,ax=plt.subplots()
    ax.plot(training_data["epoch"], training_data["train_mse"],color="blue",marker = "o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.plot(training_data["epoch"], training_data["val_mse"],color="red",marker="o")
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Training')
    plt.legend(handles=[red_patch, blue_patch])


    #Apply the model to the test dataset and plot the results --------------------------------------------------

    #Select a model to use:
    model_file = None

    #Load most recent model
    if model_file is None:
        folder_path = r'/DL_Development/Models'
        file_type = r'\*.model'
        files = glob.glob(folder_path + file_type)
        model_file = max(files, key=os.path.getctime)
        model = torch.load(model_file)
    else:
        model = torch.load(model_file)
    print("Using model:", model_file)

    # Get test data
    test_dataset = PointCloudsInFiles(r"/Romeo_Data/test", '*.las', max_points=num_points, use_columns=use_columns,
                                      filter_ground=True, filter_height=1.3)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    #Apply the model
    model.eval()
    for idx, data in enumerate(test_loader):
         data = data.to(device)
    pred = model(data)[:, 0].to('cpu').detach().numpy()
    obs = data.y.to('cpu').detach().numpy()


    #Get residuals
    resid = obs - pred

    #Plot observed vs. predicted
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(obs, pred)
    ax1.set(xlabel='Observed Biomass (Tons)', ylabel='Predicted Biomass (Tons)')

    #Plot residuals
    ax2.scatter(pred, resid)
    ax2.set(xlabel='Predicted Biomass (Tons)', ylabel='Residuals')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.5)
    plt.show()


