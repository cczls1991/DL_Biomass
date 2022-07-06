import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pn2_regressor_V3 import Net
from pointcloud_dataset_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import glob
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as metrics
from math import sqrt
import pprint as pp

if __name__ == '__main__':

    #Specify hyperparameter file
    hp_file = "D:\Sync\DL_Development\Hyperparameter_Tuning\Sunday_Monday_more_params__Hyperparameter_tuning_results_2022_06_27_08_30_41_.csv"
    #...or get most recent hyperparameter tuning
    if hp_file is None:
        folder_path = r'D:\Sync\DL_Development\Hyperparameter_Tuning'
        file_type = r'\*.csv'
        files = glob.glob(folder_path + file_type)
        hp_file = max(files, key=os.path.getctime)

    # Load hyperparameter tunings
    hp = pd.read_csv(hp_file, sep=",", header=0, index_col=0)
    hp = hp.sort_values(by="value", ascending=True)
    #Drop unwated cols
    hp = hp.drop(labels=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], axis=1)
    #Select the params with the lowest MSE value (row 1)
    hp = hp.iloc[0]
    #Convert to dictionary
    hp = hp.to_dict()
    print("Hyperparameters from:\n", hp_file, "\n")
    pp.pprint(hp, width=1)

    #SETUP ADDITIONAL HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    early_stopping = True
    num_epochs = 200
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'
    val_dataset_path = r'D:\Sync\Data\Model_Input\val'

    #Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(num_features=len(use_columns),
                activation_function=hp['activation_function'],
                neuron_multiplier=hp['neuron_multiplier'],
                dropout_probability=hp['dropout_probability']
                ).to(device)

    # Set optimizer
    if hp['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    #Set device
    print(f"Using {device} device.")

#Define training function
    def train(train_loader):
        model.train()
        loss_list = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)[:, 0]
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            print(str(f'[{idx + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} '))
            loss_list.append(loss.detach().to("cpu").numpy())
            print('Mean loss this epoch:', np.mean(loss_list))
        return np.mean(loss_list)

    def val(val_loader, ep_id): #Note sure what ep_id does
        with torch.no_grad():
            model.eval()
            losses = []
            for idx, data in enumerate(val_loader):
                data = data.to(device)
                outs = model(data)[:, 0]
                loss = F.mse_loss(outs, data.y)
                losses.append(float(loss.to("cpu")))
            return float(np.mean(losses))

    def main(num_points):

        # Get training data
        train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                           filter_height=hp['ground_filter_height'])
        val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                         filter_height=hp['ground_filter_height'])

        # Augment training data
        if hp['num_augs'] > 0:
            for i in range(hp['num_augs']):
                aug_trainset = AugmentPointCloudsInFiles(
                    train_dataset_path,
                    "*.las",
                    max_points=num_points,
                    use_columns=use_columns,
                )

                # Concat training and augmented training datasets
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
            print(
                f"Adding {hp['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

        # Set up pytorch training and validation loaders
        train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=0)

        #Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        #Training loop
        for epoch in tqdm(range(0, num_epochs), colour="green"):
            train_mse = train(train_loader)
            val_mse = val(val_loader, epoch)

            #Early stopping
            if early_stopping is True:
                if val_mse > last_val_mse:
                    trigger_times += 1
                    print("    Early stopping trigger " + str(trigger_times) + " out of " + str(hp['patience']))
                    if trigger_times >= hp['patience']:
                        print(f'\nEarly stopping at epoch {epoch}!\n')
                        return min(val_mse_list)
                else:
                    trigger_times = 0
                    last_val_mse = val_mse

            #Report epoch stats
            print("    Epoch: " + str(epoch) + "  | Mean val MSE: " + str(round(val_mse, 2)) + "  | Mean train MSE: " + str(round(train_mse, 2)))

            #Determine whether to save the model based on val MSE
            val_mse_list.append(val_mse)


        print(f"\nFinished all {num_epochs} training epochs.")

        return min(val_mse_list)

    #Run training loop with different numbers of input points

    point_num_range = range(500, 500, 30000)
    point_num_val_mse_list = []
    for point_num in point_num_range:
        mse = main(point_num)
        point_num_val_mse_list.append(mse)
        print("Done density: ", point_num)

    #Convert to df
    data = [point_num_range, point_num_val_mse_list]
    df = pd.DataFrame(data, columns=['point_num', 'val_mse'])
    t_now = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    df.to_csv(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\Point_density_effect_results_{t_now}_.csv")

    print('\nDone')