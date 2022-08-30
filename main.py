import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from pointcloud_dataloader import PointCloudsInFilesPreSampled
from augmentation import AugmentPointCloudsInFiles
from augmentation import AugmentPreSampledPoints
from datetime import datetime as dt
from torch_geometric.nn import DataParallel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import pandas as pd
from tqdm import tqdm
import pprint as pp
from testing_model import test_model
from pathlib import Path

# Supress warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # SETUP STATIC HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M")}.model'
    use_columns = ['intensity_normalized']
    use_datasets = ["BC", "RM", "PF"]  # Possible datasets: BC, RM, PF
    early_stopping = True
    num_epochs = 100
    use_presampled = True

    # Specify hyperparameter tunings
    hp = {'lr': 0.00179966410046844,
          'weight_decay': 8.0250963438986e-05,
          'num_points': 7168,  # This is currently overwritten if use_presampled is True,
          'batch_size': 36,
          'num_augs': 10,
          'patience': 10,
          'ground_filter_height': 0,
          'activation_function': "ReLU",
          'neuron_multiplier': 0,
          'dropout_probability': 0.5
          }



    # Specify dataset paths (dependent on whether or not using pre-sampled)
    if use_presampled is True:
        print("\nUsing pre-sampled points!\n")
        train_dataset_path = fr'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_train'
        val_dataset_path = r'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_val'
        test_dataset_path = r'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_test'
    else:
        train_dataset_path = r'D:\Sync\Data\Model_Input\train'
        val_dataset_path = r'D:\Sync\Data\Model_Input\val'
        test_dataset_path = r'D:\Sync\Data\Model_Input\test'

    # Report additional hyperparameters
    print(f"Dataset(s): {use_datasets}")
    print(f"Additional features used: {use_columns}")
    print(f"Using {hp['num_points']} points per plot")
    print(f"Early stopping: {early_stopping}")
    print(f"Max number of epochs: {num_epochs}")

    print("\nHyperparameters:\n")
    pp.pprint(hp, width=1)

    # Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model
    model = Net(num_features=len(use_columns),
                activation_function=hp['activation_function'],
                neuron_multiplier=hp['neuron_multiplier'],
                dropout_probability=hp['dropout_probability']
                )

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    if use_presampled == True:
        # Load pre sampled data
        train_dataset = PointCloudsInFilesPreSampled(train_dataset_path,
                                                     '*.las', dataset=use_datasets, use_column="intensity_normalized")
        val_dataset = PointCloudsInFilesPreSampled(val_dataset_path,
                                                   '*.las', dataset=use_datasets, use_column="intensity_normalized")
        test_dataset = PointCloudsInFilesPreSampled(test_dataset_path,
                                                    '*.las', dataset=use_datasets, use_column="intensity_normalized")

        # Augment pre-sampled training data
        if hp['num_augs'] > 0:
            for i in range(hp['num_augs']):
                aug_trainset = AugmentPreSampledPoints(
                    train_dataset_path,
                    "*.las",
                    use_columns="intensity_normalized",
                    dataset=use_datasets
                )

                # Concat training and augmented training datasets
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
            print(
                f"Adding {hp['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

    else:
        train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=hp['num_points'],
                                           use_columns=use_columns,
                                           filter_height=hp['ground_filter_height'], dataset=use_datasets)
        val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=hp['num_points'],
                                         use_columns=use_columns,
                                         filter_height=hp['ground_filter_height'], dataset=use_datasets)

        # Augment training data
        if hp['num_augs'] > 0:
            for i in range(hp['num_augs']):
                aug_trainset = AugmentPointCloudsInFiles(
                    train_dataset_path,
                    "*.las",
                    max_points=hp['num_points'],
                    use_columns=use_columns,
                    filter_height=hp['ground_filter_height'],
                    dataset=use_datasets
                )

                # Concat training and augmented training datasets
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
            print(
                f"Adding {hp['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

    # Set up pytorch training and validation loaders
    train_loader = DataListLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataListLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False)

    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    # Define training function
    def train(epoch):
        model.train()
        loss_list = []
        for idx, data_list in enumerate(tqdm(train_loader, colour="red", position=1, leave=False, desc=f"Epoch {epoch} training")):
            optimizer.zero_grad()

            # Predict values and ensure that pred. and obs. tensors have same shape
            outs = model(data_list)
            y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list), 4))

            # Compute mse loss for each component
            loss_bark = F.mse_loss(y[:, 0], outs[:, 0])
            loss_branch = F.mse_loss(y[:, 1], outs[:, 1])
            loss_foliage = F.mse_loss(y[:, 2], outs[:, 2])
            loss_wood = F.mse_loss(y[:, 3], outs[:, 3])

            # Set the % contribution of each component to total tree biomass
            a = 1 / 11  # Bark comprises ~11% of tree biomass across entire dataset
            b = 1 / 12  # Branches comprise ~12% of tree biomass across entire dataset
            c = 1 / 5  # Foliage comprises ~5% of tree biomass across entire dataset
            d = 1 / 72  # Wood comprises ~72% of tree biomass across entire dataset

            # Calculate mse loss using loss for each component relative to its contribution to total biomass
            loss = loss_bark * a + loss_branch * b + loss_foliage * c + loss_wood * d

            loss.backward()
            optimizer.step()
            if (idx + 1) % 1 == 0:
                #tqdm.write(str(f'[{idx + 1}/{len(train_loader)}] Loss: {loss.to("cpu"):.4f} '))
                loss_list.append(loss.detach().to("cpu").numpy())
        return np.mean(loss_list)


    # Define validation function
    def val(loader, ep_id):
        with torch.no_grad():
            model.eval()
            losses = []
            for idx, data_list in enumerate(loader):
                outs = model(data_list)
                y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list), 4))

                # Compute mse loss for each component
                loss_bark = F.mse_loss(y[:, 0], outs[:, 0])
                loss_branch = F.mse_loss(y[:, 1], outs[:, 1])
                loss_foliage = F.mse_loss(y[:, 2], outs[:, 2])
                loss_wood = F.mse_loss(y[:, 3], outs[:, 3])

                # Set the % contribution of each component to total tree biomass
                a = 1 / 11  # Bark comprises ~11% of tree biomass across entire dataset
                b = 1 / 12  # Branches comprise ~12% of tree biomass across entire dataset
                c = 1 / 5  # Foliage comprises ~5% of tree biomass across entire dataset
                d = 1 / 72  # Wood comprises ~72% of tree biomass across entire dataset

                # Calculate mse loss using loss for each component relative to its contribution to total biomass
                loss = loss_bark * a + loss_branch * b + loss_foliage * c + loss_wood * d

                losses.append(float(loss.to("cpu")))
            return float(np.mean(losses))


    def main():
        # Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        # Training loop
        for epoch in tqdm(range(0, num_epochs), colour="green", position=0, leave=True):
            train_mse = train(epoch)
            torch.cuda.empty_cache()
            val_mse = val(val_loader, epoch)

            # Save epoch train/val MSE
            with open(model_path.replace('.model', '.csv'), 'a') as f:
                f.write(
                    f'{epoch}, {train_mse}, {val_mse}\n'
                )

            # Early stopping
            if early_stopping is True:
                if val_mse > last_val_mse:
                    trigger_times += 1
                    tqdm.write("    Early stopping trigger " + str(trigger_times) + " out of " + str(hp['patience']))
                    if trigger_times >= hp['patience']:
                        print(f'\nEarly stopping at epoch {epoch}!\n')
                        return
                else:
                    trigger_times = 0
                    last_val_mse = val_mse

            # Report epoch stats
            tqdm.write("    Epoch: " + str(epoch) + "  | Mean val MSE: " + str(
                round(val_mse, 2)) + "  | Mean train MSE: " + str(round(train_mse, 2)))

            # Determine whether to save the model based on val MSE
            val_mse_list.append(val_mse)
            if val_mse <= min(val_mse_list):
                tqdm.write("    Saving model for epoch " + str(epoch))
                torch.save(model, model_path)

        print(f"\nFinished all {num_epochs} training epochs.")

        return


    # Run training loop
    main()

    # Plot the change in training and validation MSE --------------------------------------------------

    # Load most recent set of training results
    folder_path = r'D:\Sync\DL_Development\Models'
    file_type = r'\*.csv'
    files = glob.glob(folder_path + file_type)
    training_results = max(files, key=os.path.getctime)
    training_results = pd.read_csv(training_results, sep=",", header=None)
    training_results.columns = ['epoch', 'train_mse', 'val_mse']

    # Plot the change in training and validation mse over time
    fig, ax = plt.subplots()
    ax.plot(training_results["epoch"], training_results["train_mse"], color="blue", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.plot(training_results["epoch"], training_results["val_mse"], color="red", marker="o")
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Training')
    plt.legend(handles=[red_patch, blue_patch])

    # Apply the model to test data ---------------------------------------------------------------------------------

    #Apply model to training dataset
    test_model(model_file=None,
               test_dataset_path=test_dataset_path,
               use_presampled=True,
               point_cloud_vis=False,
               use_columns=['intensity_normalized'],
               use_datasets=["BC", "RM", "PF"],  # Possible datasets: BC, RM, PF
               num_points=7168)
