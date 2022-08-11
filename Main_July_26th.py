import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from augmentation import AugmentPointCloudsInFiles
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
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

    # SETUP STATIC HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    use_datasets = ["BC", "RM", "PF"]  # Possible datasets: BC, RM, PF
    num_points = 7_000
    early_stopping = True
    num_epochs = 400
    writer = SummaryWriter(comment="updated_hyperparameters")
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'
    val_dataset_path = r'D:\Sync\Data\Model_Input\val'
    test_dataset_path = r'D:\Sync\Data\Model_Input\test'

    # Report additional hyperparameters
    print(f"Dataset(s): {use_datasets}")
    print(f"Additional features used: {use_columns}")
    print(f"Using {num_points} points per plot")
    print(f"Early stopping: {early_stopping}")
    print(f"Max number of epochs: {num_epochs}")

    # Specify hyperparameter tunings
    hp = {'lr': 0.0005753187813135093,
          'weight_decay': 8.0250963438986e-05,
          'batch_size': 28,
          'num_augs': 7,
          'patience': 5,
          'ground_filter_height': 0.2,
          'activation_function': "ReLU",
          'optimizer': "Adam",
          'neuron_multiplier': 0,
          'dropout_probability': 0.55
          }

    print("\nHyperparameters:\n")
    pp.pprint(hp, width=1)

    # Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Set model
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

    # Note device, dataset
    print(f"Using {device} device.")

    # Get training val, and test datasets
    train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                       filter_height=hp['ground_filter_height'], dataset=use_datasets)
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                     filter_height=hp['ground_filter_height'], dataset=use_datasets)
    test_dataset = PointCloudsInFiles(test_dataset_path, '*.las', max_points=num_points,
                                      use_columns=use_columns, filter_height=0.2, dataset=use_datasets)

    # Augment training data
    if hp['num_augs'] > 0:
        for i in range(hp['num_augs']):
            aug_trainset = AugmentPointCloudsInFiles(
                train_dataset_path,
                "*.las",
                max_points=num_points,
                use_columns=use_columns,
                filter_height=hp['ground_filter_height'],
                dataset=use_datasets
            )

            # Concat training and augmented training datasets
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
        print(
            f"Adding {hp['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

    # Set up pytorch training and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=0)


    # Define training function
    def train():
        model.train()
        loss_list = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            #Predict values and ensure that pred. and obs. tensors have same shape
            outs = torch.reshape(model(data), (len(data.y), 1))
            data.y = torch.reshape(data.y, (len(data.y), 1))
            loss = F.mse_loss(outs, data.y)
            loss.backward()
            optimizer.step()
            if (idx + 1) % 1 == 0:
                tqdm.write(str(f'[{idx + 1}/{len(train_loader)}] Loss: {loss.to("cpu"):.4f} '))
                loss_list.append(loss.detach().to("cpu").numpy())
        return np.mean(loss_list)


    def val(loader, ep_id):  # Note sure what ep_id does
        with torch.no_grad():
            model.eval()
            losses = []
            for idx, data in enumerate(loader):
                data = data.to(device)
                outs = torch.reshape(model(data), (len(data.y), 1))
                data.y = torch.reshape(data.y, (len(data.y), 1))
                loss = F.mse_loss(outs, data.y)
                losses.append(float(loss.to("cpu")))
                print(ep_id)
            return float(np.mean(losses))


    def main():
        # Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        # Training loop
        for epoch in tqdm(range(0, num_epochs), colour="green", position=0, leave=True):
            train_mse = train()
            val_mse = val(val_loader, epoch)

            # Record epoch results
            if num_epochs > 10:
                writer.add_scalar("Training MSE", train_mse, epoch)  # Save to tensorboard summary
                writer.add_scalar("Validation MSE", val_mse, epoch)  # Save to tensorboard summary
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
            tqdm.write("    Epoch: " + str(epoch) + "  | Mean val MSE: " + str(round(val_mse, 2)) + "  | Mean train MSE: " + str(round(train_mse, 2)))

            # Determine whether to save the model based on val MSE
            val_mse_list.append(val_mse)
            if val_mse <= min(val_mse_list):
                tqdm.write("    Saving model for epoch " + str(epoch))
                torch.save(model, model_path)

        # Terminate tensorboard writer
        writer.flush()
        writer.close()

        print(f"\nFinished all {num_epochs} training epochs.")

        return


    # Run training loop
    main()

    # Plot the change in training and validation MSE --------------------------------------------------

    # Load most recent model and set of training results
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
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)

    # Apply the model
    model.eval()
    for idx, data in enumerate(test_loader):
        data = data.to(device)
        pred = torch.reshape(model(data), (len(data.y), 1)).to('cpu').detach().numpy()
        obs = torch.reshape(data.y, (len(data.y), 1)).to('cpu').detach().numpy()
        PlotID = data.PlotID

    # Calculate overall R^2 and RMSE across all component estimates
    overall_r2 = round(metrics.r2_score(obs, pred), 3)
    overall_rmse = round(sqrt(metrics.mean_squared_error(obs, pred)), 2)
    print(f"Overall R2: {overall_r2}\nOverall RMSE: {overall_rmse}")

    # Reshape for bark, branch, foliage, wood columns
    obs_arr = np.reshape(a=obs, newshape=(len(obs) // 4, 4))
    pred_arr = np.reshape(a=pred, newshape=(len(obs) // 4, 4))
    # Join arrays
    arr = arr = np.concatenate((obs_arr, pred_arr), axis=1)
    # Convert to data frame
    df = pd.DataFrame(arr,
                      columns=['bark_obs', 'branch_obs', 'foliage_obs', 'wood_obs',
                               'bark_pred', 'branch_pred', 'foliage_pred', 'wood_pred'])
    # Add plot IDs to df
    df["PlotID"] = PlotID

    # Calculate R^2 and RMSE for bark, branch, foliage, wood

    # bark
    bark_r2 = round(metrics.r2_score(df["bark_obs"], df["bark_pred"]), 3)
    bark_rmse = round(sqrt(metrics.mean_squared_error(df["bark_obs"], df["bark_pred"])), 2)
    print(f"bark R2: {bark_r2}\nbark RMSE: {bark_rmse}")

    # branch
    branch_r2 = round(metrics.r2_score(df["branch_obs"], df["branch_pred"]), 3)
    branch_rmse = round(sqrt(metrics.mean_squared_error(df["branch_obs"], df["branch_pred"])), 2)
    print(f"branch R2: {branch_r2}\nbranch RMSE: {branch_rmse}")

    # foliage
    foliage_r2 = round(metrics.r2_score(df["foliage_obs"], df["foliage_pred"]), 3)
    foliage_rmse = round(sqrt(metrics.mean_squared_error(df["foliage_obs"], df["foliage_pred"])), 2)
    print(f"foliage R2: {foliage_r2}\nfoliage RMSE: {foliage_rmse}")

    # wood
    wood_r2 = round(metrics.r2_score(df["wood_obs"], df["wood_pred"]), 3)
    wood_rmse = round(sqrt(metrics.mean_squared_error(df["wood_obs"], df["wood_pred"])), 2)
    print(f"wood R2: {wood_r2}\nwood RMSE: {wood_rmse}")

    # Calculate R^2 and RMSE for bark, branch, foliage, wood

    # bark
    bark_r2 = round(metrics.r2_score(df["bark_obs"], df["bark_pred"]), 3)
    bark_rmse = round(sqrt(metrics.mean_squared_error(df["bark_obs"], df["bark_pred"])), 2)
    print(f"bark R2: {bark_r2}\nbark RMSE: {bark_rmse}")

    # branch
    branch_r2 = round(metrics.r2_score(df["branch_obs"], df["branch_pred"]), 3)
    branch_rmse = round(sqrt(metrics.mean_squared_error(df["branch_obs"], df["branch_pred"])), 2)
    print(f"branch R2: {branch_r2}\nbranch RMSE: {branch_rmse}")

    # foliage
    foliage_r2 = round(metrics.r2_score(df["foliage_obs"], df["foliage_pred"]), 3)
    foliage_rmse = round(sqrt(metrics.mean_squared_error(df["foliage_obs"], df["foliage_pred"])), 2)
    print(f"foliage R2: {foliage_r2}\nfoliage RMSE: {foliage_rmse}")

    # wood
    wood_r2 = round(metrics.r2_score(df["wood_obs"], df["wood_pred"]), 3)
    wood_rmse = round(sqrt(metrics.mean_squared_error(df["wood_obs"], df["wood_pred"])), 2)
    print(f"wood R2: {wood_r2}\nwood RMSE: {wood_rmse}")



