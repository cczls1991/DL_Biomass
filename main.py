
#Following these resources:
#https://github.com/pyg-team/pytorch_geometric/issues/1417
#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/data_parallel.py

from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import numpy as np
import torch
import torch.nn.functional as F
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from augmentation import AugmentPointCloudsInFiles
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
from pointcloud_dataloader import read_las

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

#Add comment for this run
comment = "multi_gpu"

if __name__ == '__main__':

    # SETUP STATIC HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    use_datasets = ["BC"]  # Possible datasets: BC, RM, PF
    early_stopping = True
    num_epochs = 400
    point_cloud_vis = False
    fig_out_dir = r"D:\Sync\Figures\Testing_Model_Output_Plots" # Set out directory to save plots
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'
    val_dataset_path = r'D:\Sync\Data\Model_Input\val'
    test_dataset_path = r'D:\Sync\Data\Model_Input\test'

    # SET UP TUNING PARAMETERS
    cfg = {'lr': 0.026726850890033942,
           'batch_size': 60,
           'weight_decay': 3.2540888065641056e-10,
           'num_augs': 8,
           'num_points': 7_000,
           'neuron_multiplier': 0,
           'patience': 20,
           'ground_filter_height': 0,
           'activation_function': 'ReLU',
           'optimizer': "Adam",
           'lr_scheduler': False,
           'dropout_probability': 0.5
           }

    # Report additional hyperparameters
    print(f"Dataset(s): {use_datasets}")
    print(f"Additional features used: {use_columns}")
    print(f"Using {cfg['num_points']} points per plot")
    print(f"Early stopping: {early_stopping}")
    print(f"Max number of epochs: {num_epochs}")

    print("\nHyperparameters:\n")
    pp.pprint(cfg, width=1)

    # Set model
    model = Net(num_features=len(use_columns),
                activation_function=cfg['activation_function'],
                neuron_multiplier=cfg['neuron_multiplier'],
                dropout_probability=cfg['dropout_probability']
                )

    # Get training val, and test datasets
    train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=cfg['num_points'], use_columns=use_columns,
                                       filter_height=cfg['ground_filter_height'], dataset=use_datasets)
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=cfg['num_points'], use_columns=use_columns,
                                     filter_height=cfg['ground_filter_height'], dataset=use_datasets)
    test_dataset = PointCloudsInFiles(test_dataset_path, '*.las', max_points=cfg['num_points'], use_columns=use_columns,
                                      filter_height=cfg['ground_filter_height'], dataset=use_datasets)

    # Augment training data
    if cfg['num_augs'] > 0:
        for i in range(cfg['num_augs']):
            aug_trainset = AugmentPointCloudsInFiles(
                train_dataset_path,
                "*.las",
                max_points=cfg['num_points'],
                use_columns=use_columns,
                filter_height=cfg['ground_filter_height'],
                dataset=use_datasets
            )

            # Concat training and augmented training datasets
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
        print(
            f"Adding {cfg['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")


    # Set up dataset loaders
    train_loader = DataListLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataListLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataListLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set optimizer
    if cfg['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    if cfg['lr_scheduler'] is True:
        #Set learning rate scheduler (Source: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=10,  # Number of epochs with no improvement after which learning rate will be reduced.
                    factor=0.1,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                    cooldown=0,  # Number of epochs to wait before resuming normal operation after lr has been reduced
                    min_lr=0  # A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively
        )

    # Define training function
    def train():
        model.train()
        loss_list = []
        for idx, data_list in enumerate(train_loader):
            optimizer.zero_grad()
            # Predict values and ensure that pred. and obs. tensors have same shape
            outs = torch.reshape(model(data_list), (len(data_list)*4, 1))
            y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list)*4, 1))
            loss = F.mse_loss(outs, y)
            loss.backward()
            optimizer.step()
            if (idx + 1) % 1 == 0:
                tqdm.write(str(f'[{idx + 1}/{len(train_loader)}] Loss: {loss.to("cpu"):.4f} '))
                loss_list.append(loss.detach().to("cpu").numpy())
        return np.mean(loss_list)

    # Define validation function
    def val(loader, ep_id):
        with torch.no_grad():
            model.eval()
            losses = []
            for idx, data_list in enumerate(loader):
                outs = torch.reshape(model(data_list), (len(data_list)*4, 1))
                y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list)*4, 1))
                loss = F.mse_loss(outs, y)
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
            if cfg['lr_scheduler'] is True:
                lr_scheduler.step(val_mse)

            # Record epoch results
            with open(model_path.replace('.model', '.csv'), 'a') as f:
                f.write(
                    f'{epoch}, {train_mse}, {val_mse}\n'
                )

            # Early stopping
            if early_stopping is True:
                if val_mse > last_val_mse:
                    trigger_times += 1
                    tqdm.write("    Early stopping trigger " + str(trigger_times) + " out of " + str(cfg['patience']))
                    if trigger_times >= cfg['patience']:
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

    # Load most recent model and set of training results
    folder_path = r'D:\Sync\DL_Development\Models'
    file_type = r'\*.csv'
    files = glob.glob(folder_path + file_type)
    training_results = max(files, key=os.path.getctime)
    training_results = pd.read_csv(training_results, sep=",", header=None)
    training_results.columns = ['epoch', 'train_mse', 'val_mse']

    #Get current date and time for saving figures
    t_now = dt.now().strftime("%Y_%m_%d_%H_%M")

    # Plot the change in training and validation mse over time (learning curve)
    fig, ax = plt.subplots()
    ax.plot(training_results["epoch"], training_results["train_mse"], color="blue", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.plot(training_results["epoch"], training_results["val_mse"], color="red", marker="o")
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Training')
    plt.legend(handles=[red_patch, blue_patch])

    plt.savefig(fr'D:\Sync\Figures\Learning_Curves\Learning_Curve_{t_now}_{comment}.png')

    # Apply the model to test data --------------------------------------------------------------------------

    #IMPORTANT NOTE: BTPHR = Biomass Tonnes Per Hectare Reduced (Reduced means that vals were divided by 10 to achieve faster convergence, feature scaling is common in DL)

    model.eval()
    PlotID = []
    for idx, data_list in enumerate(test_loader):
        obs = torch.reshape(model(data_list), (len(data_list) * 4, 1)).to('cpu').detach().numpy()
        pred = torch.reshape(torch.cat([data.y for data in data_list]), (len(data_list) * 4, 1)).to(
            'cpu').detach().numpy()
        for i in range(0, len(data_list)):
            PlotID.append(data_list[i].PlotID)

    # Reshape for bark, branch, foliage, wood columns
    obs_arr = np.reshape(a=obs, newshape=(len(obs) // 4, 4))
    pred_arr = np.reshape(a=pred, newshape=(len(obs) // 4, 4))
    # Join arrays
    arr = arr = np.concatenate((obs_arr, pred_arr), axis=1)
    # Convert to data frame
    df = pd.DataFrame(arr,
                      columns=['bark_btphr_obs', 'branch_btphr_obs', 'foliage_btphr_obs', 'wood_btphr_obs',
                               'bark_btphr_pred', 'branch_btphr_pred', 'foliage_btphr_pred', 'wood_btphr_pred'],
                      index=PlotID)

    # Add observed/predicted total biomass columns to df
    df["tree_btphr_obs"] = df["bark_btphr_obs"] + df["branch_btphr_obs"] + df["foliage_btphr_obs"] + df["wood_btphr_obs"]
    df["tree_btphr_pred"] = df["bark_btphr_pred"] + df["branch_btphr_pred"] + df["foliage_btphr_pred"] + df["wood_btphr_pred"]

    # Get residuals
    df["tree_btphr_resid"] = df["tree_btphr_obs"] - df["tree_btphr_pred"]
    df["bark_btphr_resid"] = df["bark_btphr_obs"] - df["bark_btphr_pred"]
    df["branch_btphr_resid"] = df["branch_btphr_obs"] - df["branch_btphr_pred"]
    df["foliage_btphr_resid"] = df["foliage_btphr_obs"] - df["foliage_btphr_pred"]
    df["wood_btphr_resid"] = df["wood_btphr_obs"] - df["wood_btphr_pred"]

    # Calculate test metrics for each component -------------------------------------------------------------

    # Create a data frame to store component metrics
    metrics_df = pd.DataFrame(columns=["r2", "rmse", "mape"], index=["wood_btphr", "bark_btphr", "branch_btphr", "foliage_btphr", "tree_btphr"])

    # Loop through each biomass component get model performance metrics
    for comp in metrics_df.index.tolist():
        metrics_df.loc[comp, "r2"] = round(metrics.r2_score(df[f"{comp}_obs"], df[f"{comp}_pred"]), 2)
        metrics_df.loc[comp, "rmse"] = round(sqrt(metrics.mean_squared_error(df[f"{comp}_obs"], df[f"{comp}_pred"])), 2)
        metrics_df.loc[comp, "mape"] = round(
            metrics.mean_absolute_percentage_error(df[f"{comp}_obs"], df[f"{comp}_pred"]), 3)

    print(metrics_df)

    # Plot total AGB biomass obs. vs. predicted  -----------------------------------------------

    # Add dataset col
    df["dataset"] = "blank"

    # Add a column to df for dataset
    for id in df.index.tolist():
        df.loc[id, "dataset"] = id[0:2]

    # Add color for each dataset
    df["colour"] = "green"
    df.loc[df["dataset"] == "BC", "colour"] = "red"
    df.loc[df["dataset"] == "PF", "colour"] = "blue"

    # Create plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df["tree_btphr_obs"], df["tree_btphr_pred"],
               alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    plt.figtext(0.05, 0.9,
                f"R2: {metrics_df.loc['tree_btphr', 'r2']}\nRMSE: {metrics_df.loc['tree_btphr', 'rmse']}\nMAPE: {str(round(metrics_df.loc['tree_btphr', 'mape'], 2))}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes)

    # Add legend
    red_patch = mpatches.Patch(color="red", label='BC Gov')
    blue_patch = mpatches.Patch(color="blue", label='Petawawa')
    green_patch = mpatches.Patch(color="green", label='Romeo-Malette')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right')

    # Add title
    plt.title("Total Tree AGB Observed vs Predicted", fontdict=None, loc='center', fontsize=15)

    # Set axis so its scaled properly
    plt.axis('scaled')

    plt.show()

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, f'tree_btphr_obs_vs_pred_{t_now}_{comment}.png'))

    # Make plot for total AGB residuals ------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df["tree_btphr_obs"], df["tree_btphr_resid"],
               alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])
    # Add legend
    red_patch = mpatches.Patch(color="red", label='BC Gov')
    blue_patch = mpatches.Patch(color="blue", label='Petawawa')
    green_patch = mpatches.Patch(color="green", label='Romeo-Malette')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right')

    # Add title
    plt.title("Total Tree AGB Residuals", fontdict=None, loc='center', fontsize=15)

    # Set axis so its scaled properly
    plt.axis('scaled')

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, f'tree_btphr_residuals{t_now}_{comment}.png'))

    # Plot biomass component obs. vs. predicted   -----------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Add the main title
    fig.suptitle("Component Biomass Observed vs Predicted", fontsize=15)

    # Bark
    ax[0, 0].scatter(df["bark_btphr_obs"], df["bark_btphr_pred"], color="red")

    # Branch
    ax[1, 0].scatter(df["branch_btphr_obs"], df["branch_btphr_pred"], color="blue")

    # Foliage
    ax[0, 1].scatter(df["foliage_btphr_obs"], df["foliage_btphr_pred"], color="green")

    # wood
    ax[1, 1].scatter(df["wood_btphr_obs"], df["wood_btphr_pred"], color="orange")

    # Add titles
    ax[0, 0].title.set_text('Bark')
    ax[1, 0].title.set_text('Branch')
    ax[0, 1].title.set_text('Foliage')
    ax[1, 1].title.set_text('Wood')

    # Add summary stats text
    ax[0, 0].text(0.1, 0.9,
                  f"R2: {metrics_df.loc['bark_btphr', 'r2']}\nRMSE: {metrics_df.loc['bark_btphr', 'rmse']}\nMAPE: {str(round(metrics_df.loc['bark_btphr', 'mape'], 2))}",
                  horizontalalignment='left', verticalalignment='top', transform=ax[0, 0].transAxes)
    ax[1, 0].text(0.1, 0.9,
                  f"R2: {metrics_df.loc['branch_btphr', 'r2']}\nRMSE: {metrics_df.loc['branch_btphr', 'rmse']}\nMAPE: {str(round(metrics_df.loc['branch_btphr', 'mape'], 2))}",
                  horizontalalignment='left', verticalalignment='top', transform=ax[1, 0].transAxes)
    ax[0, 1].text(0.1, 0.9,
                  f"R2: {metrics_df.loc['foliage_btphr', 'r2']}\nRMSE: {metrics_df.loc['foliage_btphr', 'rmse']}\nMAPE: {str(round(metrics_df.loc['foliage_btphr', 'mape'], 2))}",
                  horizontalalignment='left', verticalalignment='top', transform=ax[0, 1].transAxes)
    ax[1, 1].text(0.1, 0.9,
                  f"R2: {metrics_df.loc['wood_btphr', 'r2']}\nRMSE: {metrics_df.loc['wood_btphr', 'rmse']}\nMAPE: {str(round(metrics_df.loc['wood_btphr', 'mape'], 2))}",
                  horizontalalignment='left', verticalalignment='top', transform=ax[1, 1].transAxes)

    # Add axis labels
    for axis in ax.flat:
        axis.set(xlabel='Observed Biomass (tons)', ylabel='Predicted Biomass (tons)')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, f'component_obs_vs_pred_{t_now}_{comment}.png'))

    # Make plot for component biomass residuals ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Add the main title
    fig.suptitle("Component Biomass Residuals", fontsize=15)

    # bark_btphr
    ax[0, 0].scatter(df["bark_btphr_obs"], df["bark_btphr_resid"], color="red")
    ax[1, 0].scatter(df["branch_btphr_obs"], df["branch_btphr_resid"], color="blue")
    ax[0, 1].scatter(df["foliage_btphr_obs"], df["foliage_btphr_resid"], color="green")
    ax[1, 1].scatter(df["wood_btphr_obs"], df["wood_btphr_resid"], color="orange")

    # Add titles
    ax[0, 0].title.set_text('Bark')
    ax[1, 0].title.set_text('Branch')
    ax[0, 1].title.set_text('Foliage')
    ax[1, 1].title.set_text('Wood')

    # Add axis labels
    for axis in ax.flat:
        axis.set(xlabel='Observed Biomass (tons)', ylabel='Residuals (tons)')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, f'component_residuals_{t_now}_{comment}.png'))

    # Visualize point clouds of four random plots with estimated and observed biomass provided  ---------------------------------------------------------------------------------

    # Create df of plot ID, observed, and predicted biomass
    df = pd.DataFrame(list(zip(PlotID, obs, pred)),
                      columns=['PlotID', 'obs_biomass', 'pred_biomass'])

    # Randomly sample 4 plots to visualize
    df = df.sample(n=4)
    coords_list = []
    # Grab the LAS files of the four plots
    for i in range(0, len(df["PlotID"]), 1):
        # Get filepath
        las_path = os.path.join(test_dataset_path, str(df['PlotID'].tolist()[i] + ".las"))
        # Load coords for each LAS
        coords_i = read_las(las_path, get_attributes=False)
        # Add to list
        coords_list.append(coords_i)

    # Plot the point clouds
    if point_cloud_vis is True:
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=[30, 30])

        # set up the axes for the first plot
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(coords_list[0][:, 0], coords_list[0][:, 1], coords_list[0][:, 2], c=coords_list[0][:, 2],
                   cmap='viridis', linewidth=0.5)

        # set up the axes for the second plot
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(coords_list[1][:, 0], coords_list[1][:, 1], coords_list[1][:, 2], c=coords_list[1][:, 2],
                   cmap='viridis', linewidth=0.5)

        # set up the axes for the third plot
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(coords_list[2][:, 0], coords_list[2][:, 1], coords_list[2][:, 2], c=coords_list[2][:, 2],
                   cmap='viridis', linewidth=0.5)

        # set up the axes for the fourth plot
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(coords_list[3][:, 0], coords_list[3][:, 1], coords_list[3][:, 2], c=coords_list[3][:, 2],
                   cmap='viridis', linewidth=0.5)

        plt.show()

