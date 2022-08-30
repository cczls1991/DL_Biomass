import os.path
import glob
import torch
from torch_geometric.loader import DataListLoader
from pointcloud_dataloader import PointCloudsInFiles
from pointcloud_dataloader import PointCloudsInFilesPreSampled
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from math import sqrt
from pointcloud_dataloader import read_las
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

def test_model(model_file=None,
                test_dataset_path=r'D:\Sync\Data\Model_Input\test',
                use_presampled=True,
                point_cloud_vis=False,
                use_columns=None,
                use_datasets=None,  # Possible datasets: BC, RM, PF
                num_points=200 #Num points is only used if use_presampled=False
               ):

    if use_datasets is None:
        use_datasets = ["BC", "RM", "PF"]
    if use_columns is None:
        use_columns = ['intensity_normalized']

    # Load most recent model if one is not specified
    if model_file is None:
        folder_path = r'D:\Sync\DL_Development\Models'
        file_type = r'\*.model'
        files = glob.glob(folder_path + file_type)
        model_file = max(files, key=os.path.getctime)
        model = torch.load(model_file)
    else:
        model = torch.load(model_file)
    print("Using model:", model_file)

    #Set out directory to save plots
    fig_out_dir = r"D:\Sync\Figures\Testing_Model_Output_Plots"

    #Set data parallelization
    print(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get test data
    if use_presampled is True:
        test_dataset = PointCloudsInFilesPreSampled(test_dataset_path,
                                                    '*.las', dataset=use_datasets, use_column="intensity_normalized")
    else:
        test_dataset = PointCloudsInFiles(test_dataset_path, '*.las', max_points=num_points,
                                          use_columns=use_columns, filter_height=0.2, dataset=use_datasets)

    test_loader = DataListLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)


    # Apply the model to test data --------------------------------------------------------------------------
    model.eval()
    PlotID = []
    for idx, data_list in enumerate(test_loader):
        # Predict values and ensure that pred. and obs. tensors have same shape
        pred = model(data_list).to('cpu').detach().numpy()
        obs = torch.reshape(torch.cat([data.y for data in data_list]), (len(data_list), 4)).to('cpu').detach().numpy()
        for i in range(0, len(data_list)):
            PlotID.append(data_list[i].PlotID)

    # Join arrays
    arr = np.concatenate((obs, pred), axis=1)
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
    plt.savefig(os.path.join(fig_out_dir, 'tree_btphr_obs_vs_pred.png'))

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

    plt.show()

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, 'tree_btphr_residuals.png'))

    # Plot biomass component obs. vs. predicted   -----------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Add the main title
    fig.suptitle("Component Biomass Observed vs Predicted", fontsize=15)

    # Bark
    ax[0, 0].scatter(df["bark_btphr_obs"], df["bark_btphr_pred"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    # Branch
    ax[1, 0].scatter(df["branch_btphr_obs"], df["branch_btphr_pred"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    # Foliage
    ax[0, 1].scatter(df["foliage_btphr_obs"], df["foliage_btphr_pred"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    # wood
    ax[1, 1].scatter(df["wood_btphr_obs"], df["wood_btphr_pred"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

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

    #Add legend
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right')

    plt.show()

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, 'component_obs_vs_pred.png'))

    # Make plot for component biomass residuals ------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Add the main title
    fig.suptitle("Component Biomass Residuals", fontsize=15)

    # bark_btphr
    ax[0, 0].scatter(df["bark_btphr_obs"], df["bark_btphr_resid"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    ax[1, 0].scatter(df["branch_btphr_obs"], df["branch_btphr_resid"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    ax[0, 1].scatter(df["foliage_btphr_obs"], df["foliage_btphr_resid"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

    ax[1, 1].scatter(df["wood_btphr_obs"], df["wood_btphr_resid"], alpha=0.8, c=df["colour"], edgecolors='none', s=30,
               label=df["dataset"])

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

    #Add legend
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='lower right')

    plt.show()

    # Save plot
    plt.savefig(os.path.join(fig_out_dir, 'component_residuals.png'))

    if point_cloud_vis is True:

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

            #Downsample points
            if coords_i.shape[0] >= num_points:
                use_idx = np.random.choice(coords_i.shape[0], num_points, replace=False)
            else:
                use_idx = np.random.choice(coords_i.shape[0], num_points, replace=True)

            #Subset to filtered points
            coords_i = coords_i[use_idx, :]
            coords_i = coords_i - np.mean(coords_i, axis=0)  # centralize coordinates

            # Add to list
            coords_list.append(coords_i)

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

if __name__ == '__main__':


    #Apply model to training dataset
    test_model(model_file=None,
               test_dataset_path=r'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_test',
               use_presampled=True,
               point_cloud_vis=False,
               use_columns=['intensity_normalized'],
               use_datasets=["BC", "RM", "PF"],  # Possible datasets: BC, RM, PF
               num_points=7168)





