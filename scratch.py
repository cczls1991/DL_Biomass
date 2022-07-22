import os.path
import glob
import torch
from torch_geometric.loader import DataLoader
from pointcloud_dataloader import PointCloudsInFiles
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from math import sqrt
import numpy as np
from pointcloud_dataloader import read_las
from itertools import compress
from pathlib import Path
import pandas as pd
from random import sample

# Apply the model to the test dataset and plot the results --------------------------------------------------

# Select a model to use:
model_file = None

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Specify model params
use_columns = ['intensity_normalized']
use_datasets = ["PF"]  # Possible datasets: BC, RM, PF
num_points = 7_000
test_dataset_path = r'D:\Sync\Data\Model_Input\test'

# Load most recent model
if model_file is None:
    folder_path = r'D:\Sync\DL_Development\Models'
    file_type = r'\*.model'
    files = glob.glob(folder_path + file_type)
    model_file = max(files, key=os.path.getctime)
    model = torch.load(model_file)
else:
    model = torch.load(model_file)
print("Using model:", model_file)

# Get test data
test_dataset = PointCloudsInFiles(test_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                  filter_height=0.2, dataset=use_datasets)

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)

# Apply the model
model.eval()
for idx, data in enumerate(test_loader):
    data = data.to(device)
pred = model(data)[:, 0].to('cpu').detach().numpy()
obs = data.y.to('cpu').detach().numpy()
PlotID = data.PlotID

# Calculate R^2 and RMSE for test dataset
test_r2 = round(metrics.r2_score(obs, pred), 3)
test_rmse = round(sqrt(metrics.mean_squared_error(obs, pred)), 2)
print(f"R2: {test_r2}\nRMSE: {test_rmse}")

# Get residuals
resid = obs - pred

# Plot observed vs. predicted
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(obs, pred)
ax1.set(xlabel='Observed Biomass (Tons)', ylabel='Predicted Biomass (Tons)')
plt.figtext(0.1, 0.65, f"R2: {test_r2}\nRMSE: {test_rmse}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax1.transAxes)

# Plot residuals
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

# Visualize point clouds of four random plots with estimated and observed biomass provided  ---------------------------------------------------------------------------------

#Create df of plot ID, observed, and predicted biomass
df = pd.DataFrame(list(zip(PlotID, obs, pred)),
                  columns=['PlotID', 'obs_biomass', 'pred_biomass'])

#Randomly sample 4 plots to visualize
df = df.sample(n=4)
coords_list = []
#Grab the LAS files of the four plots
for i in range(0, len(df["PlotID"]), 1):
    #Get filepath
    las_path = os.path.join(test_dataset_path, str(df['PlotID'].tolist()[i] + ".las"))
    # Load coords for each LAS
    coords_i = read_las(las_path, get_attributes=False)
    #Add to list
    coords_list.append(coords_i)

#Plot the point clouds
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=[30, 30])

# set up the axes for the first plot
ax = fig.add_subplot(221, projection='3d')
ax.scatter(coords_list[0][:, 0], coords_list[0][:, 1], coords_list[0][:, 2], c=coords_list[0][:, 2], cmap='viridis', linewidth=0.5)

# set up the axes for the second plot
ax = fig.add_subplot(222, projection='3d')
ax.scatter(coords_list[1][:, 0], coords_list[1][:, 1], coords_list[1][:, 2], c=coords_list[1][:, 2], cmap='viridis', linewidth=0.5)

# set up the axes for the third plot
ax = fig.add_subplot(223, projection='3d')
ax.scatter(coords_list[2][:, 0], coords_list[2][:, 1], coords_list[2][:, 2], c=coords_list[2][:, 2], cmap='viridis', linewidth=0.5)

# set up the axes for the fourth plot
ax = fig.add_subplot(224, projection='3d')
ax.scatter(coords_list[3][:, 0], coords_list[3][:, 1], coords_list[3][:, 2], c=coords_list[3][:, 2], cmap='viridis', linewidth=0.5)

plt.show()


