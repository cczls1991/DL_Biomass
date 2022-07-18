import os.path
import glob
import torch
from torch_geometric.loader import DataLoader
from pointcloud_dataloader import PointCloudsInFiles
from pointnet2_regressor import Net
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from math import sqrt

# Apply the model to the test dataset and plot the results --------------------------------------------------

# Select a model to use:
model_file = None

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Specify model params
use_columns = ['intensity_normalized']
num_points = 2_000

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
test_dataset = PointCloudsInFiles(r"D:\Sync\Data\Model_Input\test", '*.las', max_points=num_points, use_columns=use_columns,
                                  filter_height=0.2)

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)

# Apply the model
model.eval()
for idx, data in enumerate(test_loader):
    data = data.to(device)
pred = model(data)[:, 0].to('cpu').detach().numpy()
obs = data.y.to('cpu').detach().numpy()

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