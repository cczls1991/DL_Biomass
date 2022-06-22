import os.path
import glob
import torch
from torch_geometric.loader import DataLoader
from pointcloud_dataset_biomass_adapted_V2 import PointCloudsInFiles
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    # SETUP PARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    num_points = 10_000
    augment = True
    num_augs = 2
    batch_size = 4
    num_epochs = 2
    writer = SummaryWriter(comment="_10000_points_lr_0.0001_batch_size_1")
    train_dataset_path = r'/Romeo_Data/train'
    val_dataset_path = r'/Romeo_Data/val'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load most recent model
    folder_path = r'/DL_Development/Models'
    file_type = r'\*.model'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    model = torch.load(os.path.join(folder_path, max_file))
    print("Using model:", max_file)

    # Get test data
    test_dataset = PointCloudsInFiles(r"/Romeo_Data/test", '*.las', max_points=num_points, use_columns=use_columns,
                                      filter_ground=True, filter_height=1.3)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    # Get test data
    test_dataset = PointCloudsInFiles(r"/Romeo_Data/test", '*.las', max_points=num_points, use_columns=use_columns,
                                      filter_ground=True, filter_height=1.3)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    model.eval()
    for idx, data in enumerate(test_loader):
         data = data.to(device)
    pred = model(data)[:, 0].to('cpu').detach().numpy()
    obs = data.y.to('cpu').detach().numpy()


    #Get residuals
    resid = obs - pred

    #Plot predicted vs. observed values
    import matplotlib.pyplot as plt

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