#Script made using this tutorial: https://github.com/davidtvs/pytorch-lr-finder
#Algorithm from this article: https://arxiv.org/abs/1506.01186

#Get modules
from pn2_regressor_biomass_adapted_V2 import Net
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pointcloud_dataset_biomass_adapted_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles
from torch_lr_finder import LRFinder
from datetime import datetime as dt


if __name__ == '__main__':

    #SETUP PARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    num_points = 10_000
    augment = True
    num_augs = 2
    batch_size = 16
    train_dataset_path = r'/Romeo_Data/train'
    val_dataset_path = r'/Romeo_Data/val'

    #Device and model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_features=len(use_columns)).to(device)

    #Get train dataset
    train_dataset = PointCloudsInFiles(train_dataset_path,'*.las', max_points=num_points, use_columns=use_columns)
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns)


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

    # Set up pytorch training loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    #Set up pytorch validation loader

    #Set up LR finder
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()