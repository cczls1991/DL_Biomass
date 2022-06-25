import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pn2_regressor_V3 import Net
from pointcloud_dataset_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import glob
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    #SETUP HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized', 'classification', 'return_num']
    num_points = 2_000
    early_stopping = True
    patience = 5
    num_augs = 4
    batch_size = 16
    learning_rate = 0.001470856225557105
    weight_decay = 1.1792808360795387e-09
    ground_filter_height = 0.2
    activation_function = "ReLU"
    neuron_multiplier = 4
    dropout_probability = 0.55
    num_epochs = 9
    writer = SummaryWriter(comment="updated_hyperparameters")
    train_dataset_path = r'D:\Sync\Romeo_Data\train'
    val_dataset_path = r'D:\Sync\Romeo_Data\val'

    #Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(num_features=len(use_columns),
                activation_function=activation_function,
                neuron_multiplier=neuron_multiplier,
                dropout_probability=dropout_probability
                ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Set device
    print(f"Using {device} device.")

    #Get training and val datasets
    train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns, filter_height=ground_filter_height)

    print(train_dataset)

