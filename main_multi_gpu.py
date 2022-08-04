
#Following these resources:
#https://github.com/pyg-team/pytorch_geometric/issues/1417
#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/data_parallel.py

from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import torch
import torch.nn.functional as F
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from augmentation import AugmentPointCloudsInFiles
from datetime import datetime as dt
import pprint as pp
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # SETUP STATIC HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    use_datasets = ["BC", "RM", "PF"]  # Possible datasets: BC, RM, PF
    num_points = 1000
    early_stopping = True
    num_epochs = 400
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'

    # Report additional hyperparameters
    print(f"Dataset(s): {use_datasets}")
    print(f"Additional features used: {use_columns}")
    print(f"Using {num_points} points per plot")
    print(f"Early stopping: {early_stopping}")
    print(f"Max number of epochs: {num_epochs}")

    # Specify hyperparameter tunings
    hp = {'lr': 0.0005753187813135093,
          'weight_decay': 8.0250963438986e-05,
          'batch_size': 64,
          'num_augs': 7,
          'patience': 28,
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

    # Set model
    model = Net(num_features=len(use_columns),
                activation_function=hp['activation_function'],
                neuron_multiplier=hp['neuron_multiplier'],
                dropout_probability=hp['dropout_probability']
                ).to(device)

    # Get training val, and test datasets
    train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                       filter_height=hp['ground_filter_height'], dataset=use_datasets)

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

    loader = DataListLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)

    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set optimizer
    if hp['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    #Training Loop
    for epoch in range(0, num_epochs):
        print(f"Epoch: {epoch}")
        for idx, data_list in enumerate(loader):
            optimizer.zero_grad()
            outs = torch.reshape(model(data_list), (hp["batch_size"]*4, 1))
            y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (hp["batch_size"]*4, 1))
            loss = F.mse_loss(outs, y)
            loss.backward()
            optimizer.step()
            if (idx + 1) % 1 == 0:
                print(str(f'[{idx + 1}/{len(loader)}] Loss: {loss.to("cpu"):.4f} '))
