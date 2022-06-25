import os

import optuna
from optuna.trial import TrialState

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pn2_regressor_biomass_adapted_V2 import Net
from pointcloud_dataset_biomass_adapted_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles



if __name__ == '__main__':

    #Specify cuda as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")

    # SET STATIC PARAMETERS
    use_columns = ['intensity_normalized']
    model = Net(num_features=len(use_columns)).to(device)
    num_points = 10_000
    augment = True
    num_augs = 2
    batch_size = 12
    num_epochs = 100
    train_dataset_path = r'/Romeo_Data/train'
    val_dataset_path = r'/Romeo_Data/val'

    #Define obective function to be used in optuna
    def objective(trial):

        # SET TUNING PARAMETERS
        cfg = {'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
               'weight_decay': trial.suggest_loguniform('weight_decay', 1e-10, 1e-3),
               'optimizer': trial.suggest_categorical('optimizer', [optim.SGD, optim.RMSprop]
               }

        # Additional tuning parameters: n_epochs, model, optmizer, seed, activation function, n_epochs, momentum (see: https://perlitz.github.io/hyperparameter-optimization-with-optuna/)


        # Get training and val datasets
        train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                           filter_ground=True, filter_height=1.3)
        val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,
                                         filter_ground=True, filter_height=1.3)

        # Augment training data
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
                print(
                    f"Adding {i + 1} augmentation of {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

        # Set up pytorch training and validation loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


        #Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        #Loop through each epoch
        for epoch in range(0, num_epochs):

            # Set training loop
            model.train()
            loss_list = []
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)[:, 0]
                loss = F.mse_loss(out, data.y)
                loss.backward()
                optimizer.step()
                if (i + 1) % 1 == 0:
                    print(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} ')
                    loss_list.append(loss.detach().to("cpu").numpy())
            print(f'mean loss this epoch: {np.mean(loss_list)}')

            # Set validation loop
            with torch.no_grad():
                model.eval()
                losses = []
                for idx, data in enumerate(val_loader):
                    data = data.to(device)
                    outs = model(data)[:, 0]
                    loss = F.mse_loss(outs, data.y)
                    print(data.y.to('cpu').numpy() - outs.to('cpu').numpy())
                    losses.append(float(loss.to("cpu")))
                val_mse = float(np.mean(losses))

            print(f'Epoch: {epoch:02d}, Mean val MSE: {val_mse:.4f}')

            # Report validation MSE in optuna for this trial
            trial.report(val_mse, epoch)

            # Prune trial (stop early) if it is going poorly
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_mse


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600) #timeout is how long the optimization will last for

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # Look into which hyperparameters are most important using optuna.importance.get_param_importances()

