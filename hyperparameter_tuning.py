import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pn2_regressor_biomass_adapted_V2 import Net
from pointcloud_dataset_biomass_adapted_V2 import PointCloudsInFiles
from Augmentation import AugmentPointCloudsInFiles
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt


if __name__ == '__main__':

    # Specify cuda as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")

    # SET UP STATIC PARAMETERS
    use_columns = ['intensity_normalized']
    num_points = 10_000
    augment = True
    num_augs = 2
    num_epochs = 100
    run_time = 12*3600  # Time in seconds that the hyperparameter tuning will run for
    train_dataset_path = r'D:\Sync\Romeo_Data\train'
    val_dataset_path = r'D:\Sync\Romeo_Data\val'

    # Define obective function to be used in optuna

    def objective(trial):

        # SET UP TUNING PARAMETERS
        cfg = {'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
               'weight_decay': trial.suggest_loguniform('weight_decay', 1e-10, 1e-3),
               'batch_size': trial.suggest_int('batch_size', low=4, high=32, step=4),
               'activation_function': trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
               'optimizer': trial.suggest_categorical('optimizer', [torch.optim.Adam, torch.optim.AdamW]),
               }

        # Additional parameters for config: n_epochs, model, optmizer, seed, activation function, n_epochs, momentum (see: https://perlitz.github.io/hyperparameter-optimization-with-optuna/)

        model = Net(num_features=len(use_columns), activation_function=cfg['activation_function']).to(device)

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


        # Set up pytorch training and validation loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=30, shuffle=True, num_workers=0)

        # Set optimizer
        optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        # Loop through each epoch
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
                    loss_list.append(loss.detach().to("cpu").numpy())

            # Set validation loop
            with torch.no_grad():
                model.eval()
                losses = []
                for idx, data in enumerate(val_loader):
                    data = data.to(device)
                    outs = model(data)[:, 0]
                    loss = F.mse_loss(outs, data.y)
                    losses.append(float(loss.to("cpu")))
                val_mse = float(np.mean(losses))


            # Report validation MSE in optuna for this trial
            trial.report(val_mse, epoch)

            # Prune trial (stop early) if it is going poorly
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_mse


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=None, timeout=run_time, show_progress_bar=True)

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

    # Visualize parameter importance ------------------------------------------------------------
    param_importance = optuna.importance.get_param_importances(study)
    names = list(param_importance.keys())
    values = list(param_importance.values())
    plt.bar(range(len(param_importance)), values, tick_label=names)
    plt.xlabel("Parameter")
    plt.ylabel("Importance Score")
    plt.title("Optuna Variable Importance")
    plt.show()

    # Visualize optimization history  ------------------------------------------------------------
    optuna.visualization.plot_optimization_history(study)

    # Export the hyperparameter tuning results as .xlsx file  ------------------------------------------------------------
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    t_now = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    df.to_excel(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\Hyperparameter_tuning_results_{t_now}_.xlsx")