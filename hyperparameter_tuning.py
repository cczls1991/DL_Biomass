import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from augmentation import AugmentPointCloudsInFiles
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

comment = "focusing_on_important_HPs"

if __name__ == '__main__':

    # Specify cuda as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")

    # SET UP STATIC PARAMETERS
    use_columns = ['intensity_normalized']
    pruning = False
    early_stopping = True
    max_num_epochs = 400
    n_trials = None
    run_time = 3600*16  # Time in seconds that the hyperparameter tuning will run for (multiply by 3600 to convert to hours)
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'
    val_dataset_path = r'D:\Sync\Data\Model_Input\val'

    # Define obective function used in optuna
    def objective(trial):

        # SET UP TUNING PARAMETERS
        cfg = {'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
               'batch_size': trial.suggest_int('batch_size', low=2, high=32, step=2),
               'weight_decay': 3.6384310505999963e-10, #trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True),
               'num_augs': 8, #trial.suggest_int('num_augs', low=0, high=10, step=1),
               'num_points': 7000, #trial.suggest_int('num_points', low=500, high=3000, step=500),
               'neuron_multiplier': 0, #trial.suggest_int('neuron_multiplier', low=0, high=2, step=2),
               'patience': 50, #trial.suggest_int('patience', low=5, high=50, step=5),
               'ground_filter_height': 0, #trial.suggest_float("ground_filter_height", 0, 2, step=0.2),
               'activation_function': 'ReLU', #trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
               'optimizer': "Adam", #trial.suggest_categorical('optimizer', ["Adam", "AdamW"]),
               'dropout_probability': 0.55, #trial.suggest_float("dropout_probability", 0.4, 0.8, step=0.05)
               # 'lidar_attrs': trial.suggest_categorical('lidar_attrs', ['intensity_normalized', 'classification', 'return_num'])
               }


        #Specify input lidar attributes
        #if cfg['lidar_attrs'] == 'classification':
            #use_columns = ['classification']
        #elif cfg['lidar_attrs'] == 'return_num':
            #use_columns = ['return_num']
        #else:
            #use_columns = ['intensity_normalized']

        #Set model hyperparameters
        model = Net(num_features=len(use_columns),
                    activation_function=cfg['activation_function'],
                    neuron_multiplier=cfg['neuron_multiplier'],
                    dropout_probability=cfg['dropout_probability']
                    ).to(device)

        # Get training and val datasets
        train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=cfg['num_points'], use_columns=use_columns, filter_height=cfg['ground_filter_height'])
        val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=cfg['num_points'], use_columns=use_columns, filter_height=cfg['ground_filter_height'])

        # Set up pytorch training and validation loaders
        train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
        # Augment training data
        if cfg['num_augs'] > 0:
            for i in range(cfg['num_augs']):
                aug_trainset = AugmentPointCloudsInFiles(
                    train_dataset_path,
                    "*.las",
                    max_points=cfg['num_points'],
                    use_columns=use_columns
                )

                # Concat training and augmented training datasets
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])

        # Set optimizer
        if cfg['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        #Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        # Loop through each epoch
        for epoch in range(0, max_num_epochs):
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

            #Save lowest val MSE achieved in this trial
            val_mse_list.append(val_mse)

            #Early stopping
            if early_stopping is True:
                if val_mse > last_val_mse:
                    trigger_times += 1
                    if trigger_times >= cfg['patience']:
                        print(f'\nEarly stopping at epoch {epoch}!\n')
                        return min(val_mse_list)
                else:
                    trigger_times = 0
                    last_val_mse = val_mse

            # Report validation MSE in optuna for this trial epoch
            trial.report(val_mse, epoch)

            # Prune trial (stop early) if it is going poorly (built in Optuna early stopping)
            if pruning is True:
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return min(val_mse_list)


    # Begin hyperparameter tuning trials ------------------------------------------------------------
    print("Begining tuning for", comment)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=run_time, show_progress_bar=True)

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

    print("Run time:", run_time/3600, "hours")

    # Visualize parameter importance ------------------------------------------------------------
    param_importance = optuna.importance.get_param_importances(study)
    names = list(param_importance.keys())
    values = list(param_importance.values())
    plt.bar(range(len(param_importance)), values, tick_label=names)
    plt.xlabel("Parameter")
    plt.ylabel("Importance Score")
    plt.title("Optuna Variable Importance")
    plt.show()

    # Convert hyperparameter tuning results to df and export as excel file  ------------------------------------------------------------
    df = study.trials_dataframe()
    assert isinstance(df, pd.DataFrame)
    t_now = dt.now().strftime("%Y_%m_%d_%H_%M_%S")
    #Remove column suffixes
    df.columns = df.columns.str.replace("params_", "")
    df.to_csv(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\{comment}_Hyperparameter_tuning_results_{t_now}_.csv")
