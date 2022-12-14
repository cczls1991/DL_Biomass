import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from pointnet2_regressor import Net
from pointcloud_dataloader import PointCloudsInFiles
from pointcloud_dataloader import PointCloudsInFilesPreSampled
from augmentation import AugmentPointCloudsInFiles
from augmentation import AugmentPreSampledPoints
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import joblib
import glob
from tqdm import tqdm

# Supress warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Set note for this optuna study
    comment = "using_presampled_poi"

    # Set optuna study parameters
    pruning = True  # Should optuna prune trials that are going badly?
    continue_study = True  # Specify whether or not to continue previous study
    n_trials = None  # Number of trials to run, takes priority over run time
    run_time = 3600 * 24 * 4 # Time in seconds that the hyperparameter tuning will run for (multiply by 3600 to convert to hours)

    # SET UP MODEL PARAMETERS
    use_columns = ['intensity_normalized']
    use_datasets = ["BC", "RM", "PF"]  # Possible datasets: BC, RM, PF
    early_stopping = True
    max_num_epochs = 7168
    use_presampled = True

    #Specify dataset paths (dependent on whether or not using pre-sampled)
    if use_presampled is True:
        print("\nUsing pre-sampled points!\n")
        train_dataset_path = fr'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_train'
        val_dataset_path = r'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_val'
        test_dataset_path = r'D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_points_test'
    else:
        train_dataset_path = r'D:\Sync\Data\Model_Input\train'
        val_dataset_path = r'D:\Sync\Data\Model_Input\val'
        test_dataset_path = r'D:\Sync\Data\Model_Input\test'


    # Define obective function used in optuna
    def objective(trial):

        # SET UP TUNING PARAMETERS
        cfg = {'lr': trial.suggest_float("lr", 1e-6, 1e-1, log=True),
               'num_augs': trial.suggest_int('num_augs', low=0, high=10, step=1),
               'batch_size': trial.suggest_int('batch_size', low=8, high=40, step=4),
               'patience': trial.suggest_int('patience', low=5, high=30, step=5),
               'weight_decay': 8.0250963438986e-05, # trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True),
               'num_points': 7168,  # trial.suggest_int('num_points', low=5000, high=10_000, step=1000),
               'neuron_multiplier': 0,  # trial.suggest_int('neuron_multiplier', low=0, high=2, step=2),
               'ground_filter_height': 0,  # trial.suggest_float("ground_filter_height", 0, 2, step=0.2),
               'activation_function': 'ReLU',
               # trial.suggest_categorical('activation_function', ['ReLU', 'LeakyReLU', 'ELU']),
               'optimizer': "Adam",  # trial.suggest_categorical('optimizer', ["Adam", "AdamW"]),
               'dropout_probability': 0.5,  # trial.suggest_float("dropout_probability", 0.4, 0.8, step=0.05)
               }

        # Set model
        model = Net(num_features=len(use_columns),
                    activation_function=cfg['activation_function'],
                    neuron_multiplier=cfg['neuron_multiplier'],
                    dropout_probability=cfg['dropout_probability']
                    )

        if use_presampled == True:
            # Load pre sampled data
            train_dataset = PointCloudsInFilesPreSampled(train_dataset_path,
                                                         '*.las', dataset=use_datasets,
                                                         use_column="intensity_normalized")
            val_dataset = PointCloudsInFilesPreSampled(val_dataset_path,
                                                       '*.las', dataset=use_datasets, use_column="intensity_normalized")

            # Augment pre-sampled training data
            if cfg['num_augs'] > 0:
                for i in range(cfg['num_augs']):
                    aug_trainset = AugmentPreSampledPoints(
                        train_dataset_path,
                        "*.las",
                        use_columns="intensity_normalized",
                        dataset=use_datasets
                    )

                    # Concat training and augmented training datasets
                    train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
                print(
                    f"Adding {cfg['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

        else:
            train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=cfg['num_points'],
                                               use_columns=use_columns,
                                               filter_height=cfg['ground_filter_height'], dataset=use_datasets)
            val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=cfg['num_points'],
                                             use_columns=use_columns,
                                             filter_height=cfg['ground_filter_height'], dataset=use_datasets)

            # Augment training data
            if cfg['num_augs'] > 0:
                for i in range(cfg['num_augs']):
                    aug_trainset = AugmentPointCloudsInFiles(
                        train_dataset_path,
                        "*.las",
                        max_points=cfg['num_points'],
                        use_columns=use_columns,
                        filter_height=cfg['ground_filter_height'],
                        dataset=use_datasets
                    )

                    # Concat training and augmented training datasets
                    train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
                print(
                    f"Adding {cfg['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

            # Set up pytorch training and validation loaders
        train_loader = DataListLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        val_loader = DataListLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)

        # Initiate parallel processing
        model = DataParallel(model)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Set optimizer
        if cfg['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        # Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        # Loop through each epoch
        for epoch in range(0, max_num_epochs):
            model.train()
            loss_list = []
            for idx, data_list in enumerate(tqdm(train_loader, colour="green", position=0, leave=False, desc=f"Epoch {epoch} training")):
                optimizer.zero_grad()

                # Predict values and ensure that pred. and obs. tensors have same shape
                outs = model(data_list)
                y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list), 4))

                # Compute mse loss for each component
                loss_bark = F.mse_loss(y[:, 0], outs[:, 0])
                loss_branch = F.mse_loss(y[:, 1], outs[:, 1])
                loss_foliage = F.mse_loss(y[:, 2], outs[:, 2])
                loss_wood = F.mse_loss(y[:, 3], outs[:, 3])

                # Set the % contribution of each component to total tree biomass
                a = 1 / 11  # Bark comprises ~11% of tree biomass across entire dataset
                b = 1 / 12  # Branches comprise ~12% of tree biomass across entire dataset
                c = 1 / 5  # Foliage comprises ~5% of tree biomass across entire dataset
                d = 1 / 72  # Wood comprises ~72% of tree biomass across entire dataset

                # Calculate mse loss using loss for each component relative to its contribution to total biomass
                loss = loss_bark * a + loss_branch * b + loss_foliage * c + loss_wood * d

                loss.backward()
                optimizer.step()
                if (idx + 1) % 1 == 0:
                    loss_list.append(loss.detach().to("cpu").numpy())

            # Set validation loop
            with torch.no_grad():
                model.eval()
                losses = []
                for idx, data_list in enumerate(val_loader):
                    outs = model(data_list)
                    y = torch.reshape(torch.cat([data.y for data in data_list]).to(outs.device), (len(data_list), 4))

                    # Compute mse loss for each component
                    loss_bark = F.mse_loss(y[:, 0], outs[:, 0])
                    loss_branch = F.mse_loss(y[:, 1], outs[:, 1])
                    loss_foliage = F.mse_loss(y[:, 2], outs[:, 2])
                    loss_wood = F.mse_loss(y[:, 3], outs[:, 3])

                    # Set the % contribution of each component to total tree biomass
                    a = 1 / 11  # Bark comprises ~11% of tree biomass across entire dataset
                    b = 1 / 12  # Branches comprise ~12% of tree biomass across entire dataset
                    c = 1 / 5  # Foliage comprises ~5% of tree biomass across entire dataset
                    d = 1 / 72  # Wood comprises ~72% of tree biomass across entire dataset

                    # Calculate mse loss using loss for each component relative to its contribution to total biomass
                    loss = loss_bark * a + loss_branch * b + loss_foliage * c + loss_wood * d
                    losses.append(float(loss.to("cpu")))
                val_mse = float(np.mean(losses))

            # Add to val MSE list for this trial
            val_mse_list.append(val_mse)

            # Early stopping
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


    # HP TUNING SECTION ------------------------------------------------------------

    # If there is a previous study, and we specified to continue it, load  most recent study file
    if continue_study is True:
        # Check that there are files in the studies folder
        folder_path = r"E:\Optuna_Studies"
        file_type = r'\*.pkl'
        files = glob.glob(folder_path + file_type)
        if len(files) > 0:
            study_file = max(files, key=os.path.getctime)
            study = joblib.load(study_file)
            print(f"\nContinuing study: {study_file}\n")
            print("Best trial until now:")
            print(" Value: ", study.best_trial.value)
            print(" Params: ")
            for key, value in study.best_trial.params.items():
                print(f"    {key}: {value}")
            print(f"\nUsing {torch.cuda.device_count()} GPUs!")
            print(f"\nBegining tuning for ~{comment}~ study \n")
            study.optimize(objective, n_trials=n_trials, timeout=run_time, show_progress_bar=True)
            # Get current time and date
            t_now = dt.now().strftime("%Y_%m_%d_%H_%M_%S")

            # Save the study to resume later
            joblib.dump(study, fr"E:\Optuna_Studies\hp_study_{comment}_{t_now}.pkl")

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

            print("Run time:", run_time / 3600, "hours")

            # Visualize parameter importance ------------------------------------------------------------
            param_importance = optuna.importance.get_param_importances(study)
            names = list(param_importance.keys())
            values = list(param_importance.values())
            plt.bar(range(len(param_importance)), values, tick_label=names)
            plt.xlabel("Parameter")
            plt.ylabel("Importance Score")
            plt.title("Optuna Variable Importance")
            plt.show()

            # Save plot
            plt.savefig(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\{comment}hp_tuning{t_now}")

            # Convert hyperparameter tuning results to df and export as excel file  ------------------------------------------------------------
            df = study.trials_dataframe()
            assert isinstance(df, pd.DataFrame)
            # Remove column suffixes
            df.columns = df.columns.str.replace("params_", "")
            df.to_csv(
                rf"D:\Sync\DL_Development\Hyperparameter_Tuning\{comment}_Hyperparameter_tuning_results_{t_now}_.csv")

        else:
            print("\n* No previous studies to continue *")

    # If we are creating a new study -------------------------------------------------------------
    else:
        print("\nCreating new study.\n")
        print(f"\nUsing {torch.cuda.device_count()} GPUs!\n")
        print("\nBegining tuning for\n", comment)
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=run_time, show_progress_bar=True)

        # Get current time and date
        t_now = dt.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Save the study to resume later
        joblib.dump(study, fr"E:\Optuna_Studies\hp_study_{comment}_{t_now}.pkl")

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

        print("Run time:", run_time / 3600, "hours")

        # Visualize parameter importance ------------------------------------------------------------
        param_importance = optuna.importance.get_param_importances(study)
        names = list(param_importance.keys())
        values = list(param_importance.values())
        plt.bar(range(len(param_importance)), values, tick_label=names)
        plt.xlabel("Parameter")
        plt.ylabel("Importance Score")
        plt.title("Optuna Variable Importance")
        plt.show()

        # Save plot
        plt.savefig(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\{comment}hp_tuning{t_now}")

        # Convert hyperparameter tuning results to df and export as excel file  ------------------------------------------------------------
        df = study.trials_dataframe()
        assert isinstance(df, pd.DataFrame)
        # Remove column suffixes
        df.columns = df.columns.str.replace("params_", "")
        df.to_csv(rf"D:\Sync\DL_Development\Hyperparameter_Tuning\{comment}_Hyperparameter_tuning_results_{t_now}_.csv")
