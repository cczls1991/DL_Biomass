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
import sklearn.metrics as metrics
from math import sqrt
import pprint as pp

if __name__ == '__main__':

    #Specify hyperparameter file
    hp_file = "D:\Sync\DL_Development\Hyperparameter_Tuning\Sunday_Monday_more_params__Hyperparameter_tuning_results_2022_06_27_08_30_41_.csv"
    #...or get most recent hyperparameter tuning
    if hp_file is None:
        folder_path = r'D:\Sync\DL_Development\Hyperparameter_Tuning'
        file_type = r'\*.csv'
        files = glob.glob(folder_path + file_type)
        hp_file = max(files, key=os.path.getctime)

    # Load hyperparameter tunings
    hp = pd.read_csv(hp_file, sep=",", header=0, index_col=0)
    hp = hp.sort_values(by="value", ascending=True)
    #Drop unwated cols
    hp = hp.drop(labels=['number', 'datetime_start', 'datetime_complete', 'duration', 'state'], axis=1)
    #Select the params with the lowest MSE value (row 1)
    hp = hp.iloc[0]
    #Convert to dictionary
    hp = hp.to_dict()
    print("Hyperparameters from:\n", hp_file, "\n")
    pp.pprint(hp, width=1)

    #SETUP ADDITIONAL HYPERPARAMETERS
    model_path = rf'D:\Sync\DL_Development\Models\DL_model_{dt.now().strftime("%Y_%m_%d_%H_%M_%S")}.model'
    use_columns = ['intensity_normalized']
    num_points = 2_000
    early_stopping = True
    num_epochs = 200
    writer = SummaryWriter(comment="updated_hyperparameters")
    train_dataset_path = r'D:\Sync\Data\Model_Input\train'
    val_dataset_path = r'D:\Sync\Data\Model_Input\val'

    #Device, model and optimizer setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(num_features=len(use_columns),
                activation_function=hp['activation_function'],
                neuron_multiplier=hp['neuron_multiplier'],
                dropout_probability=hp['dropout_probability']
                ).to(device)

    # Set optimizer
    if hp['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    #Set device
    print(f"Using {device} device.")

    #Get training and val datasets
    train_dataset = PointCloudsInFiles(train_dataset_path, '*.las', max_points=num_points, use_columns=use_columns, filter_height=hp['ground_filter_height'])
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,  filter_height=hp['ground_filter_height'])

    #Augment training data
    if hp['num_augs'] > 0:
        for i in range(hp['num_augs']):
            aug_trainset = AugmentPointCloudsInFiles(
                train_dataset_path,
                "*.las",
                max_points=num_points,
                use_columns=use_columns,
            )

            # Concat training and augmented training datasets
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, aug_trainset])
        print(f"Adding {hp['num_augs']} augmentations of original {len(aug_trainset)} for a total of {len(train_dataset)} training samples.")

    #Set up pytorch training and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=0)

#Define training function
    def train():
        model.train()
        loss_list = []
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)[:, 0]
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                tqdm.write(str(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} '))
                loss_list.append(loss.detach().to("cpu").numpy())
            #tqdm.write(str('Mean loss this epoch:', str(np.mean(loss_list))))
        return np.mean(loss_list)

    def val(loader, ep_id): #Note sure what ep_id does
        with torch.no_grad():
            model.eval()
            losses = []
            for idx, data in enumerate(loader):
                data = data.to(device)
                outs = model(data)[:, 0]
                loss = F.mse_loss(outs, data.y)
                losses.append(float(loss.to("cpu")))
            return float(np.mean(losses))

    def main():
        #Add trigger_times, last val MSE value, and val mse list for early stopping process
        trigger_times = 0
        last_val_mse = np.inf
        val_mse_list = []

        #Training loop
        for epoch in tqdm(range(0, num_epochs), colour="green"):
            train_mse = train()
            val_mse = val(val_loader, epoch)

            #Record epoch results
            if num_epochs > 10:
                writer.add_scalar("Training MSE", train_mse, epoch)  # Save to tensorboard summary
                writer.add_scalar("Validation MSE", val_mse, epoch)  # Save to tensorboard summary
            with open(model_path.replace('.model', '.csv'), 'a') as f:
                f.write(
                    f'{epoch}, {train_mse}, {val_mse}\n'
                )

            #Early stopping
            if early_stopping is True:
                if val_mse > last_val_mse:
                    trigger_times += 1
                    tqdm.write("    Early stopping trigger " + str(trigger_times) + " out of " + str(hp['patience']))
                    if trigger_times >= hp['patience']:
                        print(f'\nEarly stopping at epoch {epoch}!\n')
                        return
                else:
                    trigger_times = 0
                    last_val_mse = val_mse

            #Report epoch stats
            tqdm.write("    Epoch: " + str(epoch) + "  | Mean val MSE: " + str(round(val_mse, 2)) + "  | Mean train MSE: " + str(round(train_mse, 2)))

            #Determine whether to save the model based on val MSE
            val_mse_list.append(val_mse)
            if val_mse <= min(val_mse_list):
                tqdm.write("    Saving model for epoch " + str(epoch))
                torch.save(model, model_path)

        # Terminate tensorboard writer
        writer.flush()
        writer.close()

        print(f"\nFinished all {num_epochs} training epochs.")

        return

    #Run training loop
    main()

    #Plot the change in training and validation MSE --------------------------------------------------

    #Load most recent model and set of training results
    folder_path = r'D:\Sync\DL_Development\Models'
    file_type = r'\*.csv'
    files = glob.glob(folder_path + file_type)
    training_results = max(files, key=os.path.getctime)
    training_results = pd.read_csv(training_results, sep=",", header=None)
    training_results.columns = ['epoch', 'train_mse', 'val_mse']

    #Plot the change in training and validation mse over time
    fig, ax = plt.subplots()
    ax.plot(training_results["epoch"], training_results["train_mse"], color="blue", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.plot(training_results["epoch"], training_results["val_mse"], color="red", marker="o")
    red_patch = mpatches.Patch(color='red', label='Validation')
    blue_patch = mpatches.Patch(color='blue', label='Training')
    plt.legend(handles=[red_patch, blue_patch])

    # Apply the model to test data ---------------------------------------------------------------------------------

    # Get test data
    test_dataset = PointCloudsInFiles(r"D:\Sync\Data\Model_Input\test", '*.las', max_points=num_points,
                                      use_columns=use_columns,
                                      filter_height=0.2)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, num_workers=0)

    model.eval()
    for idx, data in enumerate(test_loader):
        data = data.to(device)
    pred = model(data)[:, 0].to('cpu').detach().numpy()
    obs = data.y.to('cpu').detach().numpy()

    # Calculate R^2 and RMSE for test dataset
    test_r2 = round(metrics.r2_score(obs, pred), 3)
    test_rmse = round(sqrt(metrics.mean_squared_error(obs, pred)), 2)
    print(f"\nResults for test data: \nR2: {test_r2}\nRMSE: {test_rmse}")

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


