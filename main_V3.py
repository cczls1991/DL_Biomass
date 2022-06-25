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
    use_columns = ['intensity_normalized']
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
    val_dataset = PointCloudsInFiles(val_dataset_path, '*.las', max_points=num_points, use_columns=use_columns,  filter_height=ground_filter_height)

    #Augment training data
    if num_augs > 0:
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

    #Set up pytorch training and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
                #tqdm.write(str(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} '))
                loss_list.append(loss.detach().to("cpu").numpy())
            #tqdm.write('Mean loss this epoch:', np.mean(loss_list))
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
                    tqdm.write("    Early stopping trigger " + str(trigger_times) + " out of " + str(patience))
                    if trigger_times >= patience:
                        print(f'\nEarly stopping at epoch {epoch}!\n')
                        return
                else:
                    trigger_times = 0
                    last_val_mse = val_mse

            #Report epoch stats
            tqdm.write("    Epoch: " + str(epoch) + "  | Mean val MSE: " + str(
                round(val_mse, 2)) + "  | Mean train MSE: " + str(round(train_mse, 2)))

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
    model = main()


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


