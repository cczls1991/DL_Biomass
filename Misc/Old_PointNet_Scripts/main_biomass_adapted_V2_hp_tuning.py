#Notes:
        #Updated script to include intensity as a predictor
        #Updated to use 10_000 points instead of 4_000

#Get modules  
import os

import numpy as np
from time import time

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from pn2_regressor_biomass_adapted import Net
from pointcloud_dataset_biomass_adapted import PointCloudsInFiles
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import R2Score

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

if __name__ == '__main__':

    # Record start time
    start = time()

    # Set up tensorboard summary writer
    writer = SummaryWriter(comment="_10000_points_lr_0.0001_batch_size_1")

    # Set additional variables to be used for each point in LAS
    use_columns = None #['intensity']

    # Numer of features is determined by the number of used columns
    if use_columns is None:
        num_features = 0
    else:
        num_features = len(use_columns)

    #Set number of epochs
    n_epochs = 1000

    #Set the number of points to be used per plot
    n_points = 10_000

    #Define load_data function
    def load_data():

        train_dataset = PointCloudsInFiles(r'/Romeo_Data/train',
                                           '*.las',
                                           max_points=n_points, use_columns=use_columns)
        val_dataset = PointCloudsInFiles(r'/Romeo_Data/test',
                                          '*.las',
                                         max_points=n_points, use_columns=use_columns)

        print(f"We have {len(train_dataset)} training samples")

        return train_dataset, val_dataset



    #Define function for hyperparameter tuning
    def train_cifar(config, checkpoint_dir = None):

        net = Net(config["l1"])

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)

        print(f"Using {device} device.")

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        # Set the batch size
        batch_size = int(config["batch_size"])

        #Load data
        train_dataset, val_dataset = load_data()

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0)
        valloader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0)


        #Begin the training loop
        for epoch in range(1, n_epochs):
            model_path = rf'/DL_Development/Models/latest.model'
            if os.path.exists(model_path):
                model = torch.load(model_path)

            #Training
            model.train() # Put module into training mode
            for i, data in enumerate(trainloader):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.mse_loss(out[:, 0], data.y)
                loss.backward()
                optimizer.step()
                if (i + 1) % 1 == 0:
                    print(f'[{i + 1}/{len(trainloader)}] MSE Loss: {loss.to("cpu"):.4f} ')

            #Validation
            with torch.no_grad():
                model.eval() #Set module into evaluation mode
                losses = []
                r2scores = []
                for idx, data in enumerate(valloader):
                    # Send data to cuda and apply model
                    data = data.to(device)
                    outs = model(data)

                    # Get MSE and use as loss function
                    loss = F.mse_loss(outs[:, 0], data.y)
                    losses.append(float(loss.to("cpu")))

                    # Get R^2 as well, just for info (not used as loss)
                    r2score = R2Score().to(device)
                    r2score = r2score(outs[:, 0], data.y)
                    r2scores.append(float(r2score.to("cpu")))

            #Save mode
            torch.save(model, model_path)

            # Save details for epoch to summary report
            writer.add_scalar("MSE", mse, epoch)  # Save the Mean Squared Error (MSE)
            writer.add_scalar("Mean R2", r2, epoch)  # Save the mean R-Squared (R2) value


        print(f'Epoch: {epoch:02d}, Mean test MSE: {mse:.4f}')


    def test_performance(net, device="cpu"):
        test_dataset = PointCloudsInFiles(r'/Romeo_Data/test',
                                           '*.las',
                                          max_points=n_points, use_columns=use_columns)

        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0)

        with torch.no_grad():
            for idx, data in enumerate(testloader):
                # Send data to cuda and apply model
                data = data.to(device)
                outs = model(data)

                # Get MSE and use as loss function
                loss = F.mse_loss(outs[:, 0], data.y)
                losses.append(float(loss.to("cpu")))


        return correct / total

    def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
        data_dir = os.path.abspath("./data")
        load_data(data_dir)

        config = {
            "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([2, 4, 8, 16])
        }

        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)

        reporter = CLIReporter(
            # parameter_columns=["l1", "lr", "batch_size"],
            metric_columns=["loss", "mse", "training_iteration"])

        result = tune.run(
            partial(train_cifar, data_dir=data_dir),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation mse: {}".format(
            best_trial.last_result["mse"]))

        best_trained_model = Net(best_trial.config["l1"])
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(
            best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_mse = test_performance(best_trained_model, device)
        print("Best trial test set mse: {}".format(test_mse))

    # Run everything:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5)

    # Call flush() method to make sure that all pending events have been written to disk.
    writer.flush()
    #Close summary writer
    writer.close()

    print(f'Script run time: {round(time()/3600 - start/3600, 2)}', "hours over", n_epochs, "epochs")
