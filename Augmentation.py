import glob
import os
import random
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

import pandas as pd


def read_las(pointcloudfile, get_attributes=True, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


def rotate_points(coords):
    rotation = np.random.uniform(-180, 180)
    # Convert rotation values to radians
    rotation = np.radians(rotation)

    # Rotate point cloud
    rot_mat = np.array(
        [
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1],
        ]
    )

    aug_coords = coords
    aug_coords[:, :3] = np.matmul(aug_coords[:, :3], rot_mat)
    return aug_coords


def point_removal(coords, x=None):
    # Get list of ids
    idx = list(range(np.shape(coords)[0]))
    random.shuffle(idx)  # shuffle ids
    idx = np.random.choice(
        idx, random.randint(round(len(idx) * 0.9), len(idx)), replace=False
    )  # pick points randomly removing up to 10% of points

    # Remove random values
    aug_coords = coords[idx, :]  # remove coords
    if x is None:  # remove x
        aug_x = aug_coords
    else:
        aug_x = x[idx, :]

    return aug_coords, aug_x


def random_noise(coords, dim, x=None):
    # Random standard deviation value
    random_noise_sd = np.random.uniform(0.01, 0.025)

    # Add/Subtract noise
    if np.random.uniform(0, 1) >= 0.5:  # 50% chance to add
        aug_coords = coords + np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = aug_coords
        else:
            aug_x = x + np.random.normal(0, random_noise_sd, size=(np.shape(x)[0], dim))
    else:  # 50% chance to subtract
        aug_coords = coords - np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = aug_coords
        else:
            aug_x = x - np.random.normal(0, random_noise_sd, size=(np.shape(x)[0], dim))

    # Randomly choose up to 10% of augmented noise points
    use_idx = np.random.choice(
        aug_coords.shape[0], random.randint(0, round(len(aug_coords) * 0.1)), replace=False
    )
    aug_coords = aug_coords[use_idx, :]  # get random points
    aug_coords = np.append(coords, aug_coords, axis=0)  # add points
    aug_x = aug_x[use_idx, :]  # subset random values for attribute(s)
    aug_x = np.append(x, aug_x, axis=0)  # add random values for attribute(s)

    return aug_coords, aug_x


class AugmentPointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(
            self, root_dir, glob="*", max_points=200_000, use_columns=None
    ):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            use_columns (list[string]): Column names to add as additional input
        """
        self.files = list(Path(root_dir).glob(glob))
        self.max_points = max_points
        if use_columns is None:
            use_columns = []
        self.use_columns = use_columns
        super().__init__()

    def __len__(self):
        # Return length
        return len(self.files)  # NEED TO ADD MULTIPLICATION FOR NUMBER OF AUGMENTS

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get file name
        filename = str(self.files[idx])

        # Read las/laz file
        coords, attrs = read_las(filename, get_attributes=True)

        # Normalize Intensity
        attrs["intensity_normalized"] = attrs["intensity"] / 1000

        # Resample number of points to max_points
        if coords.shape[0] >= self.max_points:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=False)
        else:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=True)

        # Get x values
        if len(self.use_columns) > 0:
            x = np.empty((self.max_points, len(self.use_columns)), np.float32)
            for eix, entry in enumerate(self.use_columns):
                x[:, eix] = attrs[entry][use_idx]
        else:
            x = coords[use_idx, :]

        # Get coords
        coords = coords[use_idx, :]
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # Augmentation
        coords, x = point_removal(coords, x)
        coords, x = random_noise(coords, len(self.use_columns), x)
        coords = rotate_points(coords)

        # Load target biomass data
        #Get plot ID from filename
        plotID = self.files[idx].name.split(".")[0]
        #Load biomass data
        input_table = pd.read_csv(r"D:\Sync\Romeo_Data\Outputs\romeo_plots_w_biomass.csv", sep=",", header=0)
        #Extract target value for the correct plot ID
        target = input_table.loc[input_table["PlotID"] == int(plotID)]["total_AGB"].values

        # Transform data to tensor
        sample = Data(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(np.array(target)).float(),
            pos=torch.from_numpy(coords).float(),
        )
        if coords.shape[0] < 100:
            return None
        return sample