import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import laspy
import numpy as np
import pandas as pd
from itertools import compress

#NOTE: PointCloudsInFilesPreSampled only works if you are using normalized intensity as the use_column attribute.


def read_las(pointcloudfile, get_attributes=False, useevery=1, filter_height=0.2):
    '''
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    '''

    # Read the file
    inFile = laspy.read(pointcloudfile)
    # get the coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    #Remove points below specified threshold
    if filter_height > 0:
        filter_arr = coords[:, 2] > filter_height
        coords = coords[filter_arr]

    #Retrieve attributes
    if get_attributes == False:
        return (coords)
    else:
        las_fields= [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        for las_field in las_fields[3:]: # skip the X,Y,Z fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


def normalize_intensity(intensity_vals):
    i_norm = ((intensity_vals - min(intensity_vals)) / (max(intensity_vals) - min(intensity_vals)))*20 #Multiply by 20 so that intensity vals take on similar range to biomass vals
    return i_norm


class PointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, root_dir, glob='*', max_points=200_000, use_columns=None, filter_height=1.3, dataset = ["RM", "PF"]):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            column_name (string): Column name to use as target variable (e.g. "Classification")
            max_points (integer): the number of points that will be used from an input point cloud.
            use_columns (list[string]): Column names to add as additional input
            filter_height (numeric): height (in meters) below which points will be removed
            dataset (list[string]): dataset(s) which will be used in training and validation
        """
        #Set number of points per LAS to load
        self.max_points = max_points

        #List files
        self.files = list(Path(root_dir).glob(glob))
        #Get dataset source for each LAS file
        dataset_ID = []
        for i in range(0, len(self.files), 1):
            dataset_ID.append(self.files[i].name.split(".")[0][0:2])
        #Convert to pandas series
        dataset_ID = pd.Series(dataset_ID, dtype=str)
        #Determine whether or not to keep each file based on dataset ID
        dataset_filter = dataset_ID.isin(dataset).tolist()
        #Filter files to target dataset(s)
        self.files = list(compress(self.files, dataset_filter))

        #Set up use column
        if use_columns is None:
            use_columns = []
        self.use_columns = use_columns
        self.filter_height = filter_height

        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = str(self.files[idx])
        coords, attrs = read_las(filename, get_attributes=True, filter_height=self.filter_height)

        #Normalize Intensity
        attrs["intensity_normalized"] = normalize_intensity(attrs["intensity"])

        if coords.shape[0] >= self.max_points:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=False)
        else:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=True)
        if len(self.use_columns) > 0:
            x = np.empty((self.max_points, len(self.use_columns)), np.float32)
            for eix, entry in enumerate(self.use_columns):
                x[:, eix] = attrs[entry][use_idx]
        else:
            x = coords[use_idx, :]
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # Load biomass data
        #Get plot ID from filename
        PlotID = self.files[idx].name.split(".")[0]
        #Load biomass data
        input_table = pd.read_csv(r"D:\Sync\Data\Model_Input\model_input_plot_biomass_data.csv", sep=",", header=0)
        #Extract bark, branch, foliage, wood values for the correct plot ID
        bark_agb = input_table.loc[input_table["PlotID"] == PlotID]["bark_btphr"].values[0]
        branch_agb = input_table.loc[input_table["PlotID"] == PlotID]["branch_btphr"].values[0]
        foliage_agb = input_table.loc[input_table["PlotID"] == PlotID]["foliage_btphr"].values[0]
        wood_agb = input_table.loc[input_table["PlotID"] == PlotID]["wood_btphr"].values[0]
        #Combine AGB targets into a list
        target = [bark_agb, branch_agb, foliage_agb, wood_agb]

        #Aggregate point cloud and biomass targets for the given sample
        sample = Data(x=torch.from_numpy(x).float(),
                      y=torch.from_numpy(np.array(target)).float(),
                      pos=torch.from_numpy(coords[use_idx, :]).float(),
                      PlotID=PlotID)

        if coords.shape[0] < 100:
            return None
        return sample


class PointCloudsInFilesPreSampled(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, root_dir, glob='*', dataset=("RM", "PF", "BC"), use_column="intensity_normalized"):

        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            dataset (list[string]): dataset(s) which will be used in training and validation
            use_column: lidar attribute to use as additional predictor variable
        """

        #List files
        self.files = list(Path(root_dir).glob(glob))
        #Get dataset source for each LAS file
        dataset_ID = []
        for i in range(0, len(self.files), 1):
            dataset_ID.append(self.files[i].name.split(".")[0][0:2])
        #Convert to pandas series
        dataset_ID = pd.Series(dataset_ID, dtype=str)
        #Determine whether or not to keep each file based on dataset ID
        dataset_filter = dataset_ID.isin(dataset).tolist()
        #Filter files to target dataset(s)
        self.files = list(compress(self.files, dataset_filter))

        #Add use column
        self.use_column = use_column

        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = str(self.files[idx])
        coords, attrs = read_las(filename, get_attributes=True, filter_height=0)

        #Select target variable (use_column) from attributes dictionary
        attrs = {key: attrs[key] for key in [self.use_column]}
        x = np.array(list(attrs.items())[0][1])
        x = np.reshape(x, [len(coords), 1])

        # Load biomass data -------------

        #Get plot ID from filename
        PlotID = self.files[idx].name.split(".")[0]
        PlotID = PlotID.replace('_fps_3072', '') #Remove unwanted part of filename

        #Load biomass data
        input_table = pd.read_csv(r"D:\Sync\Data\Model_Input\model_input_plot_biomass_data.csv", sep=",", header=0)
        #Extract bark, branch, foliage, wood values for the correct plot ID
        bark_agb = input_table.loc[input_table["PlotID"] == PlotID]["bark_btphr"].values[0]
        branch_agb = input_table.loc[input_table["PlotID"] == PlotID]["branch_btphr"].values[0]
        foliage_agb = input_table.loc[input_table["PlotID"] == PlotID]["foliage_btphr"].values[0]
        wood_agb = input_table.loc[input_table["PlotID"] == PlotID]["wood_btphr"].values[0]
        #Combine AGB targets into a list
        target = [bark_agb, branch_agb, foliage_agb, wood_agb]

        #Aggregate point cloud and biomass targets for the given sample
        sample = Data(x=torch.from_numpy(x).float(),
                      y=torch.from_numpy(np.array(target)).float(),
                      pos=torch.from_numpy(coords).float(),
                      PlotID=PlotID)

        if coords.shape[0] < 100:
            return None
        return sample