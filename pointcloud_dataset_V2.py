import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import laspy
import numpy as np
import pandas as pd


def read_las(pointcloudfile, get_attributes=False, useevery=1, filter_height=1.3):
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

    def __init__(self, root_dir, glob='*', max_points=200_000, use_columns=None, filter_height=1.3):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.files = list(Path(root_dir).glob(glob))
        self.max_points = max_points
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

        # Load target biomass data
        #Get plot ID from filename
        plotID = self.files[idx].name.split(".")[0]
        #Load biomass data
        input_table = pd.read_csv(r"D:\Sync\Data\Model_Input\model_input_plot_biomass_data.csv", sep=",", header=0)
        #Extract target value for the correct plot ID
        target = input_table.loc[input_table["PlotID"] == plotID]["total_AGB"].values

        sample = Data(x=torch.from_numpy(x).float(),
                      y=torch.from_numpy(np.array(target)).float(),
                      pos=torch.from_numpy(coords[use_idx, :]).float())
        if coords.shape[0] < 100:
            return None
        return sample