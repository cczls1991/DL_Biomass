import glob
import os
import random
from datetime import datetime
from pathlib import Path

import laspy
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt

import pyarrow.parquet as pq
import pandas as pd


#NOTE: this script only works if you are using normalized intensity as the use_column attribute.



def normalize_intensity(intensity_vals):
    i_norm = ((intensity_vals - min(intensity_vals)) / (max(intensity_vals) - min(intensity_vals)))*20 #Multiply by 20 so that intensity vals take on similar range to biomass vals
    return i_norm


def read_las(pointcloudfile, get_attributes=False, useevery=1):
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


def farthest_point_sampling(coords, k):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)
    return use_idx


def write_las(outpoints, outfilepath, attribute_dict):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)


def resample_point_clouds(in_dir, out_dir, num_points, use_columns = [], samp_meth = "random", glob="*.las", use_paqruet=True):

    # Create training set for each point density
    files = list(Path(in_dir).glob(glob))

    for file in tqdm(files, colour="red"):

        # Read las/laz file
        coords, attrs = read_las(file, get_attributes=True)
        plotID = str(file).split("\\")[-1][:-4]

        # Normalize Intensity
        attrs["intensity_normalized"] = normalize_intensity(attrs["intensity"])

        #Subset attributes to use columns only
        if len(use_columns) > 0:
            attrs = {key: attrs[key] for key in use_columns}
        else:
            attrs = {}


        # Resample number of points to num_points
        if coords.shape[0] >= num_points:
            if samp_meth == "random":
                use_idx = np.random.choice(
                    coords.shape[0], num_points, replace=False
                )
            elif samp_meth == "fps":
                use_idx = farthest_point_sampling(coords, num_points)
        else:
            use_idx = np.random.choice(coords.shape[0], num_points, replace=True)

        # Get subsetted point cloud
        coords = coords[use_idx, :]

        #Get subsetted attribute
        attrs_arr = list(attrs.items())[0][1]
        attrs_arr = attrs_arr[use_idx]

        # Centralize coordinates
        coords = coords - np.mean(coords, axis=0)

        if use_paqruet is True:
            #Combine coords and attributes arrays
            coors_attr = np.column_stack((coords, attrs_arr))
            #Convert nparray to pandas df
            df = pd.DataFrame(coors_attr, columns=['x', 'y', 'z', 'i_norm'])
            #Write parquet file
            df.to_parquet(os.path.join(out_dir, str(plotID + "_" + samp_meth + "_" +  str(num_points) + ".parq")))

        else:
            # Write file in LAS format
            write_las(outpoints=coords,
                      outfilepath=os.path.join(out_dir, str(plotID + "_" + samp_meth + "_" +  str(num_points) + ".las")),
                      attribute_dict={'intensity_normalized': attrs_arr}
                      )


def check_resampling(in_dir=None):

    #Get file list
    files = list(Path(in_dir).glob("*"))
    # Random sample of 4 files for vis
    files = random.sample(files, 4)
    #Check if files are parq or las
    ext = os.path.splitext(files[0])[1]
    # Create empty coords list
    coords_list = []
    #Load parquet file
    if ext == ".parq":
        print("Loading parquet files.")
        # Grab the LAS files of the four plots
        for i in range(0, len(files), 1):
            # Get filepath
            parq_path = files[i]
            coords_attrs = pd.read_parquet(parq_path, columns=["x", "y", "z"])
            coords_i = coords_attrs.to_numpy()
            # Add to list
            coords_list.append(coords_i)
    else:
        # Grab the LAS files of the four plots
        for i in range(0, len(files), 1):
            # Get filepath
            las_path = files[i]
            # Load coords for each LAS
            coords_i = read_las(las_path, get_attributes=False)
            # Add to list
            coords_list.append(coords_i)

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=[30, 30])

    # set up the axes for the first plot
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(coords_list[0][:, 0], coords_list[0][:, 1], coords_list[0][:, 2], c=coords_list[0][:, 2],
               cmap='viridis', linewidth=0.5)

    # set up the axes for the second plot
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(coords_list[1][:, 0], coords_list[1][:, 1], coords_list[1][:, 2], c=coords_list[1][:, 2],
               cmap='viridis', linewidth=0.5)

    # set up the axes for the third plot
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(coords_list[2][:, 0], coords_list[2][:, 1], coords_list[2][:, 2], c=coords_list[2][:, 2],
               cmap='viridis', linewidth=0.5)

    # set up the axes for the fourth plot
    ax = fig.add_subplot(224, projection='3d')
    ax.scatter(coords_list[3][:, 0], coords_list[3][:, 1], coords_list[3][:, 2], c=coords_list[3][:, 2],
               cmap='viridis', linewidth=0.5)

    plt.show()


if __name__ == "__main__":

    in_path = r"D:\Sync\Data\Model_Input\lidar_data"
    out_path = r"D:\Sync\Data\Model_Input\resampled_point_clouds\fps_7168_parquet"

    check_resampling(out_path)

    resample_point_clouds(
        in_dir=in_path,
        out_dir=out_path,
        use_columns=["intensity_normalized"],
        num_points=7168,
        samp_meth="fps",
        use_paqruet=True
    )

    # Randomly sample 4 plots to check downsampling--------------------------------------------------------------------
    check_resampling(out_path)







