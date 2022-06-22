#Purpose of script: Move las files from general folder to train/test folder randomly

import os
import random
import shutil

#First move 70% of data to training folder
source = 'D:\Romeo_Data\Lidar_Clipped_to_plots' #Specify source folder
dest = 'D:\Romeo_Data\Lidar_Clipped_to_plots\train' #Specify folder to move files to
files = os.listdir(source) #List file names in directory
no_of_files = len(files) * 0.7 #Get 70% of the files

#Move files to training folder
for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)