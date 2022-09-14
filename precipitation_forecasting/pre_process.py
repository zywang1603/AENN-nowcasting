import os
import tarfile
import re
import numpy as np
import h5py
#import matplotlib.pyplot as plt
#import pandas as pd
from tqdm import tqdm
import zipfile
from datetime import datetime
#import cv2
import config
from batchcreator import minmax
from batchcreator import DataGenerator as dg
import tensorflow as tf

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path_ds = '/users/zhiyiwang/datasets/RT'


def load_h5(file_path):
    radar_img = None
    with h5py.File(file_path, 'r') as f:
        try:
            radar_img = f['image1']['image_data'][:]

            ## Set pixels out of image to 0
            out_of_image = f['image1']['calibration'].attrs['calibration_out_of_image']
            radar_img[radar_img == out_of_image] = 0
            # Sometimes 255 or other number (244) is used for the calibration
            # for out of image values, so also check the first pixel
            radar_img[radar_img == radar_img[0][0]] = 0
            # Original values are in 0.01mm/5min
            # Convert to mm/h:
            radar_img = (radar_img / 100) * 12
        except:
            print("Error: could not read image1 data, file {}".format(file_path))
    return radar_img


def rtcor2npy(in_dir, out_dir, year=None, label_dir=None, preprocess=False, overwrite=False, filenames=None):
    '''
    Preprocess the h5 file into numpy arrays.
    The timestamp, image1 and image2 data of each file is stored
    '''

    add_file_extension = ''
    prefix = ''
    if filenames is not None:
        out_dir = config.dir_rtcor_prep
        # Add file extension to filename
        # Add a prefix to filename
        add_file_extension = '.h5'
        prefix = config.prefix_rtcor
    else:
        d = in_dir + str(year) + '/'
        filenames = []
        for m in os.listdir(d):
            dir_m = os.path.join(d, m)
            if os.path.isdir(dir_m):
                for f in os.listdir(dir_m):
                    if f.endswith('.h5') and f.startswith(config.prefix_rtcor + str(year)):
                        filenames.append(f)
        filenames = sorted(filenames)

    # Create directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if label_dir and not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Pattern for detecting timestamp in filename
    regex_file = re.compile('(\d{12})\.h5')

    for filename in tqdm(filenames):
        filename = filename + add_file_extension

        timestamp = regex_file.findall(filename)[0]
        scan_fn = out_dir + '/' + "{}.npy".format(timestamp)
        year = timestamp[0:4]
        month = timestamp[4:6]
        path_scan = in_dir + str(year) + '/' + str(month) + '/' + prefix + filename
        #print(path_scan)

        if not overwrite and timestamp + '.npy' in output_files:
            # Skip this file if already processed,
            # go to next file in list
            continue
        try:
            radar_img = load_h5(path_scan)
            if preprocess:
                radar_img = perform_preprocessing(radar_img)
            np.save(scan_fn, radar_img)


        except Exception as e:
            print(e)
            # np.save(scan_fn, radar_data)



def perform_preprocessing(x, downscale256=True):
    #x = minmax(x, norm_method='minmax', undo=False, convert_to_dbz=False)
    x = np.expand_dims(x, axis=-1)
    if downscale256:
        # First make the images square size
        x = dg.pad_along_axis(dg, x, axis=0, pad_size=3)
        x = dg.pad_along_axis(dg, x, axis=1, pad_size=68)
        x = tf.image.resize(x, (256, 256))
    return x


# Get files that are already converted to numpy
output_files = sorted([f for f in os.listdir(config.dir_rtcor_prep)
                       if os.path.isfile(os.path.join(config.dir_rtcor_prep, f))])

print(len(output_files))
print('Approx {:.2f} years of data'.format(len(output_files) / 288 / 365))

# Load all target files in the training set
'''
arr = np.load('datasets/rainy2012.npy', allow_pickle = True)
arr = arr[0:100]
np.save('datasets/rainy2012.npy', arr)
'''
fn_rtcor_train = np.load('datasets/rainy2008-2018.npy', allow_pickle = True)[:,0]
fn_rtcor_val = np.load('datasets/rainy2008-2018.npy', allow_pickle = True)[:,1]

filenames_rtcor= np.append(fn_rtcor_train, fn_rtcor_val)
# flatten the array
filenames_rtcor = [item for sublist in filenames_rtcor for item in sublist]
print(len(filenames_rtcor))
# remove duplicate filenames:
filenames_rtcor = list(set(filenames_rtcor))
filenames_rtcor = filenames_rtcor
print(len(filenames_rtcor))
#print(filenames_rtcor)

rtcor2npy(config.dir_rtcor, config.dir_rtcor_prep, overwrite = False, preprocess = True, filenames = filenames_rtcor)
print(len(filenames_rtcor))

