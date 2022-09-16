import numpy as np
from batchcreator import DataGenerator, undo_prep
from model_builder import GAN
import tensorflow as tf
from validation import Evaluator
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotter
from model_builder import GAN, build_generator
import os
from datetime import datetime, timedelta
import random
import cv2
import h5py


kernel_size=3
kernel = -np.ones((kernel_size,kernel_size))
kernel[1,1] = 1

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print('Starting test run')
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_memory_growth(physical_devices[1], True)

def crop_center(img,cropx=440,cropy=440):
    # batch size, sequence, height, width, channels
    # Only change height and width
    _, _, y, x, _ = img.shape
    startx = 20+x//2-(cropx//2)
    starty = 40+y//2-(cropy//2)
    return img[:,:,starty:starty+cropy,startx:startx+cropx:,]

def eventGeneration_training(start_time, obs_time = 4 ,lead_time = 72):
    # Generate event based on starting time point, return a list: [[t-4,...,t-1,t], [t+6,t+12, t+72]]
    # Get the start year, month, day, hour, minute
    year = int(start_time[0:4])
    month = int(start_time[4:6])
    day = int(start_time[6:8])
    hour = int(start_time[8:10])
    minute = int(start_time[10:12])
    #print(datetime(year=year, month=month, day=day, hour=hour, minute=minute))
    times = [(datetime(year, month, day, hour, minute) + timedelta(minutes=30 * (x+1))) for x in range(lead_time)]
    lead = [dt.strftime('%Y%m%d%H%M') for dt in times]
    times = [(datetime(year, month, day, hour, minute) - timedelta(minutes=5 * x)) for x in range(obs_time)]
    obs = [dt.strftime('%Y%m%d%H%M') for dt in times]
    obs.reverse()
    return lead, obs

def make_train_list(arr):
    result = []
    for i in range(len(arr)):
        lead, obs = eventGeneration_training(arr[i], 6, 6)
        result.append([obs, lead])
    return result

def post_processing(img, a, b):
    processed_img = (1 + a * ((img/np.max(img)))**b) * img
    return processed_img

x_length = 6
y_length = 6

# Load model
#GAN
gan = GAN(rnn_type='LSTM', x_length=x_length,
            y_length=y_length, architecture='AENN', relu_alpha=.2,
           l_adv = 0.003, l_rec = 1, g_cycles=3, label_smoothing=0.2
            , norm_method = None, downscale256 = True, rec_with_mae= True,
           r_to_dbz = False, batch_norm = False)

print("load weights start")
gan.load_weights('saved_models/model_zhiyi0615_NL_full_gpd44')
model = gan.generator
print("load weights done")


c = 0
list_ID = ['202002091900']
list_IDs = make_train_list(list_ID)
print(list_IDs)
#print(list_IDs[-1])
print(len(list_IDs))

gen = DataGenerator(list_IDs, batch_size=1, x_seq_size=6,
                    y_seq_size=6, norm_method=None, load_prep=False,
                    downscale256=True, convert_to_dbz=False,
                    y_is_rtcor=True, shuffle=False)

cp_gen = DataGenerator(gen.list_IDs, batch_size=gen.batch_size, x_seq_size=gen.inp_shape[0],
                       y_seq_size=gen.out_shape[0], norm_method=None, load_prep=False,
                       downscale256=False, convert_to_dbz=False,
                       y_is_rtcor=gen.y_is_rtcor, shuffle=False, crop_y=False)

# mask
filename_ref = "/home/zhiyiwang/datasets/RT/2008/01/RAD_NL25_RAP_5min_200801020000.h5"
f_ref = h5py.File(filename_ref)['image1']['image_data']
rain = f_ref[202: 642, 150: 590]
mask = (rain != 65535)
mask = np.array(mask)
mask = mask + 0

for (xs_prep, ys_prep), (_, ys) in tqdm(zip(gen, cp_gen)):
    xs_prep =xs_prep
    ys_pred = model.predict(xs_prep)

    ys_pred = undo_prep(ys_pred, norm_method=None)

    ys = crop_center(ys)


    for y_pred, y_target in zip(ys_pred, ys):

        for i in range(len(y_target)):
            img = y_pred[i].reshape(440,440)
            img = medianBulr(img)
            img = np.multiply(img, mask)

            y_pred[i] = img.reshape(440,440,1)
        y_pred = post_processing(y_pred, 0.67, 0.82)

        plotter.plot_target_pred(y_target, y_pred)
        plt.show()


