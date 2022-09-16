# AENN-Nowcasting (This file is still under improvement)
Extreme-value Neural Networks for Weather Forecasting  

This repository is AENN, a variant of GANs model used for short-term precipitation forecasting. The model aims at predicting precipitation in the next 3 hours based on radar images from the last 30 minutes. Compared with the ordinary AENN model, we model the extreme-value pixels with Generalized Pareto Distribution (GPD), and set a GPD-normalizaiton scheme. Practically, we normalize pixels with high values with the cumulative distribution function of GPD, and normalize other pixels with linear functions. In this way, the model can more easily predict high-value pixels, and thus partly relieve data imbalance.



To understand the code, you can mainly focus on four most important files:

1. [batchcreator.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/batchcreator.py)  This file is used to create batches, which process the raw radar data to input data that are suitable for the model. In this file, you can mainly focus on function 'prep_data' and 'undo_prep'. The inputs of the model are sized at 765 * 700. Since most of areas in the input images are unavailable areas, which are fixed with a calibration value 65535, we set the output images to 440 * 440 in size, which leaves all available areas and cuts most unavailable areas. Of course, you can use other available data and cut the image to any sizes you want (e.g. 256 * 256).
2. [model_builder.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/model_builder.py)  This file contains the main structure of the model. In my thesis, we mainly use 'AENN' structure, so you can ignore code related to 'Tian' structure. The generator of AENN model can be selected from GRU and LSTM. For default setting, the GRU is convGRU, and LSTM is self-attention convLSTM, which is used in my thesis. If you want to change the LSTM units to ordinary convLSTM, or use other types of RNN variants, you can modify it in function 'convRNN_block' in [model_builder.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/model_builder.py).
3. [logger.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/logger.py) This file is mainly used for visualization.
4. [fit_gpd.ipynb](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/fit_gpd.ipynb) This file is used to fit GPD parameters. In practice, we fit the GPD parameters with scipy based on 500 (this number can be modified by users) images int he training set. In oder to improve spatial diversity, we divide each image into N * N pixels, and then fit each pixel with a separate set of GPD parameters. Here, N is decided by users.



If you want to run this code, you first need to change the data paths in [config.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/config.py)

Also, you need to change the paths of GPD files in [batchcreator.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/batchcreator.py)

Then you can run [testrun0401.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/testrun0401.py)  to train the model. In this file, you can change the training settings.

Training Process:  

Training set: 40,000 events selected from rainy events between 2008-2017  [NL_2008-2017.npy](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/datasets/NL_2008-2017.npy)

Validation set: 362 extreme events in 2018  [NL_extreme.npy](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/datasets/NL_extreme.npy)

Testing set: 657 extreme events in the whole available area from 2019 to 2021  [NL_test.npy](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/datasets/NL_test.npy)

Stop running: When the validation reconstruction loss does not improve in 15 epoches.  



