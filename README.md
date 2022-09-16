# AENN-nowcasting (This file is still under improvement)
Extreme-value Neural Networks for Weather Forecasting  

This repository is AENN, a variant of GANs model used for short-term precipitation forecasting. The model aims at predicting precipitation in the next 3 hours based on radar images from the last 30 minutes. Compared with the ordinary AENN model, we model the extreme-value pixels with Generalized Pareto Distribution (GPD), and set a GPD-normalizaiton scheme. Practically, we normalize pixels with high values with the cumulative distribution function of GPD, and normalize other pixels with linear functions. In this way, the model can more easily predict high-value pixels, and thus partly relieve data imbalance.

The inputs of the model are sized at 765 * 700. Since most of areas in the input images are unavailable areas, which are fixed with a calibration value 65535, we set the output images to 440 * 440 in size, which leaves all available areas and cuts most unavailable areas. Of course, you can use other available data and cut the image to any sizes you want (e.g. 256 * 256). If you want to change the output size, you can modify function 'crop_center' in [batchcreator.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/batchcreator.py).

If you want to run this code, you first need to change the data paths in [config.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/config.py)

Also, you need to change the paths of GPD files in [batchcreator.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/batchcreator.py)

Then you can run [testrun0401.py](https://github.com/zywang1603/AENN-nowcasting/blob/master/precipitation_forecasting/testrun0401.py)  to train the model. In this file, you can change the training settings.

Training Process:  

Training set: 40,000 events selected from rainy events between 2008-2017  

Validation set: 362 extreme events in 2018  

Testing set: 657 extreme events in the whole available area from 2019 to 2021  

Stop running: When the validation reconstruction loss does not improve in 15 epoches.  



