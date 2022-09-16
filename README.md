# AENN-nowcasting (This file is still under improvement)
Extreme-value Neural Networks for Weather Forecasting
If you want to run this code, you can change the paths in config.py, and then run testrun0401.py

Training Process:
Trianing set: 40,000 events selected from rainy events between 2008-2017 
Validation set: 362 extreme events in 2018
Testing set: 657 extreme events in the whole available area from 2019 to 2021
Stop running: When the validation reconstruction loss does not improve in 15 epoches.
