
path_data = '/home/zhiyiwang/datasets/'

path_project  = '/users/zhiyiwang/thesis2022/Large_Sample_Nowcasting_Evaluation/precipitation-nowcasting-using-GANs/'

# Global variables that point to the correct directory
dir_rtcor = path_data + 'RT/'  # path of RT dataset
dir_aart = path_data + 'MFBS/' # path of MFBS dataset

prefix_rtcor = 'RAD_NL25_RAP_5min_'
prefix_aart = 'RAD_NL25_RAC_MFBS_EM_5min_'

### If you train the model with KNMI data, you can ignore the following code ###
dir_rtcor_npy = path_data + 'dataset_radar_np/'
dir_aart_npy = path_data + 'dataset_aart_np/'

dir_prep = 'preprocessed/'
dir_rtcor_prep = path_data + dir_prep + 'rtcor/'
dir_aart_prep = path_data + dir_prep + 'aart/'

dir_labels = path_data + 'rtcor_rain_labels/'
dir_labels_heavy = path_data + 'rtcor_heavy_rain_labels/'

dir_pred = "/home/zhiyiwang/results"
### If you train the model with KNMI data, you can ignore the following code ###