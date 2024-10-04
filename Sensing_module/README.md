

Instructions on executing the code.
1. These matlab files are for regenerating the AOA profiles using the data collected for hardware experiments. 

Install the BLAS and LAPACK libraries.
E.g. on ubuntu 

sudo apt-get install libblas-dev liblapack-dev


Open matlab and run following commands:

a. Configure the mtimex.c file 
mex -L/usr/lib -lblas -llapack mtimesx.c

b. execute code (to process all data), check the config_values_for_different_datasets for the parameter values.
bulk_process_aoa_drone_sdr_local_position('Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/iq_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/ori_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data_tx')

c. For realtime time implementation using ROS, use the get_aoa_drone_sdr_local_position_ros.m instead

d. The RPI_SDR_data_collection_code has the python scripts to get data from the SDR. It requires configuing the SoapySDR : https://github.com/pothosware/SoapySDR dependencies.

The position and GPS data (for groundtruth) needs to be collected usign the mavros API and ROS package ceti\_sar. The data format is in the correspoding csv files (ori_data).



  

