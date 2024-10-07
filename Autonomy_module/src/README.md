# Sensing code

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

# Autonomy code

# # Setup the environment
- `cd Autonomy_module`
- Create environment: `python -m venv .venv`
- Source environment: `source .venv/bin/activate`
- Add dependencies: `pip3 install -r requirements.txt`
- Add path: `export PYTHONPATH="$PYTHONPATH:$PWD/"`


## To run an instance of engineered whale experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Nov23.json MA_rollout 1`
- The output will be saved in folder output_Engineered_whale
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_Engineered_whale

## To run aggregate results for engineered whale experiment

- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Nov23.py`
- Run batch: `bash src/configs/rebuttal_runs/Nov23_batch_run_script.sh`
- The output will be saved in folder output_Engineered_whale
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Nov23 MA_rollout`


## To run an instance of sperm whales experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Feb24.json MA_rollout 1`
- The output will be saved in folder output_sperm_whale
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_sperm_whale

## To run aggregate results for sperm whales experiment
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Feb24.py` 
- Run batch: `bash src/configs/rebuttal_runs/Feb24_batch_run_script.sh`
- The output will be saved in folder output_sperm_whale
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Feb24 MA_rollout`


## To run an instance of ablation study with the DSWP dataset 
- Run script: `python3 src/run_script.py src/configs/config_Benchmark.json MA_rollout 1`
- The output will be saved in folder output_ablation_dswp
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_ablation_dswp

## To run aggregate results for ablation study with the DSWP dataset 
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_dswp.py`
- Run batch: `bash src/configs/rebuttal_runs/DSWP_batch_run_script_MA_rollout.sh`
- The output will be saved in folder output_ablation_dswp
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Benchmark MA_rollout`




