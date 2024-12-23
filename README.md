# Title of Dataset
[Access this dataset on Dryad](https://doi.org/10.5061/dryad.05qfttfc9)

[Access this dataset on github](https://github.com/Project-CETI/avatars-code/)

This dataset contains files to regenerate results from the paper ``Reinforcement Learning-Based Framework for Whale Rendezvous via Autonomous Sensing Robots''.
The dataset and code repo include data files used to parse acoustic and VHF data and to run the multiagent rollout code to route robots to rendezvous with whales in postprocessing. The repository also includes code and data for computing AOA for VHF signals using the drone's trajectory effectively emulating a synthetic aperture radar. The sensing code files enable the computation of the VHF AOA in real-time.


## Description of the data and file structure
The folder has the following structure contents 

```
avatars-code 
|- Autonomy_module
|  |- dswp_parsed_data_moving_sensors
|  |- Engg_whale_postprocessed_trace
|  |- Feb24_Dominica_Data
|  |- src
|- Sensing_module
|- README.md
```

### Description of the data and file structure for autonomy code
<ul>  
<li>

`avatars-code/Autonomy_module/dswp_parsed_data_moving_sensors` contains the following files, per day and per whale id. 
This folder presents data for the ablation study using the DSWP dataset [1] and [2] presented in our paper (see Results). 
This folder presents data from the DSWP dataset and data augmentation presented in our paper (see Supplementary Materials). </li>

```
dswp_parsed_data_moving_sensors
|- yyyy-mm-dd_####acoustic_end_start.csv
|- yyyy-mm-dd_####aoa.csv
|- yyyy-mm-dd_####ground_truth.csv
|- yyyy-mm-dd_####surface_interval.csv
|- yyyy-mm-dd_####xy.csv
|- whale_track_yyyy-mm-dd_####.png
|- ...
```

<ul>
<li>

`yyyy-mm-dd_####acoustic_end_start.csv`: each line has the acoustic stop time in seconds, acoustic start time in seconds, and the fluke angle in degrees 
</li>

<li>

`yyyy-mm-dd_####aoa.csv`: each line has the time in seconds, sensor longitude, sensor latitude, angle of arrival obtained from the sensor, angle of arrival candidate 1 due to left-right ambiguity, angle of arrival candidate 2 due to left-right ambiguity, standard deviation (in degree) of zero mean gaussian error of the sensor, sensor type ('A' for acoustic and 'V' for VHF).
The fourth column (angle of arrival obtained from the sensor) may be left blank if both candidates for AOA are present.
</li>

<li>

`yyyy-mm-dd_####ground_truth.csv`: each line has the time in seconds, fluke longitude, fluke latitude, camera longitude, camera latitude, fluke angle in degree.
Since the camera location and fluke angle are not used in the ablation study, the last 3 columns are blank.
</li>

<li>

`yyyy-mm-dd_####surface_interval.csv`: each line has the surface interval start time in minutes, surface interval end time in minutes, and the fluke angle in degrees.
If the fluke angle is not obtained, it is set to None.
</li>

<li>

`yyyy-mm-dd_####xy.csv`: each line has the time in seconds, the longitude of a whale with sensing error, the latitude of a whale with sensing error, the standard deviation of the longitude of whale location, the standard deviation of the latitude of whale location, sensor type ('U' for underwater location, 'G' for GPS)
</li>
<li>

`whale_track_yyyy-mm-dd_####.png`: visualization of the whale trace in GPS coordinates. The surface phase is indicated in red, and the underwater phase is indicated in black. The surface interval start and end times are indicated in HH:MM format in the figure. The times in the figure are calculated from the start of the trace.
</li>
</ul>

<li>

`avatars-code/Autonomy_module/Engg_whale_postprocessed_trace` contains the following files per day and per whale id. This folder presents data for the engineered whale experiments presented in our paper (see Results). This folder has a similar structure to `avatars-code/Autonomy_module/dswp_parsed_data_moving_sensors`.
</li>

<li>

`avatars-code/Autonomy_module/Feb24_Dominica_Data` contains the following files per day and per whale id. This folder presents data for the sperm whale experiments presented in our paper (see Results). This folder has a similar structure to `avatars-code/Autonomy_module/dswp_parsed_data_moving_sensors`.
</li>

<li>

`avatars-code/Autonomy_module/src` contains source code to reproduce results in our paper. 

```
avatars-code/Autonomy_module/src
|- requirements.txt
|- configs/
| |- config_Dominica_Nov23.json
| |- config_Dominica_Feb24.json
| |- config_Benchmark.json
| |- constants.py
|- global_knowledge.py
|- system_state.py
|- system_observation.py
|- state_estimation_filter.py
|- UKF.py
|- belief_state.py
|- evaluate_localization_error.py
|- policies/
| |- rollout_policy_science.py
| |- ia_with_commitment_science.py
|- run_script.py
|- rebuttal_runs/
| |- create_config_files_Nov23.py
| |- create_config_files_Feb24.py
| |- create_config_files_dswp.py
| |- evaluate_metric.py
| |- pretty_plot_dswp.py
|- visualization.py
```
<ul><li>

`requirements.txt` contains the python package dependencies
</li><li>

`configs/config_Dominica_Nov23.json` is the default configuration for running an experiment with the engineered whale
</li><li>

`configs/config_Dominica_Feb24.json` is the default configuration for running an experiment with sperm whales
</li><li>

`configs/config_Benchmark.json` is the default configuration for running an experiment for the ablation study with the DSWP dataset 
</li><li>

`configs/constants.py` contains helper functions and global constants
</li><li>

`global_knowledge.py` contains the class for running various configurations, including the number of robots, the number of whales, experiment type, sensor types, and errors. This file reads values from `configs/config_*.json` files.
</li><li>

`system_state.py` reads the senosr data from `avatars-code/Autonomy_module/dswp_parsed_data_moving_sensors, avatars-code/Autonomy_module/Engg_whale_postprocessed_trace`, `avatars-code/Autonomy_module/Feb24_Dominica_Data`
</li><li>

`system_observation.py` parses observation at each time from `system_state.py` and returns observations that are used in `state_estimation_filter.py`
</li><li>

`state_estimation_filter.py` initializes the Unscented Kalman Filter (UKF) with the initial observations of the whale locations. It also initializes the belief with the whale robot's initial locations. The `get_next_estimation` function calls the prediction and measurement steps of UKF for each whale's location update and returns a tuple of whales' updated states to be used in the belief update step of `belief_state.py`. 
</li><li>

`UKF.py` contains the implementation of the Unscented Kalman Filter (UKF) for one whale.
</li><li>

`belief_state.py` contains the class for the belief state and its evolutions as described in the Methods section.
</li><li>

`evaluate_localization_error.py` keeps track of localization errors with ground truth whale locations.
</li><li>

`policies/rollout_policy_science.py` has the logic for our multiagent rollout code to plan movement actions for the robots. See details in the Methods and Supplementary Materials.
</li><li>

`policies/ia_with_commitment_science.py` has the logic for the base policy used in the multiagent rollout code. 
</li><li>

`run_script.py` runs an experiment with a given configuration.
</li><li>

`rebuttal_runs/create_config_files_Nov23.py` is used to create configurations and initial conditions for running aggregated experiments with the engineered whale
</li><li>

`rebuttal_runs/create_config_files_Feb24.py` is used to create configurations and initial conditions for running aggregated experiments with sperm whales
</li><li>

`rebuttal_runs/create_config_files_dswp.py` is used to create configurations and initial conditions for running an aggregated ablation study with the DSWP dataset 
</li><li>

`rebuttal_runs/evaluate_metric.py` parses the output of the aggregated run and prints the output in text and graphic formats. This file uses the functions from `rebuttal_runs/pretty_plot_dswp.py` for plotting output.
</li><li>

`visualization.py` visualizes a given run
</li>
</ul>

</li>
</ul>

### Description of the data and file structure for VHF sensing code

```
|- Datasets # contains raw data files
|- ceti-sar # ROS package to collect trajectory data
|- config_values_for_different_datasets # terminal output for the datafiles
|- scripts # collecting and saving VHF data from SDR using the SoapySDR framework

```

The `Datasets/Info_dataset_*.txt` contains the output from the Matlab terminal when processing each of the dataset files. The description of the dataset files is as follows.

`Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica` contains data files for realtime VHF acquisition using drone flights in Dominica, collected on 21st November 2023.

`Datasets/Dataset_Sci-robotics-Dominica_set_2-11-22-2023-Dominica` contains data files for realtime VHF acquisition using drone flights in Dominica, collected on 22st November 2023.

The structure of the these two datasets is as follows:
```
| - Dataset_Sci-robotics-Dominica_set_<day-mm-dd-yyyy>-Dominica # here <day> is the experiment day 
| | - Flight_<n> # <n> is the nth drone flight on that day
| | | - gps_data
| | | - gps_data_tx
| | | - iq_realtime_subsampled_data
| | | - ori_data
| | | - info.txt
```

Each folder corresponds to data collected during a drone flight. 

`Datasets/Dataset_Sci-robotics_experiments_horn_pond-Oct31-2023`: Contains the data files for experiments at Horn Pond in Boston. The format of this folder is as follows - 

```
| - Dataset_Sci-robotics_experiments_horn_pond-Oct31-2023
| | - Experiment_1_vertical_custom_tag
| | | - gps_data
| | | - gps_data_tx
| | | - iq_realtime_subsampled_data
| | | - ori_data
| | | - info.txt
| | - Experiment_1_vertical_fish_tag
| | - Experiment_2-vertical_custom_tag
| | - Experiment_2_vertical_fish_tag
| | - Experiment_3-vertical_custom_tag
```

The `Datasets/Dataset_Sci-robotics_experiments_horn_pond-Oct31-2023/*_custom_tag` folders correspond to data collected for a custom VHF tag, and the `*_fish_tag` corresponds to data collected for the off-the-shelf VHF fish tag.

`Datasets/Dataset_Sci-robotics_experiments_with_stationary_tag_Oct20-2023` contains data for experiments with a stationary tag collected in Harvard's Ohiri field. The format of this folder is as follows:

```
| - Dataset_Sci-robotics_experiments_with_stationary_tag_Oct20-2023
| | - Location_1
| | | - Experiment_1
| | | | - gps_data
| | | | - gps_data_tx
| | | | - iq_realtime_subsampled_data
| | | | - ori_data
| | | - Experiment_2
| | | - Experiment_3
| | - Location_1_inplace_rotation
| | | - Experiment_1
| | | - Experiment_2
| | | - Experiment_3
| | - Location_2
| | | - Experiment_1
| | | - Experiment_2
| | | - Experiment_3
| | | - Experiment_4
| | - Location_3
| | | - Experiment_1
| | | - Experiment_2
| | | - Experiment_3
| | | - Experiment_4
```

Each experiment here corresponds to data collected during a drone flight.



`gps_data` contains the GPS position data of the UAV when collecting VHF data. Column 1 : timestamp, Column 4: GPS latitude, Column 5: GPS longitude.

`gps_data_tx` contains the GPS position data of the signal transmitter used for calculating the true AOA. Columns are timestamp, position.latitude_deg, position.longitude_deg, position.absolute_altitude_m, position.relative_altitude_m. 

`iq_realtime_subsampled_data` contains the IQ data corresponding to the VHF signals collected at the two RX receivers of the software-defined radio. These are .mat files containing timestamp, complex data for RX 1 and complex data for RX 2.

`ori_data` contains the position and orientation data of the drone when in flight in cartesian coordinates. The columns are timestamp seconds, timestamp-nano seconds, yaw, x, y, z. This data is collected using the ROS package ceti_sar.


<!--The dataset and code repo include data files used to compute AOA for VHF signals using the drone's trajectory effectively emulating a synthetic aperture radar. The code files enable computation of the VHF AOA in real-time.

```
|- ceti-sar # ROS package to collect trajectory data
|- config_values_for_different_datasets # terminal output for the datafiles
|- scripts # collecting and saving VHF data from SDR using the SoapySDR framework
|- Datasets # contains raw data files
  |- Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica
  | |- Flight_1 to Flight_6
  |- Dataset_Sci-robotics-Dominica_set_2-11-22-2023-Dominica
  | |- Flight_1 to Flight_6
  |- Dataset_Sci-robotics_experiments_horn_pond-Oct31-2023
  | |- Experiment 1 to 3 for vertical custom tag
  | |- Experiment 1 to 2 for vertical fish tag
  |- Dataset_Sci-robotics_experiments_with_stationary_tag_Oct20-2023
  | |- Location 1 to Location 3
```

Each data folder used to compute an individual AOA measurement contains the following structure:
```
|- gps_data 
| |- gps_data.csv
|- gps_data_tx
| |- engineered_whale_gps_data.csv
|- iq_realtime_subsampled_data
| |- vhf_drone_payload_data.dat
|- ori_data
  |- displacement.csv
```
-->

Instructions on executing the code are given below.


## How to execute the code
### Setup the environment
- `cd Autonomy_module`
- Create environment: `python -m venv .venv`
- Source environment: `source .venv/bin/activate`
- Add dependencies: `pip3 install -r src/requirements.txt`
- Add path: `export PYTHONPATH="$PYTHONPATH:$PWD/"`


### To run an instance of the engineered whale experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Nov23.json MA_rollout 1`
- The output will be saved in folder `output_Engineered_whale`
- Visualize run: `python3 src/visualization.py output_Engineered_whale/Combined_Dominica_Data_*/Run_-1/MA_rollout/state_log.csv`. The output will be saved in `visualize_output/figure_frames/AVATAR_autonomy_example.gif`

### To run aggregate results for the engineered whale experiment

- Remove previously generated output `rm -r output_Engineered_whale`
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Nov23.py`
- Run batch: `bash src/configs/rebuttal_runs/Nov23_batch_run_script.sh`
- The output will be saved in folder `output_Engineered_whale`
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Nov23 MA_rollout`. The output plots will be saved in `output_Engineered_whale/results_Nov23_*.png`


### To run an instance of the sperm whale experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Feb24.json MA_rollout 1`
- The output will be saved in folder `output_sperm_whale`
- Visualize run: `python3 src/visualization.py output_sperm_whale/Feb24_Dominica_Data_*/Run_-1/MA_rollout/state_log.csv`. The output will be saved in `visualize_output/figure_frames/AVATAR_autonomy_example.gif`

### To run aggregate results for the sperm whale experiment
- Remove previously generated output `rm -r output_sperm_whale`
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Feb24.py` 
- Run batch: `bash src/configs/rebuttal_runs/Feb24_batch_run_script.sh`
- The output will be saved in folder `output_sperm_whale`
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Feb24 MA_rollout`. The output plots will be saved in `output_Engineered_whale/results_Feb24_*.png`


### To run an instance of the ablation study with the DSWP dataset 
- Run script: `python3 src/run_script.py src/configs/config_Benchmark.json MA_rollout 1`
- The output will be saved in folder `output_ablation_dswp`
- Visualize run: `python3 src/visualization.py output_ablation_dswp/Benchmark_*/Run_-1/MA_rollout/state_log.csv`. The output will be saved in `visualize_output/figure_frames/AVATAR_autonomy_example.gif`

### To run aggregate results for the ablation study with the DSWP dataset 
- Remove previously generated output `rm -r output_ablation_dswp`
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_dswp.py`
- Run batch: `bash src/configs/rebuttal_runs/DSWP_batch_run_script_MA_rollout.sh`
- The output will be saved in folder `output_ablation_dswp`
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Benchmark MA_rollout`. The output plots will be saved in `output_ablation_dswp/results_*.png`


### Instructions on executing the Sensing code

Instructions on executing the code.


These matlab files are for regenerating the AOA profiles using the data collected for hardware experiments.

1. Install the BLAS and LAPACK libraries. E.g. on ubuntu
2. sudo apt-get install libblas-dev liblapack-dev
3. Open matlab and install the signal processing toolkit, then run the following commands:

```
Configure the mtimex.c file mex -L/usr/lib -lblas -llapack mtimesx.c
```

4. AOA to files in a dataset can be obtained by running the bulk_process_aoa_drone_sdr_local_position. For example:

```
bulk_process_aoa_drone_sdr_local_position('Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/iq_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/ori_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data_tx')
```

5. For real-time implementation using ROS, use the get_aoa_drone_sdr_local_position_ros.m instead

6. The RPI_SDR_data_collection_code has the python scripts to get data from the SDR. It requires configuring the SoapySDR : https://github.com/pothosware/SoapySDR dependencies.

7. The position and GPS data (for groundtruth) need to be collected using the mavros API and ROS package ceti_sar. The data format is in the corresponding CSV files (ori_data).



<!--1. These matlab files are for regenerating the AOA profiles using the data collected for hardware experiments. 

Install the BLAS and LAPACK libraries.
E.g. on ubuntu 

sudo apt-get install libblas-dev liblapack-dev


Open matlab and run following commands:

a. Configure the mtimex.c file 
mex -L/usr/lib -lblas -llapack mtimesx.c

b. execute code (to process all data), check the config_values_for_different_datasets for the parameter values.
bulk_process_aoa_drone_sdr_local_position('Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/iq_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/ori_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data', 'Datasets/Dataset_Sci-robotics-Dominica_set_1-11-21-2023-Dominica/Location_A/Experiment_1/gps_data_tx')

c. For real-time time implementation using ROS, use the get_aoa_drone_sdr_local_position_ros.m instead

d. The RPI_SDR_data_collection_code has the python scripts to get data from the SDR. It requires configuring the SoapySDR : https://github.com/pothosware/SoapySDR dependencies.

The position and GPS data (for groundtruth) needs to be collected usign the mavros API and ROS package ceti\_sar. The data format is in the corresponding csv files (ori_data).
-->

# References
[1] Gero, S., Milligan, M., Scotia, N., Rinaldi, C., Francis, P., Gordon, J., Carlson, C. A., Steffen, A.,
Tyack, P. L., Evans, P. G., & Whitehead, H. (2014b). Behavior and social structure of the sperm
whales of dominica, west indies. Marine Mammal Science, 30, 905–922.

[2] Gero, S. (2024). The Dominica Sperm Whale Project. http://www.thespermwhaleproject.org.