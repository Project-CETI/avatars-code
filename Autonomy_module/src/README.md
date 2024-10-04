# Setup the environment
python -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"

# To run the fielded experiments with the engineered whale 

## To run an instance of engineered whale experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Nov23.json MA_rollout 1`
- The output will be saved in folder rebuttal2_output_Nov23
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside rebuttal2_output_Nov23

## To run aggregate results for engineered whale experiment

- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Nov23.py`
- Run batch: `bash src/configs/rebuttal_runs/batch_run_script_Nov23.sh`
- The output will be saved in folder rebuttal2_output_Nov23
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Nov23 MA_rollout`

# To run the fielded experiments with sperm whales 

## To run an instance of sperm whales experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Feb24.json MA_rollout 1`
- The output will be saved in folder rebuttal2_output_Feb24
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv rebuttal2_output_Feb24

## To run aggregate results for sperm whales experiment
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Feb24.py` 
- Run batch: `bash src/configs/rebuttal_runs/Feb24_batch_run_script.sh`
- The output will be saved in folder rebuttal2_output_Feb24
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Feb24 MA_rollout`

# To run the ablation study with the DSWP dataset

## To run an instance of ablation study with the DSWP dataset 
- Run script: `python3 src/run_script.py src/configs/config_Benchmark.json MA_rollout 1`
- The output will be saved in folder rebuttal2_output_benchmark
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv rebuttal2_output_benchmark

## To run aggregate results for ablation study with the DSWP dataset 
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_dswp.py`
- Run batch: `bash src/configs/rebuttal_runs/DSWP_batch_run_script_MA_rollout.sh`
- The output will be saved in folder rebuttal2_output_benchmark
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Benchmark MA_rollout`




