# Setup the environment
- Create environment: `python -m venv .venv`
- Source environment: `source .venv/bin/activate`
- Add dependencies: `pip3 install -r requirements.txt`
- Add path: `export PYTHONPATH="$PYTHONPATH:$PWD/"`

# To run the fielded experiments with the engineered whale 

## To run an instance of engineered whale experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Nov23.json MA_rollout 1`
- The output will be saved in folder output_Engineered_whale
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_Engineered_whale

## To run aggregate results for engineered whale experiment

- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Nov23.py`
- Run batch: `bash src/configs/rebuttal_runs/Nov23_batch_run_script.sh`
- The output will be saved in folder output_Engineered_whale
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Nov23 MA_rollout`

# To run the fielded experiments with sperm whales 

## To run an instance of sperm whales experiment
- Run script: `python3 src/run_script.py src/configs/config_Dominica_Feb24.json MA_rollout 1`
- The output will be saved in folder output_sperm_whale
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_sperm_whale

## To run aggregate results for sperm whales experiment
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_Feb24.py` 
- Run batch: `bash src/configs/rebuttal_runs/Feb24_batch_run_script.sh`
- The output will be saved in folder output_sperm_whale
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Feb24 MA_rollout`

# To run the ablation study with the DSWP dataset

## To run an instance of ablation study with the DSWP dataset 
- Run script: `python3 src/run_script.py src/configs/config_Benchmark.json MA_rollout 1`
- The output will be saved in folder output_ablation_dswp
- Visualize run: `src/visualization.py <filename>` filename is the path to state.csv inside output_ablation_dswp

## To run aggregate results for ablation study with the DSWP dataset 
- Generate config files for running: `python3 src/rebuttal_runs/create_config_files_dswp.py`
- Run batch: `bash src/configs/rebuttal_runs/DSWP_batch_run_script_MA_rollout.sh`
- The output will be saved in folder output_ablation_dswp
- Parse aggregate results: `python3 src/rebuttal_runs/evaluate_metric.py Benchmark MA_rollout`




