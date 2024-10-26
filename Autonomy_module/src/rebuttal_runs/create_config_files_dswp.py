import json, os, copy, pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from src.global_knowledge import Global_knowledge
import matplotlib.pyplot as plt

def save_run_id_details(config_obj, number_of_whales, max_num_agents):
    main_folder = "dswp_parsed_data_moving_sensors/"

    onlyfiles = [f for f in listdir(main_folder) if isfile(join(main_folder, f)) and 'ground_truth.csv' in f]

    
    onlyfiles = [filename for fid, filename in enumerate(onlyfiles) if fid not in [43, 183]]
    dswp_indices = np.random.choice(np.arange(len(onlyfiles)), size=(config_obj["average_num_runs"], number_of_whales))

    rebuttal_output_path = config_obj["base_output_path_prefix"]

    random_start_time_index = np.random.uniform( - config_obj["down_time_mean"] * 60, 0, \
        size = (config_obj["average_num_runs"], config_obj["number_of_whales"]))
    for run_id in range(config_obj["average_num_runs"]):
        data_min_y = 90
        data_min_x = 180
        data_max_y = -90
        data_max_x = -180
        whales_filenames = [onlyfiles[i] for i in dswp_indices[run_id]]
        
        for wid in range(number_of_whales):
            
            gt_df = pd.read_csv(main_folder + whales_filenames[wid], \
                names=['gt_sec', 'gt_lon', 'gt_lat', 'camera_lon', 'camera_lat', 'camera_aoa'], header=None)
                
            data_min_y = min(data_min_y, min(gt_df['gt_lat'].values))
            data_min_x = min(data_min_x, min(gt_df['gt_lon'].values))
            data_max_y = max(data_max_y, max(gt_df['gt_lat'].values))
            data_max_x = max(data_max_x, max(gt_df['gt_lon'].values))
            

        b_xs_all = np.random.uniform(data_min_x, data_max_x, size = max_num_agents)
        b_ys_all = np.random.uniform(data_min_y, data_max_y, size = max_num_agents)

        suffix = '_w' + str(number_of_whales)
        val_dict_foldername = rebuttal_output_path + 'intial_conditions' + suffix + '/Run_' + str(run_id)
        if not os.path.exists(val_dict_foldername):
            os.makedirs(val_dict_foldername)
            
        val_dict_filename = val_dict_foldername + '/values.csv'

        val_dict = {'initial_agent_xs': list(b_xs_all), 'initial_agent_ys': list(b_ys_all), \
            'random_start_time_index': list(random_start_time_index[run_id]), \
                'dates': [fn.split('ground_truth')[0] for fn in whales_filenames]}

        with open(val_dict_filename, 'w') as val_dict_file:
            json.dump(val_dict, val_dict_file)

        plt.scatter(b_xs_all, b_ys_all, s = 1, label = 'agents')
        plt.xlim(data_min_x, data_max_x)
        plt.ylim(data_min_y, data_max_y)
        plt.legend()
        plt.savefig(rebuttal_output_path + 'intial_conditions' + suffix + '/Run_' + str(run_id) + '/inital_agents.png')
        plt.close()

    
       

        print('bxs:', b_xs_all, ', bys:',b_ys_all)
    
    


if __name__ == '__main__':
    num_agents = 2
    num_whales = 4
    tagging_radii = [200, 300, 500]
    base_dir = 'src/configs/'
    with open(base_dir + 'config_Benchmark.json', 'r') as file:
        config_obj = json.load(file)

    policy_names = ['MA_rollout','BLE', 'VRP_TW_no_replan_CE']

    
    

    if not os.path.exists(base_dir + 'rebuttal_runs/'):
        os.makedirs(base_dir + 'rebuttal_runs/')

    
    rebuttal_output_path = 'output_ablation_dswp/'
    print(rebuttal_output_path)
    
    
    if not os.path.exists(rebuttal_output_path):
        os.makedirs(rebuttal_output_path)

    config_obj["base_output_path_prefix"] = rebuttal_output_path

    for policy_name in policy_names:

        batch_run_script = open(base_dir + 'rebuttal_runs/DSWP_batch_run_script_'+policy_name+'.sh', 'w')

        

        observation_types = ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA', 'Acoustic_xy_no_VHF','Acoustic_xy_VHF_xy']

        for observation_type in observation_types:
            for tagging_radius in tagging_radii:
                config_obj_copy = copy.deepcopy(config_obj)
                config_obj_copy["number_of_agents"] = num_agents
                config_obj_copy["number_of_whales"] = num_whales
                config_obj_copy["tagging_distance"] = tagging_radius
                config_obj_copy["observation_type"] = observation_type
                

                suffix = '_r' + str(tagging_radius) + '_'+ observation_type + '_w' + str(num_whales) + '_a' + str(num_agents)
                output_config_filename = base_dir + 'rebuttal_runs/config_DSWP' + suffix + '.json'
                with open(output_config_filename, 'w') as output_config_file:
                    json.dump(config_obj_copy, output_config_file)


                batch_run_script.write('python3 src/run_script.py ' + output_config_filename + ' ' + policy_name +\
                    ' >'+ rebuttal_output_path + 'out' + suffix + ' 2>'+ rebuttal_output_path + 'err' + suffix + ' \n')

        batch_run_script.close()

    

    save_run_id_details(config_obj, num_whales, num_agents)
            