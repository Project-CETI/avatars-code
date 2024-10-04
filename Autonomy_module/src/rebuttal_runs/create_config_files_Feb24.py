import pandas as pd 
import json, copy, os, pickle
import numpy as np
from os import listdir
from os.path import isfile, join
from itertools import combinations
import matplotlib.pyplot as plt

def save_run_id_details(config_obj, number_of_whales, max_num_agents, date_combi):
    rebuttal_output_path = config_obj["base_output_path_prefix"]

    data_min_y = 90
    data_min_x = 180
    data_max_y = -90
    data_max_x = -180

    parsed_whale_data_output = 'Feb24_Dominica_Data/'
    whalefiles = [f for f in listdir(parsed_whale_data_output) \
        if isfile(join(parsed_whale_data_output, f)) and 'ground_truth.csv' in f and any([dt in f for dt in date_combi])]
    print('files:', date_combi,whalefiles)
    first_whale_filename = [whalefile for whalefile in whalefiles if date_combi[0] in whalefile ][0]

    gt_locs = {whalefile: ([], []) for whalefile in whalefiles }
    for wid, whalefile in enumerate(whalefiles):
        gt_df = pd.read_csv(parsed_whale_data_output + whalefile, \
            names=['gt_sec', 'gt_lon', 'gt_lat', 'camera_lon', 'camera_lat', 'camera_aoa'], header=None)
        gt_df = gt_df.groupby(['gt_sec'], as_index=False).agg({'gt_lon': 'mean', 'gt_lat': 'mean', 'camera_lon': 'mean', 'camera_lat': 'mean', 'camera_aoa': 'mean'})
        gt_df = gt_df.sort_values(by=['gt_sec']) 

        init_x = gt_df.iloc[0]['gt_lon']
        init_y = gt_df.iloc[0]['gt_lat']

        gt_locs[whalefile] = (gt_df['gt_lon'].values, gt_df['gt_lat'].values)
        
        print('whalefile:', whalefile, 'gt_lat', gt_df['gt_lat'], ', gt_lon:', gt_df['gt_lon'])
        if config_obj["overlay_GPS"] == True and first_whale_filename in whalefile:
            print('first file:',whalefile)
            data_min_y = min(gt_df['gt_lat'])
            data_min_x = min(gt_df['gt_lon'])
            data_max_y = max(gt_df['gt_lat'])
            data_max_x = max(gt_df['gt_lon'])
        elif config_obj["overlay_GPS"] == False:
            data_min_y = min(data_min_y, min(gt_df['gt_lat']))
            data_min_x = min(data_min_x, min(gt_df['gt_lon']))
            data_max_y = max(data_max_y, max(gt_df['gt_lat']))
            data_max_x = max(data_max_x, max(gt_df['gt_lon']))

    b_xs_all = np.random.uniform(data_min_x, data_max_x, \
        size = (config_obj["average_num_runs"], max_num_agents))
    b_ys_all = np.random.uniform(data_min_y, data_max_y, \
        size = (config_obj["average_num_runs"], max_num_agents))

    print('bxs:', b_xs_all, ', bys:',b_ys_all)
    print('\n')


    random_start_time_index = np.random.uniform( - config_obj["surface_time_mean"] * 60, 0, \
        size = (config_obj["average_num_runs"], config_obj["number_of_whales"]))

    for run_id in range(config_obj["average_num_runs"]):
        # suffix = '_w' + str(number_of_whales)
        suffix = '_'.join(date_combi)
        val_dict_foldername = rebuttal_output_path + 'intial_conditions' + suffix + '/Run_' + str(run_id)
        if not os.path.exists(val_dict_foldername):
            os.makedirs(val_dict_foldername)
            
        val_dict_filename = val_dict_foldername + '/values.csv'

        val_dict = {'initial_agent_xs': list(b_xs_all[run_id]), 'initial_agent_ys': list(b_ys_all[run_id]), \
            'random_start_time_index': list(random_start_time_index[run_id])}

        with open(val_dict_filename, 'w') as val_dict_file:
            json.dump(val_dict, val_dict_file)

    for wid, whalefile in enumerate(whalefiles):
        plt.scatter(gt_locs[whalefile][0], gt_locs[whalefile][1], s = 10)
    plt.scatter(b_xs_all, b_ys_all, s = 1, label = 'agents')
    plt.legend()
    plt.savefig(rebuttal_output_path + 'intial_conditions' + suffix + 'inital_agents.png')
    plt.close()
    
if __name__ == '__main__':
    num_agents = [2,3]
    num_whales = [3,4]
    nagent_nwhales_combo = [[2,3], [2,4], [3,4]]
    tagging_radii = [500, 1000, 1500]
    all_possible_dates = ["2024-02-29", "2024-03-01", "2024-03-02", "2024-03-04"]

    max_num_agents = max(num_agents)
    base_dir = 'src/configs/'
    with open(base_dir + 'config_Dominica_Feb24.json', 'r') as file:
        config_obj = json.load(file)

    if not os.path.exists(base_dir + 'rebuttal_runs/'):
        os.makedirs(base_dir + 'rebuttal_runs/')

    
    rebuttal_output_path = 'output_sperm_whale/'
    print(rebuttal_output_path)
    
    if not os.path.exists(rebuttal_output_path):
        os.makedirs(rebuttal_output_path)

    config_obj["base_output_path_prefix"] = rebuttal_output_path
    batch_run_script = open(base_dir + 'rebuttal_runs/Feb24_batch_run_script.sh', 'w')

    combinations_of_dates = {num_whale: list(combinations(all_possible_dates, num_whale)) for num_whale in num_whales}

    for num_agent in num_agents:
        for num_whale in num_whales:
            # if num_whale <= num_agent:
            #     continue
            if [num_agent, num_whale] not in nagent_nwhales_combo:
                continue
            for date_combi in combinations_of_dates[num_whale]:
                for tagging_radius in tagging_radii:

                    # if num_whale == 4 and num_agent != 2:
                    #     continue

                    config_obj_copy = copy.deepcopy(config_obj)
                    config_obj_copy["number_of_agents"] = num_agent
                    config_obj_copy["number_of_whales"] = num_whale
                    config_obj_copy["tagging_distance"] = tagging_radius
                    config_obj_copy["dates"] = [d for d in date_combi]

                    suffix = '_r' + str(tagging_radius) + '_w' + '_'.join(date_combi) + '_a' + str(num_agent)
                    output_config_filename = base_dir + 'rebuttal_runs/config_Dominica_Feb24' + suffix + '.json'
                    with open(output_config_filename, 'w') as output_config_file:
                        json.dump(config_obj_copy, output_config_file)

                    batch_run_script.write('python3 src/run_script.py ' + output_config_filename + \
                        ' >'+ rebuttal_output_path + 'out' + suffix + ' 2>'+ rebuttal_output_path + 'err' + suffix + ' \n')

    batch_run_script.close()

    for num_whale in num_whales:
        for date_combi in combinations_of_dates[num_whale]:
            save_run_id_details(config_obj, num_whale, min(num_whale, max(num_agents)), date_combi)