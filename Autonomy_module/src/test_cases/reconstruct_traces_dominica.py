from src.state_estimation_filter import State_Estimation_Filter
from src.system_state import System_state
from src.system_observation import ObservationClass
from src.evaluate_localization_error import EvalObjectClass

from src.global_knowledge import Global_knowledge
from src.configs.constants import *
import numpy as np
import matplotlib.pyplot as plt
import json, pickle, os, sys
from scipy import interpolate
import matplotlib.colors as mcolors
import matplotlib
import pandas as pd
font = {'size'   : 16}
matplotlib.rc('font', **font)


def test_whales(parameter: Global_knowledge, config):
    fig,ax = plt.subplots(figsize=(24,8), dpi=300)
    folder_ = parameter.base_output_path + 'Run_' + str(-1) + '/no_policy/'
    if not os.path.exists(folder_):
        os.makedirs(folder_)
    gt_orig : System_state = System_state(parameters = parameter, run_id = -1)
    reconstructed_locs = {wid: {'x': [], 'y': []} for wid in range(gt_orig.number_of_whales)}
    css4_colors = mcolors.CSS4_COLORS
    cs = {wid: [] for wid in range(gt_orig.number_of_whales)}
    # reconstructed_gps = {wid: {'x': [], 'y': []} for wid in range(gt_orig.number_of_whales)}
    receiver_loc = {wid: {'x': [], 'y': []} for wid in range(gt_orig.number_of_whales)}
    gt_locs = {wid: {'x': [], 'y': []} for wid in range(gt_orig.number_of_whales)}
    gt_locs_only_summary = {wid: {'x': [], 'y': []} for wid in range(gt_orig.number_of_whales)}

    
    
    # partition_colors =['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
    whale_colors = {0:['indigo', 'royalblue'], 1:['deeppink', 'pink'], 2:['darkgreen', 'lime'], 3: ['orange', 'yellow'], 4:['black', 'lightgrey']}
    dist_all = []
    dist_all_summary = []
    for wid in range(parameter.number_of_whales):
        # gt = gt_orig.copy_state_for_1whale(wid)
        # f = open("src/configs/config_Dominica_Nov23_w"+str(wid)+".json", "r")
        f = open("src/configs/config_Dominica_"+config+".json", "r")
        knowledge = Global_knowledge()
        knowledge.set_attr(json.load(f))
        knowledge.number_of_whales = 1
        knowledge.dates = [knowledge.dates[wid]]

        gt_df = pd.read_csv(knowledge.parsed_whale_data_output + knowledge.dates[0] + 'ground_truth.csv', names=['gt_sec', 'gt_lon', 'gt_lat', 'camera_lon', 'camera_lat', 'camera_aoa'], header=None)
        gt_df = gt_df.groupby(['gt_sec'], as_index=False)\
        .agg({'gt_lon': 'mean', 'gt_lat': 'mean', 'camera_lon': 'mean', 'camera_lat': 'mean', 'camera_aoa': 'mean'})
        gt_df = gt_df.sort_values(by=['gt_sec']) 
        min_sec_in_gt = int(gt_df['gt_sec'].iloc[0])
        gt_df['gt_sec'] = gt_df['gt_sec'] - min_sec_in_gt


        gt : System_state = System_state(parameters = knowledge, run_id = -1)

        gt_locs[wid]['x'].append(gt.gt_whale_sighting[gt.time_in_second][0][0])
        gt_locs[wid]['y'].append(gt.gt_whale_sighting[gt.time_in_second][0][1])
        evalObject = EvalObjectClass(gt = gt)
        raw_obs: ObservationClass = ObservationClass(gt = gt)
        xs = []
        ys = []
        
        dist = [0]
        dist_summary = {0:0}
        localization_filter: State_Estimation_Filter = State_Estimation_Filter(observation = raw_obs)
        xs.append(localization_filter.data_ukf[0]['mean'][-1][0,0])
        ys.append(localization_filter.data_ukf[0]['mean'][-1][1,0])
        # cs[wid].append(css4_colors[whale_colors[wid][localization_filter.data_ukf[0]['whale_up'][-1]]])
        for t in range(gt_df['gt_sec'].values[-1] +1):

            control = Boat_Control(b_theta = np.zeros(parameter.number_of_agents), b_v = np.zeros(parameter.number_of_agents))
            gt.step(control)
            
            if config == 'Nov23':
                if gt.time_in_second not in gt.gt_whale_sighting[0].keys():
                    break
                gt_locs[wid]['x'].append(gt.gt_whale_sighting[0][gt.time_in_second][0])
                gt_locs[wid]['y'].append(gt.gt_whale_sighting[0][gt.time_in_second][1])
            else:
                if gt.time_in_second in gt.gt_whale_sighting[0].keys():
                    gt_locs[wid]['x'].append(gt.gt_whale_sighting[0][gt.time_in_second][0])
                    gt_locs[wid]['y'].append(gt.gt_whale_sighting[0][gt.time_in_second][1])
                if gt.time_in_second in gt_df['gt_sec'].values:
                    gt_locs_only_summary[wid]['x'].append(gt.gt_whale_sighting[0][gt.time_in_second][0])
                    gt_locs_only_summary[wid]['y'].append(gt.gt_whale_sighting[0][gt.time_in_second][1])

            raw_obs.set_observation_t(gt, t + 1)
            observations_x_y_v_theta_up, Pcov = localization_filter.get_next_estimation(observation = raw_obs)

            xs.append(localization_filter.data_ukf[0]['mean'][-1][0,0])
            ys.append(localization_filter.data_ukf[0]['mean'][-1][1,0])

            cs[wid].append(css4_colors[whale_colors[wid][localization_filter.data_ukf[0]['whale_up'][-1]]])

            # if np.isnan(gt.current_whale_ys[0]):
            #     break
            if config == 'Nov23':
                dist.append(get_distance_from_latLon_to_meter(ys[-1], xs[-1], gt.current_whale_ys[0], gt.current_whale_xs[0]))
            else:
                if gt.time_in_second in gt.gt_whale_sighting[0].keys():
                    dist.append(get_distance_from_latLon_to_meter(ys[-1], xs[-1], gt_locs[wid]['y'][-1], gt_locs[wid]['x'][-1]))
                if gt.time_in_second in gt_df['gt_sec'].values:
                    dist_summary[gt.time_in_second] = get_distance_from_latLon_to_meter(ys[-1], xs[-1], gt_locs[wid]['y'][-1], gt_locs[wid]['x'][-1])
            for sensor in gt.receiver_allw_allt_loc[0][t].keys():
                receiver_xy = gt.receiver_allw_allt_loc[0][t][sensor]
            plot_flag = (t == len(gt.whales_allt_loc[0]) - 2)
            evalObject.update_logs(raw_obs, observations_x_y_v_theta_up, Pcov, plot = plot_flag, \
                folder_ = folder_, wid_ = wid, receiver_xy = receiver_xy)
            reconstructed_locs[wid]['x'].append(observations_x_y_v_theta_up[0][0]/ 1000)
            reconstructed_locs[wid]['y'].append(observations_x_y_v_theta_up[0][1]/ 1000)
        
        receiver_loc[wid]['x'] = [x[0] for x in gt.ta_observation[0].values()]
        receiver_loc[wid]['y'] = [x[1] for x in gt.ta_observation[0].values()]
        # plt.scatter(reconstructed_locs[wid]['x'], reconstructed_locs[wid]['y'], s = 5, c = partition_colors[wid], label = 'belief')
        # xs = [x[0,0] for x in localization_filter.data_ukf[0]['mean']]
        # ys = [x[1,0] for x in localization_filter.data_ukf[0]['mean']]
        plt.scatter(xs, ys, s = 5, c = 'black', label = 'belief')
        plt.scatter(xs[0], ys[0], s = 50, c = 'black', label = 'belief')
        plt.scatter(gt_locs[wid]['x'], gt_locs[wid]['y'], s = 1, c = 'red', label = 'true loc')
        plt.scatter(receiver_loc[wid]['x'], receiver_loc[wid]['y'], s = 1, c = 'green', label = 'receiver loc')
        plt.xlabel('Easting (km)')
        plt.ylabel('Northing (km)')
        plt.legend()
        # plt.savefig(folder_ + 'engineered_whale_'+str(wid)+'_Nov23.png')
        plt.savefig(folder_ + config + '_whale_' + str(wid) + '.png')
        print('done whale ', wid)
        plt.close()
        plt.clf()

        dist_mean = np.mean(np.array(dist))
        dist_std = np.std(np.array(dist))
        plt.plot(dist)
        plt.xlabel('Seconds')
        # plt.ylim(0, 10000)
        plt.ylabel('Error (meter)')
        plt.legend()
        plt.title('Mean_error:'+str(round(dist_mean, 2)) + ', std_error:'+str(round(dist_std, 2)))
        plt.grid()
        plt.savefig(folder_ + config + '_whale_' + str(wid) + '_error.png')
        # plt.savefig(folder_ + 'whale_error_'+str(wid)+'_'+config+'.png')
        plt.close()
        plt.clf()

        dist_mean_summary = np.mean(np.array(list(dist_summary.values())))
        dist_std_summary = np.std(np.array(list(dist_summary.values())))
        plt.plot(list(dist_summary.keys()), list(dist_summary.values()))
        plt.scatter(list(dist_summary.keys()), list(dist_summary.values()))
        plt.xlabel('Seconds')
        # plt.ylim(0, 10000)
        plt.ylabel('Error (meter)')
        plt.legend()
        # plt.title('Mean_error:'+str(round(dist_mean_summary, 2)) + ', std_error:'+str(round(dist_std_summary, 2)))
        plt.grid()
        plt.savefig(folder_ + config + '_whale_' + str(wid) + '_error_summary.png')
        # plt.savefig(folder_ + 'whale_error_'+str(wid)+'_'+config+'.png')
        plt.close()
        plt.clf()


        dist_all.extend(dist)
        dist_all_summary.extend(dist_summary)

    fig,ax = plt.subplots(figsize=(24,8), dpi=300)
    for wid in range(parameter.number_of_whales):
        plt.scatter(reconstructed_locs[wid]['x'], reconstructed_locs[wid]['y'], s = 20, c = cs[wid])
        # plt.scatter(reconstructed_locs[wid]['x'][0], reconstructed_locs[wid]['y'][0], s = 30, c = cs[wid][0])
        # plt.text(reconstructed_locs[wid]['x'][0], reconstructed_locs[wid]['y'][0], s = str(wid))
    plt.xlabel('Easting (km)')
    plt.ylabel('Northing (km)')
    # plt.savefig(folder_ + 'combined_engineered_'+str(parameter.number_of_whales)+'whales_Nov23.png')
    plt.savefig(folder_ + 'combined_' + config + '_whale_'+str(parameter.number_of_whales)+'whales_'+config+'.png')
    print('Localization error mean ', np.mean(np.array(dist_all)),', error std ', np.std(np.array(dist_all)))
    print('Fluke Localization error mean ', np.mean(np.array(dist_all_summary)),', error std ', np.std(np.array(dist_all_summary)))

if __name__ == '__main__':
    
    config = sys.argv[1]
    if config not in ['Feb24', 'Nov23']:
        config = 'Feb24'#'Nov23'
    # f = open("src/configs/config_Dominica_Nov23.json", "r")
    f = open("src/configs/config_Dominica_"+config+".json", "r")
    knowledge = Global_knowledge()
    knowledge.set_attr(json.load(f))
    
    test_whales(knowledge, config)
