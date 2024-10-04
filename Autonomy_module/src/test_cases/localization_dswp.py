from src.global_knowledge import Global_knowledge
from src.UKF import Adaptive_UKF_ARCTAN
import numpy as np
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
from src.configs.constants import *
import matplotlib.pyplot as plt
import traceback
from src.belief_state import Belief_State
whale_belief_up_color = [65/255, 105/255, 225/255]
whale_belief_down_color = [0, 0, 0]

whale_up_color = [200/255, 200/255, 200/255]
whale_down_color = [50/255, 50/255, 50/255]

if __name__ == '__main__':
    try:
        f = open("src/configs/config_Benchmark.json", "r")
        parameters = Global_knowledge()
        parameters.set_attr(json.load(f))
    

        line = "4501;2;14890.718655197781,17782.05097766267;50797.951603055255,43591.526778913314;4.9743249194347925,6.257202387538349;15.000000149991095,15.000000149984787;4;14899.589463298467,17831.721077999224,26835.39697178155,29975.63708166395;50765.57806384846,43590.37049638989,23923.11110908872,25623.154960892367;-1.249648855940048,-0.3067948038455221,1.0341947386796637,2.269899024766434;0.9498830575857526,1.647754536148256,1.1059977871353002,1.297757503044411;False,False,False,False;3,2;1767,3022,2485,2614;2305,3303,3322,2922;1e-07|1e-07,8.834969593133244e-07|9.403063761781904e-06,2.143414139965731e-05|2.9560015365095575e-06,4.7824455181425975e-06|3.4475288964513657e-07"
        st = Belief_State(knowledge = parameters, state_str = line)
        exit()
        onlyfiles = [f for f in listdir(parameters.parsed_whale_data_output) \
            if isfile(join(parameters.parsed_whale_data_output, f)) and 'ground_truth.csv' in f]
        onlyfiles = [filename for fid, filename in enumerate(onlyfiles) if fid not in [43, 183]]
        # dswp_indices = np.random.choice(np.arange(len(onlyfiles)), 1)
        dswp_indices = [1]
        dates = [onlyfiles[i].split('ground_truth')[0] for i in dswp_indices]

        surface_intervals_df = pd.read_csv(parameters.parsed_whale_data_output + dates[0] + 'surface_interval.csv', \
            names=['surface_start', 'surface_stop', 'fluke_camera_aoa'], header=None)
        surface_intervals_df = surface_intervals_df.sort_values(by=['surface_start'])
        gt_surface_interval_scenario = []
        for _, surface_durations_row in surface_intervals_df.iterrows():
            ss = int(surface_durations_row['surface_start']) 
            se = int(surface_durations_row['surface_stop'])
            gt_surface_interval_scenario.append((ss, se))


        acoustic_stop_start_df = pd.read_csv(parameters.parsed_whale_data_output + dates[0] + 'acoustic_end_start.csv', \
            names=['acoustic_silent_start', 'acoustic_silent_end', 'fluke_camera_aoa'], header=None)
        acoustic_stop_start_df = acoustic_stop_start_df.sort_values(by=['acoustic_silent_start'])
        fluke_dir = acoustic_stop_start_df.iloc[0]['fluke_camera_aoa']

        gt_df = pd.read_csv(parameters.parsed_whale_data_output + dates[0] + 'ground_truth.csv', \
            names=['gt_sec', 'gt_lon', 'gt_lat', 'camera_lon', 'camera_lat', 'camera_aoa'], header=None)        
        gt_df = gt_df.groupby(['gt_sec'], as_index=False)\
            .agg({'gt_lon': 'mean', 'gt_lat': 'mean', 'camera_lon': 'mean', 'camera_lat': 'mean', 'camera_aoa': 'mean'})
        gt_df = gt_df.sort_values(by=['gt_sec']) 

        gt_locs = dict(zip(gt_df['gt_sec'], \
                zip(gt_df['gt_lon'], gt_df['gt_lat'])))
        

        def check_interval(value, intervals):
            return any([1 if min_val <= value <= max_val else 0 for (min_val, max_val) in intervals])
    
        if parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
            sensor_xy_df = pd.read_csv(parameters.parsed_whale_data_output + dates[0] +'xy.csv', \
                names=['sensor_sec', 'w_long', 'w_lat', 'long_std', 'lat_std', 'sensor_name'], header=None)
            sensor_xy_df = sensor_xy_df.sort_values(by=['sensor_sec'])
            if parameters.observation_type == 'Acoustic_xy_no_VHF':
                sensor_xy_df = sensor_xy_df[sensor_xy_df['sensor_name'] == 'U']
            loc_observation = dict(zip(sensor_xy_df['sensor_sec'], \
                zip(sensor_xy_df['w_long'], sensor_xy_df['w_lat'], sensor_xy_df['long_std'], sensor_xy_df['lat_std'])))
        
            current_whale_up_col = sensor_xy_df['sensor_sec'].apply(check_interval,     intervals=gt_surface_interval_scenario)
            acoustic_silent_start_ind = sensor_xy_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_start'].values, axis=1)
            acoustic_silent_end_ind = sensor_xy_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_end'].values, axis=1)
        
            extra_obs = dict(zip(sensor_xy_df['sensor_sec'], zip(current_whale_up_col, acoustic_silent_start_ind, acoustic_silent_end_ind)))
                
        else:
            sensor_df = pd.read_csv(parameters.parsed_whale_data_output + dates[0] +'aoa.csv', \
                names=['sensor_sec', 'sensor_lon', 'sensor_lat','aoa', 'aoa1', 'aoa2', 'std_error', 'sensor_name'], header=None)
            sensor_df = sensor_df.sort_values(by=['sensor_sec'])
            if parameters.observation_type == 'Acoustic_AOA_no_VHF':
                sensor_df = sensor_df[sensor_df['sensor_name'] == 'A']
            ta_observation = dict(zip(sensor_df['sensor_sec'], \
                zip(sensor_df['sensor_lon'], sensor_df['sensor_lat'], sensor_df['aoa'], sensor_df['aoa1'], sensor_df['aoa2'], sensor_df['std_error'])))
       
            current_whale_up_col = sensor_df['sensor_sec'].apply(check_interval, intervals=gt_surface_interval_scenario)
            acoustic_silent_start_ind = sensor_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_start'].values, axis=1)
            acoustic_silent_end_ind = sensor_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_end'].values, axis=1)
            extra_obs = dict(zip(sensor_df['sensor_sec'], zip(current_whale_up_col, acoustic_silent_start_ind, acoustic_silent_end_ind)))
    
        init_time = int(gt_df['gt_sec'].values[0]) 
        end_time = int(gt_df['gt_sec'].values[-1]) 
        init_x = gt_locs[init_time][0] # gt_df['gt_lon'].values[0]
        init_y = gt_locs[init_time][1] # gt_df['gt_lat'].values[0]
        next_x, next_y = get_gps_from_start_vel_bearing(init_x, init_y, Whale_speed_mtpm / (5*60), fluke_dir * np.pi / 180)
        # init_loc_w = np.array([init_x, init_y, next_x - init_x, next_y - init_y])
        init_loc_w = np.array([init_x, init_y])
        filter = Adaptive_UKF_ARCTAN(parameters)
        intitial_variance = np.diag([parameters.initial_obs_xy_error[0,0]/10, parameters.initial_obs_xy_error[1,1]/10, \
            parameters.initial_obs_xy_error[0,0]/100, parameters.initial_obs_xy_error[1,1]/100])
        filter.initialize_filter(init_loc_w, intitial_variance)

        wt_xs = []
        wt_ys = []
        w_xs = []
        w_ys = []
        w_ups = []
        w_up_times = []
        s_xs = []
        s_ys = []
        for curr_time in range(init_time, end_time):
            current_whale_up = any([curr_time >= ss and curr_time <= se for (ss,se) in gt_surface_interval_scenario])

            if parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                receiver_error_var = np.array([loc_observation[curr_time][2]**2]) \
                    if curr_time in loc_observation.keys() else None
                current_observed_xy = [np.array([loc_observation[curr_time][0], loc_observation[curr_time][1]]) \
                    if curr_time in loc_observation.keys() else None]
                obs = ObservationClass_whale_TA(current_whale_up=current_whale_up, current_receiver_error = receiver_error_var, \
                    receiver_current_loc = None, \
                        current_observed_AOA_candidate1= None, current_observed_AOA_candidate2=None, \
                            current_observed_xy = current_observed_xy)
            else:
                receiver_error_var = np.array([ta_observation[curr_time][5]**2]) if curr_time in ta_observation.keys() else None
                receiver_error_std = ta_observation[curr_time][5] if curr_time in ta_observation.keys() else None
                receiver_loc = np.array([ta_observation[curr_time][0], ta_observation[curr_time][1]]).reshape(1,2) \
                    if curr_time in ta_observation.keys() else None
                if curr_time in ta_observation.keys():
                    s_xs.append(receiver_loc[0,0])
                    s_ys.append(receiver_loc[0,1])
                received_AOA_candidate1 = np.array([ta_observation[curr_time][3]]) if curr_time in ta_observation.keys() else None
                received_AOA_candidate2 = np.array([ta_observation[curr_time][4]]) if curr_time in ta_observation.keys() else None

                obs = ObservationClass_whale_TA(current_whale_up=current_whale_up, current_receiver_error = receiver_error_var, \
                    receiver_current_loc = receiver_loc, \
                        current_observed_AOA_candidate1= received_AOA_candidate1, current_observed_AOA_candidate2=received_AOA_candidate2, \
                            current_observed_xy = None)
                
            if curr_time != 0:
                extra_obs_ = (extra_obs[curr_time][1], extra_obs[curr_time][2]) \
                    if curr_time in extra_obs.keys() else None
                # if parameters.observation_type not in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy'] and obs.current_whale_up and received_AOA_candidate1 is not None:
                #     ta = ta_observation[curr_time]
                #     ta_m1 =  ta_observation[curr_time-1] if curr_time-1 in ta_observation.keys() else None
                #     ta_p1 =  ta_observation[curr_time+1] if curr_time+1 in ta_observation.keys() else None
                #     print('whale up', ta, ta_m1, ta_p1)
                filter.state_estimation(observation = obs, acoustic_silent_start_end = extra_obs_)

            w_xs.append(filter.hat_x_k[0,0])
            w_ys.append(filter.hat_x_k[1,0])
            w_ups.append(current_whale_up)
            if current_whale_up:
                w_up_times.append(curr_time)

            wt_xs.append(gt_locs[curr_time][0])
            wt_ys.append(gt_locs[curr_time][1])

      
    
            if 1==2 and curr_time > 0 and curr_time % (15*60) == 0:
                dist = [get_distance_from_latLon_to_meter(wt_ys[t], wt_xs[t], w_ys[t], w_xs[t]) for t in range(len(wt_xs))]

                min_x = min([min(wt_xs), min(w_xs)])
                max_x = max([max(wt_xs), max(w_xs)])
                min_y = min([min(wt_ys), min(w_ys)])
                max_y = max([max(wt_ys), max(w_ys)])

                x_dist = round(get_distance_from_latLon_to_meter(min_y, min_x, min_y, max_x)/1000, 3)
                y_dist = round(get_distance_from_latLon_to_meter(min_y, min_x, max_y, min_x)/1000, 3)

                w_cols = np.array([whale_belief_up_color if up ==1 else whale_belief_down_color for up in w_ups])
                wt_cols = np.array([whale_up_color if up ==1 else whale_down_color for up in w_ups])
       
                plt.scatter(w_xs, w_ys, label = 'belief', c = w_cols, s = 10)
                plt.scatter(wt_xs, wt_ys, label = 'gt', c = wt_cols, s = 5)
                plt.legend()
                plt.xlabel(str(x_dist))
                plt.ylabel(str(y_dist))
                plt.show()
                plt.close()
                plt.clf()

                plt.plot(np.arange(len(dist)), dist)
                plt.scatter(w_up_times, np.ones(len(w_up_times)))
                plt.grid()
                plt.show()
                plt.close()
                plt.clf()

        min_x = min([min(wt_xs), min(w_xs)])
        max_x = max([max(wt_xs), max(w_xs)])
        min_y = min([min(wt_ys), min(w_ys)])
        max_y = max([max(wt_ys), max(w_ys)])

        x_dist = round(get_distance_from_latLon_to_meter(min_y, min_x, min_y, max_x)/1000, 3)
        y_dist = round(get_distance_from_latLon_to_meter(min_y, min_x, max_y, min_x)/1000, 3)
        
        w_cols = np.array([whale_belief_up_color if up ==1 else whale_belief_down_color for up in w_ups])
        wt_cols = np.array([whale_up_color if up ==1 else whale_down_color for up in w_ups])

        fig, ax = plt.subplots(2)
        ax[0].scatter(w_xs, w_ys, label = 'belief', c = w_cols, s = 10)
        ax[0].scatter(wt_xs, wt_ys, label = 'gt', c = wt_cols, s = 5)
        # if parameters.observation_type in ['Acoustic_AOA_VHF_AOA', 'Acoustic_AOA_no_VHF']:
        #     plt.scatter(s_xs, s_ys, label = 'sensor')
        ax[0].set_xlabel(str(x_dist))
        ax[0].set_ylabel(str(y_dist))
        
        dist = [get_distance_from_latLon_to_meter(wt_ys[t], wt_xs[t], w_ys[t], w_xs[t]) for t in range(len(wt_xs))]
        ax[1].plot(np.arange(len(dist)), dist)
        ax[1].scatter(w_up_times, np.ones(len(w_up_times)))
        ax[1].grid()
        plt.show()
        plt.close()
        plt.clf()

    except Exception as e:
        print(e)
        full_traceback = traceback.extract_tb(e.__traceback__)
        print(full_traceback)