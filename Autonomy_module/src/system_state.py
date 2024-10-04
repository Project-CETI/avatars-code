from src.global_knowledge import Global_knowledge
import typing as t
import numpy as np
import pandas as pd
import copy, pickle, os, csv
from os import listdir
from os.path import isfile, join
from src.parse_dswp_data import parse_dswp_data
from src.configs.constants import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib as mpl
from scipy import interpolate
import collections, json
from src.UKF import Adaptive_UKF_ARCTAN

AOA_Sensor_Obs = collections.namedtuple('AOA_Sensor_Obs', ['sx', 'sy', 'true_AOA', 'obs_AOA', 'AOA_error_std'])
AOA_Sensor_Obs_candidate = collections.namedtuple('AOA_Sensor_Obs_candidate', ['sx', 'sy', 'true_AOA', 'obs_AOA', 'obs_AOA_candidate1', 'obs_AOA_candidate2','AOA_error_std'])
XY_Sensor_Obs = collections.namedtuple('XY_Sensor_Obs', ['xy', 'xy_cov'])
class System_state:

    def __init__(self, parameters: Global_knowledge, run_id, copy_constructor = False) -> None:
        self.time_in_minute = 0
        self.time_in_second = 0

        self.parameters : Global_knowledge = parameters
        self.gt_surface_interval_scenario: t.Dict[int, t.List[t.Tuple[int, int]]] = {} 
        self.number_of_whales = parameters.number_of_whales
        self.run_id = run_id
        
        self.whales_allt_loc = {} # Fixed at the begining on set for DSWP dataset
        self.whales_allt_up = {} # Fixed at the begining for DSWP dataset
        
        self.initial_agent_xs: np.ndarray = None # for comparison
        self.initial_agent_ys: np.ndarray = None # for comparison
        
        self.current_whale_xs = {} 
        self.current_whale_ys = {} 
        self.current_whale_up = {} 
        self.current_whale_assigned = {} 
        self.current_agent_xs: np.ndarray = None
        self.current_agent_ys: np.ndarray = None

        self.data_min_x : float = None
        self.data_min_y : float = None
        self.data_max_x : float = None
        self.data_max_y : float = None

        # These are set for t=0 initially, at each time step we add new value here         
        self.acoustic_sensors_allt_loc: t.Dict[int, t.Dict[int, t.Tuple[float, float]]] = {} # {time: {sensor_id: (loc_x, loc_y)}}
        self.vhf_sensors_allt_loc: t.Dict[int, t.Dict[int, t.Tuple[float, float]]] = {} # {time: {sensor_id: (loc_x, loc_y)}}
        # self.receiver_allw_allt_loc: t.Dict[int, t.Dict[int, t.Tuple[float, float]]] = {} # {wid: {time: {sensor_id: (loc_x, loc_y) } } }
        self.receiver_allw_allt_loc: t.Dict[int, t.Dict[int, t.Dict[str, AOA_Sensor_Obs]]] = {} # {wid: {time: {sensor_id: AOA_Sensor_Obs } } }
        # self.true_allt_AOA: t.Dict[int, t.Dict[int, float]] = {} # {wid: {time: {sensor_id: AOA}}}
        
        if copy_constructor:
            return

        
        self.file_names: t.List[int] = []
        self.data_from_dominica()

    def copy_state_for_1whale(self, wid):

        obj = System_state(parameters = self.parameters, run_id = self.run_id, copy_constructor = True)
        
        obj.gt_surface_interval_scenario = {0: self.gt_surface_interval_scenario[wid]} 
        obj.number_of_whales = 1
        
        obj.whales_allt_loc = {0: self.whales_allt_loc[wid]} # Fixed at the begining on set for DSWP dataset
        obj.whales_allt_up = {0: self.whales_allt_up[wid]} # Fixed at the begining for DSWP dataset
        
        obj.initial_agent_xs = self.initial_agent_xs # for comparison
        obj.initial_agent_ys = self.initial_agent_ys # for comparison
        
        obj.current_whale_xs = {0: self.current_whale_xs[wid]} 
        obj.current_whale_ys = {0: self.current_whale_ys[wid]} 
        obj.current_whale_up = {0: self.current_whale_up[wid]}
        obj.current_whale_assigned = {0: self.current_whale_assigned[wid]}
        
        obj.current_agent_xs = self.current_agent_xs
        obj.current_agent_ys = self.current_agent_ys

        obj.data_min_x = self.data_min_x
        obj.data_min_y = self.data_min_y
        obj.data_max_x = self.data_max_x
        obj.data_max_y = self.data_max_y

        obj.data_min_x_wid = [x for x in self.data_min_x_wid]
        obj.data_min_y_wid = [x for x in self.data_min_y_wid]
        obj.data_max_x_wid = [x for x in self.data_max_x_wid]
        obj.data_max_y_wid = [x for x in self.data_max_y_wid]


        # These are set for t=0 initially, at each time step we add new value here         
        obj.acoustic_sensors_allt_loc = self.acoustic_sensors_allt_loc 
        obj.vhf_sensors_allt_loc = self.vhf_sensors_allt_loc
        obj.receiver_allw_allt_loc = {0: self.receiver_allw_allt_loc[wid]}
        obj.ta_observation = {0: self.ta_observation[wid]}
        obj.gt_whale_sighting = {0: self.gt_whale_sighting[wid]}
        obj.visible_xy = {0: self.visible_xy[wid]}
        # obj.true_allt_AOA = {0: self.true_allt_AOA[wid]}

        obj.file_names = [wid]

        obj.extra_info_from_experiment = {}
        for element in self.extra_info_from_experiment.keys():
            obj.extra_info_from_experiment[element] = {0: self.extra_info_from_experiment[element][wid]}

        obj.gt_surface_interval_scenario = {0: self.gt_surface_interval_scenario[wid]}

        obj.extra_obs = {0: self.extra_obs[wid]} 
        return obj

    
    
    
    def data_from_dominica(self, plot = False):
        self.gt_surface_interval_scenario = {wid : [] for wid in range(self.number_of_whales)}
        
        self.file_names = np.arange(self.number_of_whales)

        if self.parameters.experiment_type == 'Benchmark_Shane_Data':
            suffix = '_w' + str(self.parameters.number_of_whales)
        else:
            suffix = '_'.join(self.parameters.dates)
            dates = self.parameters.dates[: self.number_of_whales]
        val_dict_filename = self.parameters.base_output_path_prefix + 'intial_conditions' + suffix + '/Run_' + str(self.run_id) + '/values.csv'
        
        if isfile(val_dict_filename):
            with open(val_dict_filename, 'r') as val_dict_file:
                val_dict = json.load(val_dict_file)
                if "random_start_time_index" in val_dict.keys() and self.parameters.number_of_whales <= len(val_dict["random_start_time_index"]):
                    random_start_time_index = np.array(val_dict["random_start_time_index"][:self.parameters.number_of_whales]).astype(int)

                if "initial_agent_xs" in val_dict.keys() and self.parameters.number_of_agents <= len(val_dict["initial_agent_xs"]) and \
                    "initial_agent_ys" in val_dict.keys() and self.parameters.number_of_agents <= len(val_dict["initial_agent_ys"]):
                    self.initial_agent_xs = np.array(val_dict["initial_agent_xs"][:self.parameters.number_of_agents])
                    self.initial_agent_ys = np.array(val_dict["initial_agent_ys"][:self.parameters.number_of_agents])
                if self.parameters.experiment_type == 'Benchmark_Shane_Data' and \
                    "dates" in val_dict.keys() and self.parameters.number_of_whales <= len(val_dict["dates"]): 
                    self.parameters.dates = val_dict["dates"]
                    dates = self.parameters.dates[: self.number_of_whales]
        else:
            if self.number_of_whales == 1:
                random_start_time_index = np.zeros(1).astype(int)
            else:
                random_start_time_index = np.random.uniform(-self.parameters.surface_time_mean * 60, 0, \
                size = self.number_of_whales).astype(int)
            
            if self.parameters.experiment_type == 'Benchmark_Shane_Data':
                onlyfiles = [f for f in listdir(self.parameters.parsed_whale_data_output) \
                    if isfile(join(self.parameters.parsed_whale_data_output, f)) and 'ground_truth.csv' in f]
                onlyfiles = [filename for fid, filename in enumerate(onlyfiles) if fid not in [43, 183]]
                dswp_indices = np.random.choice(np.arange(len(onlyfiles)), size=self.number_of_whales)

                dates = [onlyfiles[i].split('ground_truth')[0] for i in dswp_indices]


        for wid in range(self.number_of_whales):
            surface_intervals_df = pd.read_csv(self.parameters.parsed_whale_data_output + dates[wid] + 'surface_interval.csv', \
                names=['surface_start', 'surface_stop', 'fluke_camera_aoa'], header=None)
            surface_intervals_df['surface_start'] = surface_intervals_df['surface_start'] #- min_sec_in_gt
            surface_intervals_df['surface_stop'] = surface_intervals_df['surface_stop'] #- min_sec_in_gt
            surface_intervals_df = surface_intervals_df.sort_values(by=['surface_start'])

            for _, surface_durations_row in surface_intervals_df.iterrows():
                ss = int(surface_durations_row['surface_start']) #+ random_start_time_index[wid]
                se = int(surface_durations_row['surface_stop']) #+ random_start_time_index[wid]
                self.gt_surface_interval_scenario[wid].append((ss, se))

        if not isfile(val_dict_filename):
            order_index = sorted(np.arange(self.number_of_whales), key=lambda wid: len(self.gt_surface_interval_scenario[wid]))
            random_start_time_index = random_start_time_index[order_index]

        random_start_time_index = random_start_time_index - max(random_start_time_index)

        self.visible_xy = {wid: None for wid in range(self.parameters.number_of_whales)}

        
        
        if self.parameters.experiment_type == 'Combined_Dominica_Data':
            surface_time_samples = []
            bottom_time_samples = []
            for wid in range(self.number_of_whales):
                se_old = None
                for (ss, se) in self.gt_surface_interval_scenario[wid]:
                    if ss is not None and se is not None:
                        surface_time_samples.append((se - ss)) # / self.parameters.observations_per_minute)
                    if se_old is not None and ss is not None:
                        bottom_time_samples.append((ss - 1 - se_old)) # / self.parameters.observations_per_minute)
                    se_old = se
            self.parameters.surface_time_mean = int(np.mean(surface_time_samples))
            self.parameters.surface_time_var = int(np.std(surface_time_samples))
            self.parameters.down_time_mean = int(np.mean(bottom_time_samples))
            self.parameters.down_time_var = int(np.std(bottom_time_samples))
        else:
            self.parameters.surface_time_mean *= 60
            self.parameters.surface_time_var *= 60
            self.parameters.down_time_mean *= 60
            self.parameters.down_time_var *= 60
    
        min_traj_len = np.iinfo('i').max
        max_traj_len = np.iinfo('i').min
        
        self.whales_allt_loc = {wid: [] for wid in range(self.number_of_whales)}
        self.whales_allt_up = {wid: [] for wid in range(self.number_of_whales)}
        self.receiver_allw_allt_loc: t.Dict[int, t.Dict[int, t.Dict[str, AOA_Sensor_Obs]]] = {wid: {} for wid in range(self.number_of_whales)}

       

        

        # suffix = '_w' + str(self.parameters.number_of_whales)
       
        self.gt_whale_sighting = {wid: {} for wid in range(self.number_of_whales)}

        self.ta_observation = {wid: {} for wid in range(self.number_of_whales)}
        self.loc_observation = {wid: {} for wid in range(self.number_of_whales)}
        self.extra_obs = {wid: {} for wid in range(self.number_of_whales)}

        self.data_min_y = 90
        self.data_min_x = 180
        self.data_max_y = -90
        self.data_max_x = -180
        self.data_min_y_wid = [90 for _ in range(self.parameters.number_of_whales)]
        self.data_min_x_wid = [180 for _ in range(self.parameters.number_of_whales)]
        self.data_max_y_wid = [-90 for _ in range(self.parameters.number_of_whales)]
        self.data_max_x_wid = [-180 for _ in range(self.parameters.number_of_whales)]

        
        self.extra_info_from_experiment = {'initial_observation': {wid: [] for wid in range(self.number_of_whales)}, \
            'initial_observation_cov': {wid: [] for wid in range(self.number_of_whales)}}

        for wid in range(self.number_of_whales):

            
            gt_df = pd.read_csv(self.parameters.parsed_whale_data_output + dates[wid] + 'ground_truth.csv', \
                names=['gt_sec', 'gt_lon', 'gt_lat', 'camera_lon', 'camera_lat', 'camera_aoa'], header=None)
                
            gt_df = gt_df.groupby(['gt_sec'], as_index=False)\
                .agg({'gt_lon': 'mean', 'gt_lat': 'mean', 'camera_lon': 'mean', 'camera_lat': 'mean', 'camera_aoa': 'mean'})
            gt_df = gt_df.sort_values(by=['gt_sec']) 
            min_sec_in_gt = int(gt_df['gt_sec'].iloc[0])
            if self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                sensor_xy_df = pd.read_csv(self.parameters.parsed_whale_data_output + dates[wid] +'xy.csv', \
                    names=['sensor_sec', 'w_long', 'w_lat', 'long_std', 'lat_std', 'sensor_name'], header=None)
                # print(sensor_xy_df['w_lat'])
                # print(sensor_xy_df['w_long'])
                sensor_xy_df = sensor_xy_df.sort_values(by=['sensor_sec'])
                if self.parameters.observation_type == 'Acoustic_xy_no_VHF':
                    sensor_xy_df = sensor_xy_df[sensor_xy_df['sensor_name'] == 'U']
                sensor_xy_df = sensor_xy_df[(sensor_xy_df['sensor_sec'] >= min_sec_in_gt) & sensor_xy_df['sensor_sec'] <= min_sec_in_gt + self.parameters.n_horizon_for_state_estimation*60]
            else:
                sensor_df = pd.read_csv(self.parameters.parsed_whale_data_output + dates[wid] +'aoa.csv', \
                    names=['sensor_sec', 'sensor_lon', 'sensor_lat','aoa', 'aoa1', 'aoa2', 'std_error', 'sensor_name'], header=None)
                sensor_df = sensor_df.sort_values(by=['sensor_sec'])
                sensor_df = sensor_df[sensor_df['sensor_sec'] >= min_sec_in_gt]
                if self.parameters.experiment_type == 'Benchmark_Shane_Data' and \
                    self.parameters.observation_type == 'Acoustic_AOA_no_VHF':
                    sensor_df = sensor_df[sensor_df['sensor_name'] == 'A']
                if self.parameters.experiment_type == 'Benchmark_Shane_Data':
                    sensor_df = sensor_df[sensor_df['sensor_sec'] <= min_sec_in_gt + self.parameters.n_horizon_for_state_estimation*60]

            if self.parameters.experiment_type == 'Benchmark_Shane_Data':
                gt_df = gt_df[gt_df['gt_sec'] <= min_sec_in_gt + self.parameters.n_horizon_for_state_estimation*60]
            

            
            
            gt_df['gt_sec'] = gt_df['gt_sec'] - min_sec_in_gt + random_start_time_index[wid]
            if self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                sensor_xy_df['sensor_sec'] = sensor_xy_df['sensor_sec'] - min_sec_in_gt + random_start_time_index[wid]
                min_traj_len = min(min_traj_len, max(sensor_xy_df['sensor_sec']))
                max_traj_len = max(max_traj_len, max(sensor_xy_df['sensor_sec']))
                self.data_min_y_wid[wid] = min(sensor_xy_df['w_lat'])
                self.data_min_x_wid[wid] = min(sensor_xy_df['w_long'])
                self.data_max_y_wid[wid] = max(sensor_xy_df['w_lat'])
                self.data_max_x_wid[wid] = max(sensor_xy_df['w_long'])
            else:
                sensor_df['sensor_sec'] = sensor_df['sensor_sec'] - min_sec_in_gt + random_start_time_index[wid]

                min_traj_len = min(min_traj_len, max(sensor_df['sensor_sec']))
                max_traj_len = max(max_traj_len, max(sensor_df['sensor_sec']))
            
                self.data_min_y_wid[wid] = min(sensor_df['sensor_lat'])
                self.data_min_x_wid[wid] = min(sensor_df['sensor_lon'])
                self.data_max_y_wid[wid] = max(sensor_df['sensor_lat'])
                self.data_max_x_wid[wid] = max(sensor_df['sensor_lon'])

            self.data_min_x = min(self.data_min_x, self.data_min_x_wid[wid])
            self.data_max_x = max(self.data_max_x, self.data_max_x_wid[wid])
            self.data_min_y = min(self.data_min_y, self.data_min_y_wid[wid])
            self.data_max_y = max(self.data_max_y, self.data_max_y_wid[wid])


            
            
            acoustic_stop_start_df = pd.read_csv(self.parameters.parsed_whale_data_output + dates[wid] + 'acoustic_end_start.csv', \
                names=['acoustic_silent_start', 'acoustic_silent_end', 'fluke_camera_aoa'], header=None)
            acoustic_stop_start_df['acoustic_silent_start'] = acoustic_stop_start_df['acoustic_silent_start'] - min_sec_in_gt + random_start_time_index[wid]
            acoustic_stop_start_df['acoustic_silent_end'] = acoustic_stop_start_df['acoustic_silent_end'] - min_sec_in_gt + random_start_time_index[wid]
            acoustic_stop_start_df = acoustic_stop_start_df.sort_values(by=['acoustic_silent_start'])
            fluke_dir = acoustic_stop_start_df.iloc[0]['fluke_camera_aoa']

            for sind, ss_se in enumerate(self.gt_surface_interval_scenario[wid]):
                self.gt_surface_interval_scenario[wid][sind] = \
                    (ss_se[0] - min_sec_in_gt + random_start_time_index[wid], ss_se[1] - min_sec_in_gt + random_start_time_index[wid])

            init_loc_w = None
            init_time = None
            for _, gt_row in gt_df.iterrows():
                self.gt_whale_sighting[wid][gt_row['gt_sec']] = (gt_row['gt_lon'], gt_row['gt_lat'])

                if self.parameters.experiment_type == 'Feb24_Dominica_Data':

                    ind = [i for i, ss_se in enumerate(self.gt_surface_interval_scenario[wid]) \
                        if ss_se[0] <= gt_row['gt_sec'] and ss_se[1] >= gt_row['gt_sec']]

                    if len(ind):
                        ss = self.gt_surface_interval_scenario[wid][ind[0]][0]
                        se = self.gt_surface_interval_scenario[wid][ind[0]][1]
                    
                        for t_ in range(ss, se + 1):
                            self.gt_whale_sighting[wid][t_] = (gt_row['gt_lon'], gt_row['gt_lat'])
                            

                if init_loc_w is None:
                    init_time = int(gt_row['gt_sec']) #+ random_start_time_index[wid]

                    next_loc = get_gps_from_start_vel_bearing(gt_row['gt_lon'], gt_row['gt_lat'], Whale_speed_mtpm / (5*60), fluke_dir * np.pi / 180)

                    init_loc_w = np.array([gt_row['gt_lon'], gt_row['gt_lat'], next_loc[0] - gt_row['gt_lon'], next_loc[1] - gt_row['gt_lat']])

            def check_interval(value, intervals):
                return any([1 if min_val <= value <= max_val else 0 for (min_val, max_val) in intervals])
            if self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                # print(len(sensor_xy_df))
                
                current_whale_up_col = sensor_xy_df['sensor_sec'].apply(check_interval, intervals=self.gt_surface_interval_scenario[wid])
                acoustic_silent_start_ind = sensor_xy_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_start'].values, axis=1)
                acoustic_silent_end_ind = sensor_xy_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_end'].values, axis=1)
                self.extra_obs[wid] = dict(zip(sensor_xy_df['sensor_sec'], zip(current_whale_up_col, acoustic_silent_start_ind, acoustic_silent_end_ind)))
                # print([k for k,v in self.extra_obs[wid].items() if v[0]==0])
                self.loc_observation[wid] = dict(zip(sensor_xy_df['sensor_sec'], \
                    zip(sensor_xy_df['w_long'], sensor_xy_df['w_lat'], sensor_xy_df['long_std'], sensor_xy_df['lat_std'])))
                
                
            else:

                current_whale_up_col = sensor_df['sensor_sec'].apply(check_interval, intervals=self.gt_surface_interval_scenario[wid])
                acoustic_silent_start_ind = sensor_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_start'].values, axis=1)
                acoustic_silent_end_ind = sensor_df.apply(lambda row: row['sensor_sec'] in acoustic_stop_start_df['acoustic_silent_end'].values, axis=1)
                self.extra_obs[wid] = dict(zip(sensor_df['sensor_sec'], zip(current_whale_up_col, acoustic_silent_start_ind, acoustic_silent_end_ind)))
                # print([k for k,v in self.extra_obs[wid].items() if v[0]==0])
                self.ta_observation[wid] = dict(zip(sensor_df['sensor_sec'], \
                        zip(sensor_df['sensor_lon'], sensor_df['sensor_lat'], sensor_df['aoa'], sensor_df['aoa1'], sensor_df['aoa2'], sensor_df['std_error'])))

                # for _, sensor_row in sensor_df.iterrows():
                #     if sensor_row['aoa'] is None or np.isnan(sensor_row['aoa']):
                #         continue
                #     if sensor_row['std_error'] is None or np.isnan(sensor_row['std_error']):
                #         sensor_row['std_error'] = self.parameters.Acoustic_AOA_obs_error_std_degree
                #     obs_sec = int(sensor_row['sensor_sec']) #+ random_start_time_index[wid]
                    
                #     current_whale_up = any([ss_se[0] <= sensor_row['sensor_sec'] and sensor_row['sensor_sec'] <= ss_se[1] \
                #         for ss_se in self.gt_surface_interval_scenario[wid]])
                #     acoustic_silent_start_ind = any([row['acoustic_silent_start'] == sensor_row['sensor_sec'] for _, row in acoustic_stop_start_df.iterrows()])
                #     acoustic_silent_end_ind = any([row['acoustic_silent_end'] == sensor_row['sensor_sec'] for _, row in acoustic_stop_start_df.iterrows()])
                #     self.extra_obs[wid][obs_sec] \
                #         = (current_whale_up, acoustic_silent_start_ind, acoustic_silent_end_ind)
                #     self.ta_observation[wid][obs_sec] \
                #         = (sensor_row['sensor_lon'], sensor_row['sensor_lat'],sensor_row['aoa'], sensor_row['aoa1'], sensor_row['aoa2'], sensor_row['std_error'])

            filter = Adaptive_UKF_ARCTAN(self.parameters)
            intitial_variance = np.diag([self.parameters.initial_obs_xy_error[0,0], self.parameters.initial_obs_xy_error[1,1], \
                self.parameters.initial_obs_xy_error[0,0]/10, self.parameters.initial_obs_xy_error[1,1]/10])
            # intitial_variance = np.diag([self.parameters.initial_obs_xy_error[0,0]] * 4)
            filter.initialize_filter(init_loc_w, intitial_variance)

            for prev_time in range(init_time, 1):
                current_whale_up = any([prev_time >= ss and prev_time <= se for (ss,se) in self.gt_surface_interval_scenario[wid]])
                
                if self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                    receiver_error_var = np.array([self.loc_observation[wid][prev_time][2]**2]) \
                        if prev_time in self.loc_observation[wid].keys() else None
                    current_observed_xy = [np.array([self.loc_observation[wid][prev_time][0], self.loc_observation[wid][prev_time][1]]) \
                        if prev_time in self.loc_observation[wid].keys() else None]
                    obs = ObservationClass_whale_TA(current_whale_up=current_whale_up, current_receiver_error = receiver_error_var, \
                        receiver_current_loc = None, \
                            current_observed_AOA_candidate1= None, current_observed_AOA_candidate2=None, \
                                current_observed_xy = current_observed_xy)
                else:
                    receiver_error_var = np.array([self.ta_observation[wid][prev_time][5]**2]) if prev_time in self.ta_observation[wid].keys() else None
                    receiver_error_std = self.ta_observation[wid][prev_time][5] if prev_time in self.ta_observation[wid].keys() else None
                    receiver_loc = np.array([self.ta_observation[wid][prev_time][0], self.ta_observation[wid][prev_time][1]]).reshape(1,2) \
                        if prev_time in self.ta_observation[wid].keys() else None

                    received_AOA_candidate1 = np.array([self.ta_observation[wid][prev_time][3]]) if prev_time in self.ta_observation[wid].keys() else None
                    received_AOA_candidate2 = np.array([self.ta_observation[wid][prev_time][4]]) if prev_time in self.ta_observation[wid].keys() else None

                    # received_AOA = np.array([self.ta_observation[wid][prev_time][2]]) if prev_time in self.ta_observation[wid].keys() else None
                    obs = ObservationClass_whale_TA(current_whale_up=current_whale_up, current_receiver_error = receiver_error_var, \
                        receiver_current_loc = receiver_loc, \
                            current_observed_AOA_candidate1= received_AOA_candidate1, current_observed_AOA_candidate2=received_AOA_candidate2, \
                                current_observed_xy = None)
                
                # prev_hat_x = np.copy(filter.hat_x_k)
                if prev_time != 0:
                    extra_obs = (self.extra_obs[wid][prev_time][1], self.extra_obs[wid][prev_time][2]) \
                        if prev_time in self.extra_obs[wid].keys() else None
                    filter.state_estimation(observation = obs, acoustic_silent_start_end = extra_obs)
        
                if any(np.isnan(filter.hat_x_k)):
                    print('here')

            # if self.parameters.experiment_type == 'Benchmark_Shane_Data':
            #     gt_row_this = gt_df.loc[gt_df['gt_sec'] == 0]
            #     init_loc_w_2 = np.array([gt_row_this['gt_lon'].values[0], gt_row_this['gt_lat'].values[0]])  
            #     filter = Adaptive_UKF_ARCTAN(self.parameters)
            #     intitial_variance = np.diag([self.parameters.initial_obs_xy_error[0,0], self.parameters.initial_obs_xy_error[1,1], \
            #     	self.parameters.initial_obs_xy_error[0,0]/10, self.parameters.initial_obs_xy_error[1,1]/10])
            #     filter.initialize_filter(init_loc_w_2, intitial_variance)

            
            self.extra_info_from_experiment['initial_observation'][wid] = filter.hat_x_k.reshape(4)
            self.extra_info_from_experiment['initial_observation_cov'][wid] = filter.P_k

            self.current_whale_up[wid] = current_whale_up
            self.current_whale_xs[wid] = filter.hat_x_k[0, 0]
            self.current_whale_ys[wid] = filter.hat_x_k[1, 0]


            # self.receiver_allw_allt_loc[wid] = {0: {'ta_' + str(wid): AOA_Sensor_Obs(sx = receiver_loc[0, 0] if receiver_loc is not None else None, \
            #     sy = receiver_loc[0, 1] if receiver_loc is not None else None, \
            #         true_AOA = None, obs_AOA = received_AOA, AOA_error_std = receiver_error_std)}}

            if self.parameters.experiment_type != 'Benchmark_Shane_Data' or self.parameters.observation_type not in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                self.receiver_allw_allt_loc[wid] = {0: {'ta_' + str(wid): AOA_Sensor_Obs_candidate(sx = receiver_loc[0, 0] if receiver_loc is not None else None, \
                    sy = receiver_loc[0, 1] if receiver_loc is not None else None, \
                        true_AOA = None, obs_AOA = None, \
                            obs_AOA_candidate1 = received_AOA_candidate1, obs_AOA_candidate2 = received_AOA_candidate2, \
                                AOA_error_std = receiver_error_std)}}
            else:
                self.receiver_allw_allt_loc[wid] = {0: {'b_' + str(wid): XY_Sensor_Obs(\
                    xy = (current_observed_xy if current_observed_xy is not None else None), \
                        xy_cov = receiver_error_var) } }

        
        if not isfile(val_dict_filename):
            if self.parameters.overlay_GPS:
                self.initial_agent_xs = np.array([np.random.uniform(self.data_min_x_wid[0], self.data_max_x_wid[0]) for bid in range(self.parameters.number_of_agents)])
                self.initial_agent_ys = np.array([np.random.uniform(self.data_min_y_wid[0], self.data_max_y_wid[0]) for bid in range(self.parameters.number_of_agents)])
            else:   
                self.initial_agent_xs = np.array([np.random.uniform(self.data_min_x, self.data_max_x) for bid in range(self.parameters.number_of_agents)])
                self.initial_agent_ys = np.array([np.random.uniform(self.data_min_y, self.data_max_y) for bid in range(self.parameters.number_of_agents)])
            

        # suffix = '_w' + str(self.parameters.number_of_whales)
        # suffix = '_'.join(self.parameters.dates)
        # val_dict_filename = self.parameters.base_output_path_prefix + 'intial_conditions' + suffix + '/Run_' + str(self.run_id) + '/values.csv'
        # if isfile(val_dict_filename):
        #     with open(val_dict_filename, 'r') as val_dict_file:
        #         val_dict = json.load(val_dict_file)
        #         if "initial_agent_xs" in val_dict.keys() and self.parameters.number_of_agents <= len(val_dict["initial_agent_xs"]) and \
        #             "initial_agent_ys" in val_dict.keys() and self.parameters.number_of_agents <= len(val_dict["initial_agent_ys"]):
        #             self.initial_agent_xs = np.array(val_dict["initial_agent_xs"][:self.parameters.number_of_agents])
        #             self.initial_agent_ys = np.array(val_dict["initial_agent_ys"][:self.parameters.number_of_agents])
                
        
        self.current_agent_xs = np.copy(self.initial_agent_xs)
        self.current_agent_ys = np.copy(self.initial_agent_ys)

        for wid in range(self.number_of_whales): 
            self.current_whale_assigned[wid] = False
            # next_surface_end_time = [ se \
            #     for (ss,se) in self.gt_surface_interval_scenario[wid] if self.time_in_second >= ss and self.time_in_second <= se]
            # if self.current_whale_up[wid] and len(next_surface_end_time) > 0:
            #     next_surface_end_time = next_surface_end_time[0]
            
            # if self.parameters.overlay_GPS and wid != 0:
            #     temp_whale_loc_x = self.current_whale_xs[wid] - self.data_min_x_wid[wid] + self.data_min_x_wid[0]
            #     temp_whale_loc_y = self.current_whale_ys[wid] - self.data_min_y_wid[wid] + self.data_min_y_wid[0]
            #     if self.parameters.speed_based_rendezvous == True:
            #         self.current_whale_assigned[wid] = False
            #         continue
            #         if self.current_whale_up[wid]:
            #             l = [get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 temp_whale_loc_y, temp_whale_loc_x) / self.parameters.boat_max_speed_mtpm \
            #                     for bid in range(self.parameters.number_of_agents)]
            #             # print(min(l), (next_surface_end_time - current_time)/60 - 5)
            #             if min(l)<=(next_surface_end_time - self.time_in_second)/60 - 5:
            #                 # print('Herre')
            #                 self.current_whale_assigned[wid] = True
            #     else:
            #         self.current_whale_assigned[wid] = False
            #         continue
            #         self.current_whale_assigned[wid] = self.current_whale_up[wid] and \
            #             any([get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 temp_whale_loc_y, temp_whale_loc_x) <= self.parameters.tagging_distance \
            #                     for bid in range(self.parameters.number_of_agents)])
            # else:
            #     if self.parameters.speed_based_rendezvous == True:
            #         self.current_whale_assigned[wid] = False
            #         continue
            #         if self.current_whale_up[wid]:
            #             l = [get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 self.current_whale_ys[wid], self.current_whale_xs[wid]) / self.parameters.boat_max_speed_mtpm \
            #                     for bid in range(self.parameters.number_of_agents)]
            #             # print(min(l), (next_surface_end_time - current_time)/60 - 5)
            #             if min(l) <= (next_surface_end_time - self.time_in_second)/60 - 5:
            #                 self.current_whale_assigned[wid] = True
            #     else:
            #         self.current_whale_assigned[wid] = False
            #         continue
            #         self.current_whale_assigned[wid] = self.current_whale_up[wid] and \
            #             any([get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 self.current_whale_xs[wid], self.current_whale_ys[wid]) <= self.parameters.tagging_distance \
            #                     for bid in range(self.parameters.number_of_agents)])

            
        # min_traj_len = min([ for wid in range(self.number_of_whales)])
        if self.parameters.experiment_type != 'Benchmark_Shane_Data':
            self.parameters.n_horizon_for_state_estimation = int(max_traj_len)
            self.parameters.n_horizon_for_evaluation = self.parameters.n_horizon_for_state_estimation - self.parameters.down_time_mean
        else:
            self.parameters.n_horizon_for_state_estimation *= 60 
            self.parameters.n_horizon_for_evaluation *= 60

        
    
        return self.file_names
        
    def step(self, control: Boat_Control):
        
        current_time = self.time_in_second

        for wid in range(self.number_of_whales):
            self.current_whale_up[wid] = any([current_time + 1 >= ss and current_time + 1 <= se \
                for (ss,se) in self.gt_surface_interval_scenario[wid]])

            if self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                receiver_error_var = np.array([self.loc_observation[wid][current_time + 1][2]**2]) \
                    if current_time + 1 in self.loc_observation[wid].keys() else None
                current_observed_xy = np.array([self.loc_observation[wid][current_time + 1][0], self.loc_observation[wid][current_time + 1][1]]) \
                    if current_time + 1 in self.loc_observation[wid].keys() else None            
                
                self.receiver_allw_allt_loc[wid][current_time + 1] = {'b_' + str(wid): XY_Sensor_Obs(\
                    xy = (current_observed_xy if current_observed_xy is not None else None), \
                        xy_cov = receiver_error_var) }
            else:
                receiver_x = self.ta_observation[wid][current_time + 1][0] if current_time + 1 in self.ta_observation[wid].keys() else None
                receiver_y = self.ta_observation[wid][current_time + 1][1] if current_time + 1 in self.ta_observation[wid].keys() else None
                # received_AOA = self.ta_observation[wid][current_time + 1][2] if current_time + 1 in self.ta_observation[wid].keys() else None
                received_AOA_candidate1 = self.ta_observation[wid][current_time + 1][3] if current_time + 1 in self.ta_observation[wid].keys() else None
                received_AOA_candidate2 = self.ta_observation[wid][current_time + 1][4] if current_time + 1 in self.ta_observation[wid].keys() else None
                receiver_error_std = self.ta_observation[wid][current_time + 1][5] if current_time + 1 in self.ta_observation[wid].keys() else None
                # if receiver_x is not None and np.isnan(receiver_x):
                #     receiver_error_std = None
                #     received_AOA = None
                #     receiver_x = None
                #     receiver_y = None
                # self.receiver_allw_allt_loc[wid][current_time + 1] = {'ta_' + str(wid) : AOA_Sensor_Obs(sx = receiver_x, sy = receiver_y, \
                #     true_AOA = None, obs_AOA = received_AOA, AOA_error_std = receiver_error_std)}
                self.receiver_allw_allt_loc[wid][current_time + 1] = {'ta_' + str(wid): AOA_Sensor_Obs_candidate(sx = receiver_x, sy = receiver_y, \
                        true_AOA = None, obs_AOA = None, \
                            obs_AOA_candidate1 = received_AOA_candidate1, obs_AOA_candidate2 = received_AOA_candidate2, \
                                AOA_error_std = receiver_error_std)}

            
                        
            if current_time + 1 in self.gt_whale_sighting[wid].keys():
                self.current_whale_xs[wid] = self.gt_whale_sighting[wid][current_time + 1][0]
                self.current_whale_ys[wid] = self.gt_whale_sighting[wid][current_time + 1][1]
            else:
                self.current_whale_xs[wid] = None
                self.current_whale_ys[wid] = None

        for bid in range(self.parameters.number_of_agents):
            bxy = get_gps_from_start_vel_bearing(self.current_agent_xs[bid], self.current_agent_ys[bid], \
                control.b_v[bid], np.pi/2 - control.b_theta[bid])
            self.current_agent_xs[bid] = bxy[0]
            self.current_agent_ys[bid] = bxy[1]
            
                
        for wid in range(self.number_of_whales):
            if self.current_whale_xs[wid] is None:
                continue
            self.visible_xy[wid] = None

            next_surface_end_time = [ se \
                for (ss,se) in self.gt_surface_interval_scenario[wid] if current_time + 1 >= ss and current_time + 1 <= se]
            if self.current_whale_up[wid] and len(next_surface_end_time) > 0:
                next_surface_end_time = next_surface_end_time[0]

            if self.parameters.overlay_GPS and wid != 0:
                temp_whale_loc_x = self.current_whale_xs[wid] - (self.data_min_x_wid[wid] + self.data_max_x_wid[wid])/2 \
                    + (self.data_min_x_wid[0] + self.data_max_x_wid[0])/2
                temp_whale_loc_y = self.current_whale_ys[wid] - (self.data_min_y_wid[wid] + self.data_max_y_wid[wid])/2\
                    + (self.data_min_y_wid[0] + self.data_max_y_wid[0])/2
            else:
                temp_whale_loc_x = self.current_whale_xs[wid]
                temp_whale_loc_y = self.current_whale_ys[wid]
            if self.parameters.speed_based_rendezvous == True:
                if not self.current_whale_assigned[wid] and self.current_whale_up[wid]:
                    
                    dist = [get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
                        temp_whale_loc_y, temp_whale_loc_x) for bid in range(self.parameters.number_of_agents)]
                    if min(dist) <= self.parameters.visual_distance:
                        self.visible_xy[wid] = (temp_whale_loc_x, temp_whale_loc_y)
                    
            # else:
            self.current_whale_assigned[wid] = self.current_whale_assigned[wid] or (self.current_whale_up[wid] and \
                any([get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
                    temp_whale_loc_y, temp_whale_loc_x) <= self.parameters.tagging_distance \
                        for bid in range(self.parameters.number_of_agents)]))
            # else:
            #     if self.parameters.speed_based_rendezvous == True:
            #         if not self.current_whale_assigned[wid] and self.current_whale_up[wid]:
            #             # l = [get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #             #     self.current_whale_ys[wid], self.current_whale_xs[wid]) / self.parameters.boat_max_speed_mtpm \
            #             #         for bid in range(self.parameters.number_of_agents)]
            #             # if min(l) <= (next_surface_end_time - current_time)/60 - 5:
            #             #     self.current_whale_assigned[wid] = True
            #             dist = [get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 self.current_whale_ys[wid], self.current_whale_xs[wid])\
            #                     for bid in range(self.parameters.number_of_agents)]

            #     else:
            #         self.current_whale_assigned[wid] = self.current_whale_assigned[wid] or (self.current_whale_up[wid] and \
            #             any([get_distance_from_latLon_to_meter(self.current_agent_ys[bid], self.current_agent_xs[bid], \
            #                 self.current_whale_ys[wid], self.current_whale_xs[wid]) <= self.parameters.tagging_distance \
            #                     for bid in range(self.parameters.number_of_agents)]))
        self.time_in_second += 1
        
    
    
    def visualize_whales(self, folder_ = None, time_start_end = None):
        plt.ioff()
        plt.cla()
        for wid in range(self.number_of_whales):
            w_locs = np.array(self.whales_allt_loc[wid])
            if time_start_end is not None:
                times = time_start_end[1] - time_start_end[0] + 1
                w_locs = w_locs[:times]
            else:
                times = len(w_locs)
            cols = np.array([whale_assigned_color if self.current_whale_assigned[wid] \
                else whale_surface_color if  self.whales_allt_up[wid][t] else whale_underwater_color \
                    for t in range(times)])
            w_sizes = np.array([8 if self.whales_allt_up[wid][t] else 2 \
                for t in range(times)])
            plt.scatter(w_locs[:, 0], w_locs[:, 1], c = cols, s = w_sizes)
        if folder_ is None:
            plt.show()
        else:
            plt.savefig(folder_ + 'state_'+str(time_start_end[1])+ '.png')
        plt.close()
        
            
    

