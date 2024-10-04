from src.system_state import System_state
from src.global_knowledge import Global_knowledge
import numpy as np
import typing as t
from src.configs.constants import *

class ObservationClass:
    
    def __init__(self, gt: System_state) -> None:
        self.run_id : int = gt.run_id
        self.file_names = gt.file_names
        self.parameters: Global_knowledge = gt.parameters
        self.number_of_whales: int = gt.number_of_whales
        self.current_time = gt.time_in_second if self.parameters.experiment_type in ['Combined_Dominica_Data','Feb24_Dominica_Data'] \
            else gt.time_in_minute
    
        self.current_agent_xs: np.ndarray = None
        self.current_agent_ys: np.ndarray = None
        self.current_whale_up: t.Dict[int, bool] = {}
        self.current_whale_assigned: t.Dict[int, bool] = {}
        self.initial_observation: t.Dict[int, np.ndarray] = {}
        self.current_receiver_error: t.Dict[int, t.List[np.ndarray]] = {}
        self.current_receivers_loc: t.Dict[int, t.List[np.ndarray]] = {}
        self.current_observed_AOA: t.Dict[int, t.List[np.ndarray]] = {}
        self.current_observed_xy: t.Dict[int, t.List[np.ndarray]] = {}
        self.vhf_taken = {}

        self.current_observed_AOA_candidate1: t.Dict[int, t.List[np.ndarray]] = {}
        self.current_observed_AOA_candidate2: t.Dict[int, t.List[np.ndarray]] = {}

        self.last_surface_start_time = [np.nan for wid in range(gt.number_of_whales)]
        self.last_surface_end_time = [np.nan for wid in range(gt.number_of_whales)]
     
        
        self.set_observation_t(gt, 0)

    def set_observation_t(self, gt: System_state, time_step: int):
        
        self.current_whale_up = gt.current_whale_up

        self.current_time = time_step
        
        if self.parameters.experiment_type in ['Benchmark_Shane_Data', 'Feb24_Dominica_Data', 'Combined_Dominica_Data']:
            self.extra_obs = {wid: gt.extra_obs[wid][time_step] if time_step in gt.extra_obs[wid].keys() \
                else (gt.current_whale_up[wid], False, False) \
                    for wid in range(gt.number_of_whales)}
        for wid in range(gt.number_of_whales):

            start_times_before = [int(surface_start) \
                for (surface_start, _) in gt.gt_surface_interval_scenario[wid] if surface_start <= time_step]
            end_times_before = [int(surface_end) \
                for (surface_start, surface_end) in gt.gt_surface_interval_scenario[wid] if surface_end < time_step]
            self.last_surface_start_time[wid] = start_times_before[-1] if len(start_times_before) > 0 else np.nan
            self.last_surface_end_time[wid] = end_times_before[-1] if len(end_times_before) > 0 else np.nan

        

        self.current_whale_assigned = gt.current_whale_assigned
        self.current_agent_xs: np.ndarray = gt.current_agent_xs
        self.current_agent_ys: np.ndarray = gt.current_agent_ys
        
        if self.current_time == 0:
            if self.parameters.experiment_type in ['Benchmark_Shane_Data', 'Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                self.initial_observation_cov = {}
                self.data_min_y_wid = gt.data_min_y_wid
                self.data_min_x_wid = gt.data_min_x_wid
                self.data_max_y_wid = gt.data_max_y_wid
                self.data_max_x_wid = gt.data_max_x_wid

            for wid in range(self.number_of_whales):
                if self.parameters.experiment_type in ['Benchmark_Shane_Data', 'Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                    self.initial_observation[wid] = gt.extra_info_from_experiment['initial_observation'][wid]
                    self.initial_observation_cov[wid] = gt.extra_info_from_experiment['initial_observation_cov'][wid]
                    
                else:
                    self.initial_observation[wid] = np.random.multivariate_normal \
                        (np.array([gt.current_whale_xs[wid], gt.current_whale_ys[wid]]), \
                            self.parameters.initial_obs_xy_error)
        self.gt_for_eval = np.array([[gt.current_whale_xs[wid], gt.current_whale_ys[wid]] for wid in range(self.number_of_whales)])

        for wid in range(self.number_of_whales):
            
            self.current_receiver_error[wid] = None
            self.current_receivers_loc[wid] = None
            self.current_observed_AOA[wid] = None
            self.current_observed_xy[wid] = None
            if self.parameters.speed_based_rendezvous == True:
                self.current_observed_xy[wid] = gt.visible_xy[wid]

            self.current_observed_AOA_candidate1[wid] = None
            self.current_observed_AOA_candidate2[wid] = None

            if not self.current_whale_up[wid]:
                self.vhf_taken[wid] = False
            elif wid not in self.vhf_taken.keys() or self.vhf_taken[wid]:
                self.vhf_taken[wid] = False
            else:
                self.vhf_taken[wid] = True

           
            if self.parameters.experiment_type in ['Feb24_Dominica_Data', 'Combined_Dominica_Data'] or \
            (self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA']):
                self.current_receivers_loc[wid] = []
                self.current_receiver_error[wid] = []
                self.current_observed_AOA_candidate1[wid] = []
                self.current_observed_AOA_candidate2[wid] = []
                try:
                    for sensor, sensed_val in gt.receiver_allw_allt_loc[wid][self.current_time].items():
                        self.current_receivers_loc[wid].append((sensed_val.sx, sensed_val.sy))
                        self.current_observed_AOA_candidate1[wid].append(sensed_val.obs_AOA_candidate1)
                        self.current_observed_AOA_candidate2[wid].append(sensed_val.obs_AOA_candidate2)
                        self.current_receiver_error[wid].append(sensed_val.AOA_error_std**2 if sensed_val.AOA_error_std is not None else None)
                except Exception as e:
                    print(e)
            
            elif self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                
                self.current_receiver_error[wid] = []
                self.current_observed_xy[wid] = []
                try:
                    for sensor, sensed_val in gt.receiver_allw_allt_loc[wid][self.current_time].items():
                        self.current_receiver_error[wid].append(sensed_val.xy_cov if sensed_val.xy_cov is not None else None)
                        self.current_observed_xy[wid].append(sensed_val.xy if sensed_val.xy is not None else None)
                except Exception as e:
                    print(e)
                
                    
                
            
    def get_observation_t(self, wid: int) -> ObservationClass_whale:
        if self.parameters.experiment_type in ['Feb24_Dominica_Data', 'Combined_Dominica_Data'] or \
            (self.parameters.experiment_type == 'Benchmark_Shane_Data' and self.parameters.observation_type in ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA']):
            return ObservationClass_whale_TA(current_whale_up = self.current_whale_up[wid], current_receiver_error = self.current_receiver_error[wid], \
                receiver_current_loc = self.current_receivers_loc[wid], \
                    current_observed_AOA_candidate1 = self.current_observed_AOA_candidate1[wid], \
                        current_observed_AOA_candidate2 = self.current_observed_AOA_candidate2[wid],\
                            current_observed_xy = self.current_observed_xy[wid])
        
        return ObservationClass_whale_TA(current_whale_up = self.current_whale_up[wid], current_receiver_error = self.current_receiver_error[wid], \
                receiver_current_loc = None, \
                    current_observed_AOA_candidate1 = None, \
                        current_observed_AOA_candidate2 = None,\
                            current_observed_xy = self.current_observed_xy[wid])
        
    def terminal(self):
        if sum(self.current_whale_assigned.values()) == self.number_of_whales:
            return True
        return False

