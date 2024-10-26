import numpy as np
import typing as t

class Global_knowledge:
    
    def __init__(self) -> None:
        self.Acoustic_AOA_obs_error_std_degree: float = None
        self.Vhf_AOA_obs_error_std_degree: float = None

        self.Acoustic_XY_obs_error_cov_m2: np.ndarray = None
        self.Vhf_XY_obs_error_cov_m2: np.ndarray = None

        self.initial_obs_xy_error: np.ndarray = None

        
        self.experiment_type: str = None
        self.observation_type: str = None

        self.number_of_whales: int = None
        self.number_of_agents: int = None
        
        self.base_output_path: str = None
        self.n_horizon_for_state_estimation = None,
        self.n_horizon_for_evaluation = None

        self.average_num_runs: int = None
        self.discount_factor: int = None

        self.agents_own_vhf : bool = None
        self.Acoustic_sensor_speed_mtpm: float = None
        self.Vhf_speed_mtpm: float = None

        self.max_listening_radius: float = None

        self.surface_time_var: float = None
        self.surface_time_mean: float = None
        self.down_time_mean: float = None
        self.down_time_var: float = None

        self.base_output_path_prefix: str = None

        self.tagging_distance: float = None
        self.visual_distance: float = None
        self.parsed_whale_data_output: str = None
        self.boat_max_speed_mtpm = None

        self.overlay_GPS : bool = None
        self.speed_based_rendezvous : bool = None
        self.dates :t.List[str] = None


    def set_attr(self, dict_ = {}):
        if dict_ is not None:
            for key, value in dict_.items():
                if key == "experiment_type":
                    assert value in ['Pure_Simulation', 'Benchmark_Shane_Data', 'Pamguard_output', 'Combined_Dominica_Data', 'Feb24_Dominica_Data'], \
                        "experiment_type should be in ['Pure_Simulation', 'Benchmark_Shane_Data', 'Pamguard_output', 'Combined_Dominica_Data', 'Feb24_Dominica_Data']"

                if key == "observation_type":
                    assert value in ['Acoustic_AOA_no_VHF','Acoustic_xy_no_VHF', 'Acoustic_AOA_VHF_AOA', 'Acoustic_xy_VHF_xy'], \
                        "observation_type should be in ['Acoustic_AOA_no_VHF','Acoustic_xy_no_VHF', 'Acoustic_AOA_VHF_AOA', 'Acoustic_xy_VHF_xy']"

                elif key in ["Acoustic_XY_obs_error_cov_m2", "Vhf_XY_obs_error_cov_m2"]:
                    value = np.array(value)
                
                elif key in ["initial_obs_xy_error"]:
                    value = np.array(value)
                
                setattr(self, key, value)
            
            if self.experiment_type in ['Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                self.base_output_path = self.base_output_path_prefix + self.experiment_type \
                + '_r' + str(self.tagging_distance) + '_w' + '_'.join(self.dates[:self.number_of_whales]) + '_a' + str(self.number_of_agents) + '/' 
            else:
                self.base_output_path = self.base_output_path_prefix + self.experiment_type + '_' + self.observation_type \
                + '_r' + str(self.tagging_distance) + '_w' + str(self.number_of_whales) + '_a' + str(self.number_of_agents) + '/' 

            if self.experiment_type == 'Combined_Dominica_Data':
                self.observations_per_minute = 60
                self.parsed_whale_data_output = 'Engg_whale_postprocessed_trace/' 
            elif self.experiment_type == 'Feb24_Dominica_Data':
                self.observations_per_minute = 60
                self.parsed_whale_data_output = 'Feb24_Dominica_Data/'
            else:   
                self.observations_per_minute = 60
                self.parsed_whale_data_output = 'dswp_parsed_data_moving_sensors/'

            if self.observation_type == 'Acoustic_AOA_no_VHF':
                self.acoustic_obs_type = 'AOA'
                self.vhf_obs_type = None
            elif self.observation_type == 'Acoustic_xy_no_VHF':
                self.acoustic_obs_type = 'xy'
                self.vhf_obs_type = None
            elif self.observation_type == 'Acoustic_AOA_VHF_AOA':
                self.acoustic_obs_type = 'AOA'
                self.vhf_obs_type = 'AOA'
            elif self.observation_type == 'Acoustic_xy_VHF_xy':
                self.acoustic_obs_type = 'xy'
                self.vhf_obs_type = 'xy'
            
    def get_system_seed_txyz_fnPrefix_path(self, run_id):
        return self.base_output_path + 'Run_' + str(run_id) + '/'
        
    def get_system_gt_txyz_fnPrefix_path(self, run_id):
        path = self.base_output_path + 'Run_'+ str(run_id) + '/'
        return path + 'System_gt_txyz_', path

    def get_rawObservation_filterInput_fnPrefix_path(self, run_id):
        path = self.base_output_path + 'Run_'+ str(run_id) + '/' + self.observation_type + '/'
        return path + 'RawObservation_filterInput_', path

    def get_stateEstimation_filterOutput_txyz_fnPrefix_path(self, run_id):
        path = self.base_output_path + 'Run_'+ str(run_id) + '/' + self.observation_type + '/'
        return path + 'StateEstimation_filterOutput_txyz_', path

    def get_beliefState0Output_path(self, run_id):
        path = self.base_output_path + 'Run_'+ str(run_id) + '/' + self.observation_type + '/agents_' + str(self.number_of_agents) + '/'
        return path + 'BeliefState0_' , path

        
    def get_policyOutput_path(self, run_id, policyname):
        path = self.base_output_path + 'Run_'+ str(run_id) + '/' + self.observation_type + \
            '/agents_' + str(self.number_of_agents)  + '/viewing_rad_' +str(int(self.tagging_distance)) + '/' + policyname + '/'
        return path