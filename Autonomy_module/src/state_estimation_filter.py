from src.system_observation import ObservationClass
from src.global_knowledge import Global_knowledge
from src.UKF import Adaptive_UKF_ARCTAN
from src.belief_state import Belief_State
import numpy as np
from src.configs.constants import *


class State_Estimation_Filter:

    def __init__(self, observation: ObservationClass):
        self.parameters : Global_knowledge = observation.parameters
        prefix, self.output_folder_name = self.parameters.get_stateEstimation_filterOutput_txyz_fnPrefix_path(observation.run_id)

        self.data_ukf = {wid: {'mean': [], 'std': [], 'whale_up': []} for wid in range(observation.number_of_whales)}
        
        # if self.parameters.experiment_type == 'Combined_Dominica_Data': 
        #     self.ukfs = {wid: Adaptive_UKF_GPS(self.parameters) for wid in range(observation.number_of_whales)}
        # else:
        self.ukfs = {wid: Adaptive_UKF_ARCTAN(self.parameters) for wid in range(observation.number_of_whales)}
        
        for wid in range(observation.number_of_whales):
            if self.parameters.experiment_type in ['Benchmark_Shane_Data', 'Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                intitial_variance = observation.initial_observation_cov[wid]
            else:
                intitial_variance = np.diag([observation.parameters.initial_obs_xy_error[0,0]]*4)
            self.ukfs[wid].initialize_filter(observation.initial_observation[wid], intitial_variance)
            
            
            # if self.parameters.experiment_type == 'Feb24_Dominica_Data':
            #     self.ukfs[wid].state_estimation(observation.get_observation_t(wid), \
            #         (observation.extra_obs[wid][1], observation.extra_obs[wid][2]))
            # else:
            #     self.ukfs[wid].state_estimation(observation.get_observation_t(wid))
            self.data_ukf[wid]['mean'].append(self.ukfs[wid].hat_x_k)
            self.data_ukf[wid]['std'].append(self.ukfs[wid].P_k)
            self.data_ukf[wid]['whale_up'].append(observation.current_whale_up[wid])

        w_xs = np.array([self.data_ukf[wid]['mean'][0][0, 0] for wid in range(observation.number_of_whales)])
        w_ys = np.array([self.data_ukf[wid]['mean'][0][1, 0] for wid in range(observation.number_of_whales)])
        w_thetas = np.array([np.arctan2(self.data_ukf[wid]['mean'][0][3, 0], self.data_ukf[wid]['mean'][0][2, 0]) for wid in range(observation.number_of_whales)])
        if self.parameters.experiment_type in ['Benchmark_Shane_Data','Combined_Dominica_Data', 'Feb24_Dominica_Data']:
            w_vs = np.zeros(observation.number_of_whales)
            for wid in range(observation.number_of_whales):
                dx = self.data_ukf[wid]['mean'][0][2, 0]
                dy = self.data_ukf[wid]['mean'][0][3, 0]
                w_vs[wid] = get_distance_from_latLon_to_meter(w_ys[wid] + dy, w_xs[wid] + dx, w_ys[wid], w_xs[wid])
        else:
            w_vs = np.array([np.linalg.norm([self.data_ukf[wid]['mean'][0][2, 0], self.data_ukf[wid]['mean'][0][3, 0]]) for wid in range(observation.number_of_whales)])
        
        num_agents = observation.current_agent_xs.shape[0]
        Pcov = {wid: self.data_ukf[wid]['std'][-1] for wid in range(observation.number_of_whales)}

        # signature = inspect.signature(Belief_State.__init__).parameters
        # for name, parameter in signature.items():
        #     print(name, parameter.default, parameter.annotation, parameter.kind)

        if self.parameters.experiment_type in ['Benchmark_Shane_Data','Combined_Dominica_Data', 'Feb24_Dominica_Data']:
            if self.parameters.overlay_GPS and self.parameters.experiment_type == 'Feb24_Dominica_Data':
                for wid in range(1, observation.number_of_whales):
                    w_xs[wid] = w_xs[wid] - (observation.data_min_x_wid[wid] + observation.data_max_x_wid[wid]) / 2 \
                        + (observation.data_min_x_wid[0] + observation.data_max_x_wid[0]) / 2
                    w_ys[wid] = w_ys[wid] - (observation.data_min_y_wid[wid] + observation.data_max_y_wid[wid]) /2 \
                        + (observation.data_min_y_wid[0] + observation.data_max_y_wid[0]) / 2

            ws_xy = [convert_longlat_to_xy_in_meters(w_xs[wid], w_ys[wid]) for wid in range(observation.number_of_whales)]
            bs_xy = [convert_longlat_to_xy_in_meters(observation.current_agent_xs[bid], observation.current_agent_ys[bid]) \
                for bid in range(observation.parameters.number_of_agents)]
            w_xs = np.array([w_xy[0] for w_xy in ws_xy])
            w_ys = np.array([w_xy[1] for w_xy in ws_xy])
            b_xs = np.array([b_xy[0] for b_xy in bs_xy])
            b_ys = np.array([b_xy[1] for b_xy in bs_xy])
            self.belief_state0 = Belief_State(self.parameters, state_str = None, time = 0, \
                number_of_boats = num_agents, b_x = b_xs, b_y = b_ys, \
                    b_theta = np.zeros(num_agents), b_v = np.zeros(num_agents), \
                        number_of_whales = observation.number_of_whales, w_x = w_xs, w_y = w_ys, w_theta = w_thetas, w_v = w_vs, \
                            assigned_whales = [], \
                                w_last_surface_start_time = observation.last_surface_start_time, w_last_surface_end_time = observation.last_surface_end_time, \
                                    Pcov = Pcov)
        else:
            self.belief_state0 = Belief_State(self.parameters, state_str = None, time = 0, \
                number_of_boats = num_agents, b_x = observation.current_agent_xs, b_y = observation.current_agent_ys, \
                    b_theta = np.zeros(num_agents), b_v = np.zeros(num_agents), \
                        number_of_whales = observation.number_of_whales, w_x = w_xs, w_y = w_ys, w_theta = w_thetas, w_v = w_vs, \
                            assigned_whales = [], \
                                w_last_surface_start_time = observation.last_surface_start_time, w_last_surface_end_time = observation.last_surface_end_time, \
                                    Pcov = Pcov)
    
    def get_next_estimation(self, observation: ObservationClass):
        observations_x_y_v_theta_up = np.zeros(shape= (observation.number_of_whales, 7))
        cov = {wid: None for wid in range(observation.number_of_whales)}
        for wid in range(observation.number_of_whales):
            if self.parameters.speed_based_rendezvous == True and \
                observation.current_observed_xy[wid][0] is not None and observation.current_observed_xy[wid][1] is not None:
                    self.data_ukf[wid]['mean'].append(np.array([observation.current_observed_xy[wid][0], observation.current_observed_xy[wid][1], 0,0]))
                    self.data_ukf[wid]['std'].append(np.zeros((4,4)))
                    self.data_ukf[wid]['whale_up'].append(True)
            else:
                if self.parameters.experiment_type in ['Benchmark_Shane_Data','Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                    self.ukfs[wid].state_estimation(observation.get_observation_t(wid), \
                        (observation.extra_obs[wid][1], observation.extra_obs[wid][2]))
                else:
                    self.ukfs[wid].state_estimation(observation.get_observation_t(wid))
                self.data_ukf[wid]['mean'].append(self.ukfs[wid].hat_x_k)
                self.data_ukf[wid]['std'].append(self.ukfs[wid].P_k)
                self.data_ukf[wid]['whale_up'].append(observation.current_whale_up[wid])

            if 1==2 and self.parameters.experiment_type == 'Benchmark_Shane_Data':
                observations_x_y_v_theta_up[wid, 0] = self.data_ukf[wid]['mean'][observation.current_time][0]
                observations_x_y_v_theta_up[wid, 1] = self.data_ukf[wid]['mean'][observation.current_time][1]
                observations_x_y_v_theta_up[wid, 2] = np.linalg.norm([self.data_ukf[wid]['mean'][observation.current_time][2], self.data_ukf[wid]['mean'][observation.current_time][3]])
                observations_x_y_v_theta_up[wid, 3] = np.arctan2(self.data_ukf[wid]['mean'][observation.current_time][3], self.data_ukf[wid]['mean'][observation.current_time][2])
            else:
                # try:
                if self.parameters.overlay_GPS and wid != 0 and self.parameters.experiment_type == 'Feb24_Dominica_Data':
                    temp_x = self.data_ukf[wid]['mean'][observation.current_time][0] \
                        - (observation.data_min_x_wid[wid] + observation.data_max_x_wid[wid])/2 \
                        + (observation.data_min_x_wid[0] + observation.data_max_x_wid[0])/2
                    temp_y = self.data_ukf[wid]['mean'][observation.current_time][1] - \
                        (observation.data_min_y_wid[wid] + observation.data_max_y_wid[wid])/2 \
                            + (observation.data_min_y_wid[0] + observation.data_max_y_wid[0])/2
                    w_xy = convert_longlat_to_xy_in_meters(temp_x, temp_y)
                else:
                    w_xy = convert_longlat_to_xy_in_meters(self.data_ukf[wid]['mean'][observation.current_time][0], self.data_ukf[wid]['mean'][observation.current_time][1])
                # except Exception as e:
                #     print(e)
                observations_x_y_v_theta_up[wid, 0] = w_xy[0]
                observations_x_y_v_theta_up[wid, 1] = w_xy[1]
                dx = self.data_ukf[wid]['mean'][observation.current_time][2]
                dy = self.data_ukf[wid]['mean'][observation.current_time][3]
                # try:
                w_v = get_distance_from_latLon_to_meter\
                            (self.data_ukf[wid]['mean'][observation.current_time][1] + dy, self.data_ukf[wid]['mean'][observation.current_time][0] + dx, \
                                self.data_ukf[wid]['mean'][observation.current_time][1], self.data_ukf[wid]['mean'][observation.current_time][0])
                # except Exception as e:
                #     print(e)
                observations_x_y_v_theta_up[wid, 2] = w_v
                observations_x_y_v_theta_up[wid, 3] = np.arctan2(dy, dx)

                observations_x_y_v_theta_up[wid, 5] = self.data_ukf[wid]['mean'][observation.current_time][0]
                observations_x_y_v_theta_up[wid, 6] = self.data_ukf[wid]['mean'][observation.current_time][1]
            
            observations_x_y_v_theta_up[wid, 4] = observation.current_whale_up[wid]

            cov[wid] = self.data_ukf[wid]['std'][observation.current_time]


        return observations_x_y_v_theta_up, cov


            
