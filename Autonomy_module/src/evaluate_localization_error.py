from src.system_state import System_state
from src.system_observation import ObservationClass
from src.global_knowledge import Global_knowledge
import numpy as np
import typing as t
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import os
from src.configs.constants import *

class EvalObjectClass:

    def __init__(self, gt: System_state):
        self.data_min_y = gt.data_min_y
        self.data_min_x = gt.data_min_x
        self.data_max_y = gt.data_max_y
        self.data_max_x = gt.data_max_x
        self.gt_x = {wid: [] for wid in range(gt.number_of_whales)}
        self.gt_y = {wid: [] for wid in range(gt.number_of_whales)}
        self.loc_x = {wid: [] for wid in range(gt.number_of_whales)}
        self.loc_y = {wid: [] for wid in range(gt.number_of_whales)}
        self.loc_Pcov = {wid: [] for wid in range(gt.number_of_whales)}
        self.ups = {wid: [] for wid in range(gt.number_of_whales)}
        self.r_locs = {wid: [] for wid in range(gt.number_of_whales)}
        self.localization_error = {wid: [] for wid in range(gt.number_of_whales)}
        self.l = -1

        if gt.parameters.experiment_type in ['Benchmark_Shane_Data','Combined_Dominica_Data', 'Feb24_Dominica_Data']:
            self.loc_long = {wid: [] for wid in range(gt.number_of_whales)}
            self.loc_lat = {wid: [] for wid in range(gt.number_of_whales)}

            self.receiver_long = {wid: [] for wid in range(gt.number_of_whales)}
            self.receiver_lat = {wid: [] for wid in range(gt.number_of_whales)}
    
    def update_logs(self, raw_obs: ObservationClass, observations_x_y_v_theta_up: np.ndarray, Pcov, \
        plot = False, folder_ = None, wid_ = None, receiver_xy = None):
        
        for wid in range(raw_obs.number_of_whales):
            self.gt_x[wid].append(raw_obs.gt_for_eval[wid, 0])
            self.gt_y[wid].append(raw_obs.gt_for_eval[wid, 1])
            self.loc_x[wid].append(observations_x_y_v_theta_up[wid, 0])
            self.loc_y[wid].append(observations_x_y_v_theta_up[wid, 1])

            if raw_obs.parameters.experiment_type in ['Benchmark_Shane_Data','Combined_Dominica_Data', 'Feb24_Dominica_Data']:
                self.loc_long[wid].append(observations_x_y_v_theta_up[wid, 5])
                self.loc_lat[wid].append(observations_x_y_v_theta_up[wid, 6])

                if receiver_xy is not None:
                    self.receiver_long[wid].append(receiver_xy[0])
                    self.receiver_lat[wid].append(receiver_xy[1])

            self.loc_Pcov[wid].append(Pcov[wid])

            self.ups[wid].append(observations_x_y_v_theta_up[wid, 4])
            # self.r_locs[wid].append(raw_obs.current_receivers_loc[wid][0])
            if raw_obs.parameters.experiment_type in ['Benchmark_Shane_Data','Combined_Dominica_Data', 'Feb24_Dominica_Data']:
                # try:
                loc_error = get_distance_from_latLon_to_meter(self.gt_y[wid][-1], self.gt_x[wid][-1], \
                        observations_x_y_v_theta_up[wid, 6], observations_x_y_v_theta_up[wid, 5])
                # except Exception as e:
                #     print(e)
            else:
                loc_error = np.linalg.norm([ self.gt_x[wid][-1] - self.loc_x[wid][-1], self.gt_y[wid][-1] - self.loc_y[wid][-1] ])
            # if loc_error > 2000:
            #     print('here: localization error more than 2000 for whale ',wid)
            self.localization_error[wid].append(loc_error)
        self.l += 1
        if plot == False:
            return
        plt.ioff()
        plt.cla()
        
        for wid in range(raw_obs.number_of_whales):
            
            plt.scatter(raw_obs.gt_for_eval[wid, 0], raw_obs.gt_for_eval[wid, 1], \
                c = 'magenta' if raw_obs.current_whale_up[wid] else 'red', label ='Whale current location')

            if raw_obs.current_receivers_loc is not None and raw_obs.current_receivers_loc[wid] is not None:
                for receiver_xy in raw_obs.current_receivers_loc[wid]:
                    plt.scatter(receiver_xy[0], receiver_xy[1], s = 1, \
                        c = 'black' if raw_obs.current_whale_up[wid] else 'red', label ='receiver current location')
        
           
            if 1==1:
                plt.scatter(observations_x_y_v_theta_up[wid, 5], observations_x_y_v_theta_up[wid, 6], \
                    c = 'royalblue' if raw_obs.current_whale_up[wid] else 'black', label ='Mean whale particles')
               
            
            
           

            plt.scatter(self.gt_x[wid], self.gt_y[wid], \
                label = 'Whale true previous locations', c = np.array(['magenta' if w_up else 'red' for w_up in self.ups[wid]]), s = 1)
            
           
            if 1==1:
                plt.scatter(self.loc_long[wid], self.loc_lat[wid],\
                    label = 'Previous mean whale particles', \
                        c = np.array(['royalblue' if w_up else 'black' for w_up in self.ups[wid]]), s = 1)
                if len(self.receiver_long[wid]) > 0:
                    plt.scatter(self.receiver_long[wid], self.receiver_lat[wid], label = 'Sensor', c = 'green', s = 1)

            if len(self.ups[wid]) > 0:
                plt.title('Particles after '+ str(self.l) +' steps, whale is ' + ('up' if raw_obs.current_whale_up[wid] else 'down'))
                plt.legend()

            if folder_ is None:
                folder_wid = raw_obs.parameters.base_output_path + 'whale_' + (str(wid_) if wid_ is not None else str(wid)) + '/'
            else:
                folder_wid = folder_ + 'whale_' + (str(wid_) if wid_ is not None else str(wid)) + '/'

            if not os.path.exists(folder_wid):
                os.makedirs(folder_wid)
            # plt.savefig(folder_wid + 'particles' + str(self.l) +'.png')
            plt.savefig(folder_wid + 'particles.png')
            
            plt.close()

            l = len(self.ups[wid])
            plt.cla()
            plt.plot(self.localization_error[wid], label = 'error')
            cols = np.array([[0, 0, 1] if up else [1, 1, 1] for up in self.ups[wid]])
            plt.scatter(np.arange(l), self.ups[wid], c = cols, label = 'whale up')

            plt.grid()
            plt.legend()   
            plt.title('Mean_error:' + str(round(np.mean(self.localization_error[wid]), 2)) \
                + ' std_error:' + str(round(np.std(self.localization_error[wid]), 2)))

            plt.savefig(folder_wid + 'localization_error.png')
    
            plt.cla()
            # self.ax.cla()
