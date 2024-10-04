from src.configs.constants import *
from src.global_knowledge import Global_knowledge
import numpy as np
import traceback, scipy
from scipy.linalg import cholesky





class Filter:
    def __init__(self) -> None:
        pass
    def initialize_filter(self, mean, covariance):
        pass

    def state_estimation(self, observation):
        pass

#TODO: change code such that more than one receiver can be present
# observation.receiver_current_loc, current_receiver_error, current_observed_AOA -> make them list to numpy

class Adaptive_UKF_ARCTAN(Filter):
    
    def __init__(self, parameters: Global_knowledge) -> None:
        # self.speed = 0 
        self.parameters : Global_knowledge = parameters
        self.T = 1
        self.F = np.array([[1, 0, self.T, 0], [0, 1, 0, self.T], [0, 0, 1, 0], [0, 0, 0, 1]])
    
        # if parameters.experiment_type == 'Benchmark_Shane_Data':
        #     self.Q = np.array([[10 * np.power(self.T, 2) / 2, 0, 10 * np.power(self.T, 1) / 1, 0], \
        #         [0, 10 * np.power(self.T, 2) / 2, 0, 10 * np.power(self.T, 1) / 1], \
        #             [0, 0, 10 * self.T, 0], \
        #                 [0, 0, 0, 10 * self.T]])
        # else:
        if 1==1:
            ie = np.diag([1e-10, 1e-10, 1e-20, 1e-20])
            self.Q = np.array([[ie[0,0] * np.power(self.T, 2) / 2, 0, ie[2,2] * np.power(self.T, 1) / 1, 0], \
                [0, ie[1,1] * np.power(self.T, 2) / 2, 0, ie[3,3] * np.power(self.T, 1) / 1], \
                    [0, 0, ie[2,2] * self.T, 0], \
                        [0, 0, 0, ie[3,3] * self.T]])

        self.n = self.Q.shape[0]
        if parameters.observation_type in ['Acoustic_xy_VHF_xy', 'Acoustic_xy_no_VHF']:
            # self.R_down = parameters.Acoustic_XY_obs_error_cov_m2[0]
            # self.R_up = parameters.Vhf_XY_obs_error_cov_m2[0]
            self.R_down = np.diag([2e-6, 2e-6])
            self.R_up = np.diag([1e-8, 1e-8])
        else:
            self.R_down = np.array([parameters.Acoustic_AOA_obs_error_std_degree ** 2])
            self.R_up = np.array([parameters.Vhf_AOA_obs_error_std_degree ** 2])
        
        self.n2_p1 = 2 * self.n + 1
        # self.alpha = 1e-2
        #self.alpha = 1e-4    
         
        self.beta = 2
        self.alpha = 1e-2
        self.kappa = 0 
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.debug_stmt = ''
        self.time_step = 0

    def initialize_filter(self, mean: np.ndarray, covariance: np.ndarray):
        self.hat_x_k = np.zeros(self.n).reshape(self.n, 1)
        s = mean.shape[0]
        self.hat_x_k[:s,0] = mean # TODO: shape?
        self.P_k = np.copy(self.Q)
        if self.parameters.experiment_type == 'Benchmark_Shane_Data':
            self.P_k = np.copy(covariance)

        self.surface_behavior_ys = [] #[self.hat_x_k[1, 0]]
        self.surface_behavior_xs = [] #[self.hat_x_k[0, 0]]
        self.surface_behavior_ts = [] #[0]

    def system_equation(self, x: np.ndarray, v: np.ndarray):
        fx = np.matmul(self.F, x)
        return fx + v

    def system_equation_noniose(self, x: np.ndarray):
        fx = np.matmul(self.F, x)
        return fx
    
    def observation_function_noniose(self, x: np.ndarray, observation: ObservationClass_whale):
        
        if self.parameters.experiment_type not in ['Benchmark_Shane_Data'] or \
            (observation.current_whale_up and self.parameters.vhf_obs_type == 'AOA') or \
                (observation.current_whale_up == False and self.parameters.acoustic_obs_type == 'AOA'):
            theta = np.zeros(shape = (len(observation.receiver_current_loc), x.shape[1]))
            # try:
            # if self.parameters.experiment_type == 'Benchmark_Shane_Data':
                    
            #         for l, observer_loc in enumerate(observation.receiver_current_loc): 
            #             y0 = x[1, :] - observer_loc[1]
            #             x0 = x[0, :] - observer_loc[0]
            #             theta_ = np.arctan2(y0, x0) * Radian_to_degree
            #             theta[l, :] = theta_
            # else:
            if 1==1:
                    for l, observer_loc in enumerate(observation.receiver_current_loc): 
                        p1_long = observer_loc[0]
                        p2s_long = x[0,:]
                        p1_lat = observer_loc[1]
                        p2s_lat = x[1, :]
                        for i in range(p2s_lat.shape[0]):
                            theta[l,i] = np.mod(Geodesic.WGS84.Inverse(p1_lat, p1_long, p2s_lat[i], p2s_long[i])['azi1'], 360)
                    # return theta.reshape(1, p2s_lat.shape[0])
            # except Exception as e:
            #     print('Debug')
            return theta

        elif (observation.current_whale_up and self.parameters.vhf_obs_type == 'xy') or \
            (observation.current_whale_up ==False and self.parameters.acoustic_obs_type == 'xy'):
            xy = np.array([x[0], x[1]]).reshape(2, self.n2_p1)
            return xy
        return np.zeros((2, self.n2_p1))
    
    def observation_function(self, x: np.ndarray, noise: np.ndarray, observation: ObservationClass_whale):
        return self.observation_function_noniose(x, observation) + noise

    def extract_observed_value(self, observaion: ObservationClass_whale):
        if observaion.current_observed_AOA == None:
            return None
        if observaion.current_whale_up:
            if self.parameters.vhf_obs_type == 'AOA':
                return np.array([obs_aoa for obs_aoa in observaion.current_observed_AOA]).reshape(-1,1)
            elif self.parameters.vhf_obs_type == 'xy':
                return observaion.current_observed_xy
        else:
            if self.parameters.acoustic_obs_type == 'AOA':
                return np.array([obs_aoa for obs_aoa in observaion.current_observed_AOA]).reshape(-1,1)
            elif self.parameters.acoustic_obs_type == 'xy':
                return observaion.current_observed_xy
        return None

    def isPD(self, B):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def find_nearest_PSD(self, A):
        # https://github.com/florisvb/pyUKFsqrt/blob/main/ukf_sqrt/utils.py
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
    
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3


    def state_estimation(self, observation: ObservationClass_whale, acoustic_silent_start_end = None):
        self.time_step += 1
        try:
            if self.parameters.experiment_type in ['Benchmark_Shane_Data','Feb24_Dominica_Data', 'Combined_Dominica_Data'] :
                self.state_estimation_AOA_TA(observation, acoustic_silent_start_end)
            elif self.parameters.observation_type in ['Acoustic_AOA_VHF_AOA', 'Acoustic_AOA_no_VHF']:
                self.state_estimation_AOA(observation)
            else:
                self.state_estimation_xy(observation)
        except Exception as e:
            print('Exception: ', e)
            self.debug_stmt += 'time:' + str(self.time_step) + ' Adaptive_UKF_ARCTAN.state_estimation '+ str(e)
            full_traceback = traceback.extract_tb(e.__traceback__)
            filename, lineno, funcname, text = full_traceback[-1]
            for traceback_ in full_traceback:
                print(traceback_)
            print("Error in ", filename, " line:", lineno, " func:", funcname, ", exception:", e)
    
    def state_estimation_AOA(self, observation: ObservationClass_whale):
        

        c = self.n + self.lambda_
        Wm = np.concatenate(([self.lambda_ / c], 0.5 / c + np.zeros(2 * self.n)))
        Wc = np.concatenate(([self.lambda_ / c + (1 - self.alpha ** 2 + self.beta)], 0.5 / c + np.zeros(2 * self.n)))
        
        U = (self.n + self.lambda_) * self.P_k
        try:
            self.U = cholesky(U)
        except Exception as e:
            print('Issue with matrix decomposition: ', e)
            #TODO: here
            try:
                if not self.isPD(U): # is not positive definite Do something   
                    U = self.find_nearest_PSD(U)
                    self.U = cholesky(U)
            except Exception as e:
                print('PD ?', e)

            # self.P_k = np.diag([self.parameters.initial_obs_xy_error[0,0]]*4)
            # self.U = cholesky((self.n + self.lambda_) * self.P_k)

        # Calculate sigma points
        points = np.zeros((self.n, 2 * self.n + 1))
        points[:, 0] = self.hat_x_k[:, 0]
        for i in range(self.n):
            points[:, i + 1] = self.hat_x_k[:, 0] + self.U[i]
            points[:, self.n + i + 1] = self.hat_x_k[:, 0] - self.U[i]

        predicted_x = np.zeros((self.n, 1))
        all_predictions = self.system_equation_noniose(points)
        for i in range(2 * self.n + 1):
            predicted_x[:, 0] += all_predictions[:, i] * Wm[i]
        predicted_P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            predicted_error_XX = all_predictions[:, i].reshape(self.n, 1) - predicted_x
            predicted_P += np.matmul(predicted_error_XX, \
                np.transpose(predicted_error_XX)) * Wc[i] 
        predicted_P += self.Q

        observed_z = self.extract_observed_value(observation) # N \times 1
        # if (observed_z is not None and not np.isnan(observed_z).any()) and len(observed_z) > 0:
        if self.parameters.experiment_type == 'Benchmark_Shane_Data':
            flag = (observed_z is not None and not np.isnan(observed_z).any() and len(observed_z) > 0)
        else:
            flag = observed_z is not None and not any([ae is None for ae in observed_z.flatten() ])
        
        if flag:
        
            if observation.current_receiver_error is None or any([obs is None for obs in observation.current_receiver_error])\
                or np.isnan(observation.current_receiver_error):
                R = self.R_up if observation.current_whale_up else self.R_down
            else:
                R = np.diag([receiver_error for receiver_error in observation.current_receiver_error])
            R_shape = R.shape[0]
            
            predicted_z = np.zeros(shape =(R_shape, 1))
            all_observations = self.observation_function_noniose(points, observation)
            for i in range(2 * self.n + 1):
                predicted_z += (all_observations[:, i] * Wm[i]).reshape((R_shape, 1))
            predicted_z.reshape(R_shape, 1)
            predicted_z = np.mod(predicted_z, 360)
            observation_error = angle_diff_degree(observed_z, predicted_z)
            
            error_cov_YY = np.zeros((R_shape, R_shape))
            for i in range(2 * self.n + 1):
                error_YY = angle_diff_degree(all_observations[:, i], predicted_z[:,0]).reshape(R_shape, 1)
                error_cov_YY += np.matmul(error_YY, np.transpose(error_YY)) * Wc[i]
            error_cov_YY += R

            error_cov_XY = np.zeros((self.n, R_shape))
            for i in range(2 * self.n + 1):
                error_XX = (all_predictions[:, i] - predicted_x[:,0]).reshape(self.n, 1)
                error_YY = angle_diff_degree(all_observations[:, i], predicted_z[:,0]).reshape(R_shape, 1)
                error_cov_XY += np.matmul(error_XX, np.transpose(error_YY)) * Wc[i]
        
            K = np.dot(error_cov_XY, np.linalg.inv(error_cov_YY)) # TODO: dot or matmul
            self.hat_x_k = predicted_x + np.matmul(K, observation_error)
            self.P_k = predicted_P - np.matmul(K, np.matmul(error_cov_YY, np.transpose(K)))  # when obs is too large P is non-positive definte

            if self.isPD(self.P_k): # is not positive definite Do something   
                self.P_k = self.find_nearest_PSD(self.P_k)
                # A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].
                # [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                # [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
                # matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

            new_angle = self.observation_function_noniose(self.hat_x_k, observation)
            self.post_estimation_observation_error = angle_diff_degree(observed_z, new_angle)[0,0]

            self.surface_behavior_ys.append(self.hat_x_k[1, 0])
            self.surface_behavior_xs.append(self.hat_x_k[0, 0])
            self.surface_behavior_ts.append(self.time_step)
        else:
            self.hat_x_k = predicted_x
            self.P_k = predicted_P

            self.estimate_theta_from_surface_behavior(observation.current_whale_up)

        if np.isnan(self.P_k).any() or np.isnan(self.hat_x_k).any():
            print('UKF: state_estimation_AOA: np.isnan(self.P_k).any() or np.isnan(self.hat_x_k).any()')
        # return Loc_xy(x = self.hat_x_k[0,0], y = self.hat_x_k[1,0]), self.P_k

    def estimate_theta_from_surface_behavior(self, whale_up):
        if self.parameters.experiment_type in ['Combined_Dominica_Data', 'Feb24_Dominica_Data'] \
            and len(self.surface_behavior_ys) > 120 and whale_up:
            self.calculated_theta_whale_up = True
            # if self.prev_len_surface_behavior == len(self.surface_behavior_ts) and self.refresh_data == False:
            #     return

            # len = len(self.surface_behavior_ys)
            t_window = len(self.surface_behavior_ys) # min(len(self.surface_behavior_ys), 60*5)
            values_x = np.array(self.surface_behavior_xs[-t_window:])
            values_y = np.array(self.surface_behavior_ys[-t_window:])
            ts = self.surface_behavior_ts[-t_window:]
            # speed = np.mean([np.sqrt((x1 - x0)**2 + (y1 - y0)**2)/(t1 - t0) for x0, x1, y0, y1, t0, t1 \
            #     in zip(values_x[:-1], values_x[1:], values_y[:-1], values_y[1:], ts[:-1], ts[1:]) ])
            speed = np.mean([get_distance_from_latLon_to_meter(y0, x0, y1, x1)/(t1 - t0) for x0, x1, y0, y1, t0, t1 \
                in zip(values_x[:-1], values_x[1:], values_y[:-1], values_y[1:], ts[:-1], ts[1:]) ])
            speed = min(speed, Whale_speed_mtpm/(5*60))
            # speed = min([2.2522522522522524e-6, speed])
            
            NSxy = t_window * np.sum(np.multiply(values_x, values_y))
            SxSy = np.sum(values_x) * np.sum(values_y)
            NSx2 = t_window * np.sum(np.square(values_x))
            Sx2 = np.square(np.sum(values_x))
            
            slope_line = (NSxy - SxSy) / (NSx2 - Sx2) # np.arctan2(NSxy - SxSy, NSx2 - Sx2)
            Sy = np.sum(values_y)
            Sx = np.sum(values_x)
            intercept_line = (Sy - slope_line * Sx) / t_window
            slope_line2 = -1 / slope_line

            start_point = (values_x[0], values_y[0])
            intercept_line2_start = start_point[1] - slope_line2 * start_point[0]
            start_meet_x = (intercept_line2_start - intercept_line) / (slope_line - slope_line2)
            start_meet_y = slope_line * start_meet_x + intercept_line

            end_point = (values_x[-1], values_y[-1])
            intercept_line2_end = end_point[1] - slope_line2 * end_point[0]
            end_meet_x = (intercept_line2_end - intercept_line) / (slope_line - slope_line2)
            end_meet_y = slope_line * end_meet_x + intercept_line

            theta = np.arctan2(NSxy - SxSy, NSx2 - Sx2)
            if end_meet_y >= start_meet_y and end_meet_x >= start_meet_x: # should be in the 1st quadrant
                theta = theta if (theta >= 0 and theta <= np.pi / 2) else np.pi + theta 
            elif end_meet_y >= start_meet_y and end_meet_x <= start_meet_x: # should be in the 2nd quadrant
                theta = theta if (theta >= np.pi / 2 and theta <= np.pi) else np.pi + theta
            elif end_meet_y <= start_meet_y and end_meet_x >= start_meet_x: # should be in the 4th quadrant
                theta = theta if (theta <= 0 and theta >= - np.pi / 2) else - np.pi + theta 
            elif end_meet_y <= start_meet_y and end_meet_x <= start_meet_x: # should be in the 3rd quadrant
                theta = theta if (theta <= - np.pi / 2 and theta >= - np.pi) else - np.pi + theta 
            else:
                theta = np.arctan2(end_meet_y - start_meet_y, end_meet_x - start_meet_x)


            # delta_x = speed * np.cos(theta)
            # delta_y = speed * np.sin(theta)
            theta_ = angle_diff_radian(np.pi/2, theta)
            next_point = get_gps_from_start_vel_bearing(self.hat_x_k[0,0], self.hat_x_k[1,0], speed, theta_)
            self.delta_x = next_point[0] - self.hat_x_k[0,0]
            self.delta_y = next_point[1] - self.hat_x_k[1,0]
            self.hat_x_k[2,0] = self.delta_x #/10
            self.hat_x_k[3,0] = self.delta_y #/10



    def state_estimation_AOA_TA(self, observation: ObservationClass_whale_TA, acoustic_silent_start_end = None):
        self.prev_len_surface_behavior = len(self.surface_behavior_ts)
        if not hasattr(self, 'dive_phase_data_collection_start'):
            self.dive_phase_data_collection_start = False
        
        if (acoustic_silent_start_end is not None and acoustic_silent_start_end[1]): # silence end
            self.dive_phase_data_collection_start = True
        elif acoustic_silent_start_end is not None and acoustic_silent_start_end[0]: # silence start
            self.dive_phase_data_collection_start = False

        if not hasattr(self, 'calculated_theta_whale_up'):
            self.calculated_theta_whale_up = False
        
        
        self.selected_aoa = None
        c = self.n + self.lambda_
        Wm = np.concatenate(([self.lambda_ / c], 0.5 / c + np.zeros(2 * self.n)))
        Wc = np.concatenate(([self.lambda_ / c + (1 - self.alpha ** 2 + self.beta)], 0.5 / c + np.zeros(2 * self.n)))
        
        U = (self.n + self.lambda_) * self.P_k
        try:
            self.U = cholesky(U)
        except Exception as e:
            print('Issue with matrix decomposition: ', e)
            #TODO: here
            try:
                if not self.isPD(U): # is not positive definite Do something   
                    U = self.find_nearest_PSD(U)
                    self.U = cholesky(U)
            except Exception as e:
                print('PD ?', e)

            # self.P_k = np.diag([self.parameters.initial_obs_xy_error[0,0]]*4)
            # self.U = cholesky((self.n + self.lambda_) * self.P_k)

        # Calculate sigma points
        points = np.zeros((self.n, 2 * self.n + 1))
        points[:, 0] = self.hat_x_k[:, 0]
        for i in range(self.n):
            points[:, i + 1] = self.hat_x_k[:, 0] + self.U[i]
            points[:, self.n + i + 1] = self.hat_x_k[:, 0] - self.U[i]

        predicted_x = np.zeros((self.n, 1))
        all_predictions = self.system_equation_noniose(points)
        for i in range(2 * self.n + 1):
            predicted_x[:, 0] += all_predictions[:, i] * Wm[i]
        predicted_P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            predicted_error_XX = all_predictions[:, i].reshape(self.n, 1) - predicted_x
            predicted_P += np.matmul(predicted_error_XX, \
                np.transpose(predicted_error_XX)) * Wc[i] 
        predicted_P += self.Q


        if observation.current_receiver_error == None:
            observed_z = None

        # TODO: revert
        # if self.parameters.experiment_type == 'Combined_Dominica_Data':
        #     observed_z = self.extract_observed_value(observation)
        # else:
         
        if self.parameters.observation_type in ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA']:
            if observation.current_observed_AOA_candidate1 is None or (isinstance(observation.current_observed_AOA_candidate1, list) \
                and any([obs is None for obs in observation.current_observed_AOA_candidate1])) \
                    or np.isnan(observation.current_observed_AOA_candidate1):
                flag = False
            else:
                flag = True
                observed_z = np.array([observation.current_observed_AOA_candidate1, observation.current_observed_AOA_candidate2]).reshape(-1,1)
        else:
            observed_z = observation.current_observed_xy[0]
        
            flag = observed_z is not None and not any([ae is None for ae in observed_z.flatten() ])
        
        if flag:
            if self.parameters.observation_type in ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']:
                R = self.R_up if observation.current_whale_up else self.R_down

            elif observation.current_receiver_error is None or \
                (isinstance(observation.current_receiver_error, list) and any([obs is None for obs in observation.current_receiver_error]))\
                    or np.isnan(observation.current_receiver_error):
                R = self.R_up if observation.current_whale_up else self.R_down
            else:
                R = np.diag([receiver_error for receiver_error in observation.current_receiver_error])
            R_shape = R.shape[0]
            
            if self.parameters.observation_type in ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA']:
                predicted_z = np.zeros(shape =(R_shape, 1))
                all_observations = self.observation_function_noniose(points, observation)
                for i in range(2 * self.n + 1):
                    predicted_z += (all_observations[:, i] * Wm[i]).reshape((R_shape, 1))
                predicted_z.reshape(R_shape, 1)
                predicted_z = np.mod(predicted_z, 360)
                
                diff1 = angle_diff_degree(observed_z[0], predicted_z)
                diff2 = angle_diff_degree(observed_z[1], predicted_z)
                observation_error = diff1 if abs(diff1) < abs(diff2) else diff2
                self.selected_aoa = observed_z[0, 0] if abs(diff1) < abs(diff2) else observed_z[1, 0]


                error_cov_YY = np.zeros((R_shape, R_shape))
                for i in range(2 * self.n + 1):
                    error_YY = angle_diff_degree(all_observations[:, i], predicted_z[:,0]).reshape(R_shape, 1)
                    error_cov_YY += np.matmul(error_YY, np.transpose(error_YY)) * Wc[i]
                error_cov_YY += R

                error_cov_XY = np.zeros((self.n, R_shape))
                for i in range(2 * self.n + 1):
                    error_XX = (all_predictions[:, i] - predicted_x[:,0]).reshape(self.n, 1)
                    error_YY = angle_diff_degree(all_observations[:, i], predicted_z[:,0]).reshape(R_shape, 1)
                    error_cov_XY += np.matmul(error_XX, np.transpose(error_YY)) * Wc[i]
        
            else:
                predicted_z = np.zeros(shape =(R_shape, 1))
                all_observations = self.observation_function_noniose(points, observation)
                for i in range(2 * self.n + 1):
                    predicted_z += (all_observations[:, i] * Wm[i]).reshape((R_shape, 1))
                predicted_z.reshape(R_shape, 1)
                
                observation_error = np.array([observed_z[1] - predicted_z[1,0], observed_z[0] - predicted_z[0,0]]).reshape(R_shape, 1)
            
                error_cov_YY = np.zeros((R_shape, R_shape))
                test = []
                for i in range(2 * self.n + 1):
                    
                    error_YY = np.array([all_observations[1, i] - predicted_z[1, 0], \
                        all_observations[0, i] - predicted_z[0, 0]]).reshape(R_shape, 1)
                    test.append(np.matmul(error_YY, np.transpose(error_YY))[0,0])
    
                    error_cov_YY += np.matmul(error_YY, np.transpose(error_YY)) * Wc[i]
                error_cov_YY += R

                error_cov_XY = np.zeros((self.n, R_shape))
                for i in range(2 * self.n + 1):
                    error_XX = (all_predictions[:, i] - predicted_x[:,0]).reshape(self.n, 1)
                    error_YY = np.array([all_observations[1, i] - predicted_z[1, 0], \
                        all_observations[0, i] - predicted_z[0, 0]]).reshape(R_shape, 1)
                    
                    error_cov_XY += np.matmul(error_XX, np.transpose(error_YY)) * Wc[i]

            K = np.dot(error_cov_XY, np.linalg.inv(error_cov_YY)) # TODO: dot or matmul
            self.hat_x_k = predicted_x + np.matmul(K, observation_error)
            self.P_k = predicted_P - np.matmul(K, np.matmul(error_cov_YY, np.transpose(K)))  # when obs is too large P is non-positive definte

            if self.isPD(self.P_k): # is not positive definite Do something   
                self.P_k = self.find_nearest_PSD(self.P_k)
                # A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].
                # [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                # [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
                # matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

            if self.parameters.observation_type in ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA']:
                new_angle = self.observation_function_noniose(self.hat_x_k, observation)
                #TODO: revert
                # if self.parameters.experiment_type == 'Combined_Dominica_Data':
                #     self.post_estimation_observation_error = angle_diff_degree(observed_z, new_angle)[0,0]
                # else:
                self.post_estimation_observation_error = min(angle_diff_degree(observed_z[0], new_angle)[0,0], angle_diff_degree(observed_z[1], new_angle)[0,0])

            if self.dive_phase_data_collection_start == True:
                self.dive_phase_data_collection_start = False
                self.surface_behavior_ys = [self.hat_x_k[1, 0]]
                self.surface_behavior_xs = [self.hat_x_k[0, 0]]
                self.surface_behavior_ts = [self.time_step]
                self.calculated_theta_whale_up = False
            else:
                self.surface_behavior_ys.append(self.hat_x_k[1, 0])
                self.surface_behavior_xs.append(self.hat_x_k[0, 0])
                self.surface_behavior_ts.append(self.time_step)

        else:
            self.hat_x_k = predicted_x
            self.P_k = predicted_P

            if self.calculated_theta_whale_up == False:
                self.estimate_theta_from_surface_behavior(observation.current_whale_up)
            

        if np.isnan(self.P_k).any() or np.isnan(self.hat_x_k).any():
            print('UKF: state_estimation_AOA: np.isnan(self.P_k).any() or np.isnan(self.hat_x_k).any()')

        
        # return Loc_xy(x = self.hat_x_k[0,0], y = self.hat_x_k[1,0]), self.P_k
    
    def state_estimation_xy(self, observation: ObservationClass_whale):
        # return Loc_xy(x = self.hat_x_k[0,0], y = self.hat_x_k[1,0]), self.P_k
        NotImplemented







