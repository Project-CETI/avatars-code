from src.global_knowledge import Global_knowledge
from src.configs.constants import *
from src.state_estimation_filter import State_Estimation_Filter
from src.system_state import System_state
from src.system_observation import ObservationClass 
from src.evaluate_localization_error import EvalObjectClass
from src.belief_state import Belief_State, Policy_base
from src.policies.rollout_policy_science import MA_Rollout_science
from policies.vrp_tw import VRP_TW
from policies.ble2_auction_based import BLE2
from src.policies.ble import BLE
import json, os, pickle, traceback, multiprocessing, time, sys
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

def get_ground_truth(parameter: Global_knowledge, run_id) -> System_state:
    gt_prefix, gt_path = parameter.get_system_gt_txyz_fnPrefix_path(run_id)
    gt_filenames = [f for f in listdir(gt_path) if isfile(join(gt_path, f)) and gt_prefix.split('/')[-1] in f] if os.path.exists(gt_path) \
        else []
    #if ground truth already exists do not recreate to make the comparison fair for num_w, num_b and exp_type (simu/exp_data)
    if len(gt_filenames) == 0:
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        gt = System_state(parameters = parameter, run_id = run_id) 
        # Set whale locs, surfacetimes in terms of planning time may include -ve if whales have previously surfaced where Filter should start
        # if parameter.experiment_type == 'Benchmark_Shane_Data':
        #     gt.use_dswp_data() # whale_locs, surfacings, bxy0
        # elif parameter.experiment_type == 'Combined_Dominica_Data':
        #     # gt.data_from_dominica_experiment_Nov23() 
        #     print('ERROR')
        #     exit()
        f = open(gt_prefix + '-'.join(map(str, gt.file_names)), 'wb')
        pickle.dump(gt, f)
        f.close()
    else:
        f = open(gt_path + gt_filenames[0], 'rb')
        gt = pickle.load(f)
        f.close()
    return gt

def get_observation(gt: System_state) -> ObservationClass:
    raw_obs_prefix, raw_obs_path = gt.parameters.get_rawObservation_filterInput_fnPrefix_path(gt.run_id)
    ro_filenames = [f for f in listdir(raw_obs_path) if isfile(join(raw_obs_path, f)) and raw_obs_prefix.split('/')[-1] in f \
        and '-'.join(map(str, gt.file_names)) in f] if os.path.exists(raw_obs_path) else []
    if len(ro_filenames) == 0:
        if not os.path.exists(raw_obs_path):
            os.makedirs(raw_obs_path)
        raw_obs = ObservationClass(gt = gt) 
        # Initialize obs for t0
        # combined exp change parameters (mean times) from the estimated surfacing behavior. changes "parameter" in combined_experiment
        f = open(raw_obs_prefix + '-'.join(map(str, gt.file_names)), 'wb')
        pickle.dump(raw_obs, f)
        f.close()
    else:
        f = open(raw_obs_path + ro_filenames[0], 'rb')
        raw_obs = pickle.load(f)
        f.close()
    return raw_obs

def get_filter(raw_obs: ObservationClass) -> State_Estimation_Filter:
    state_estm_prefix, state_estm_path = raw_obs.parameters.get_stateEstimation_filterOutput_txyz_fnPrefix_path(raw_obs.run_id)
    se_filenames = [f for f in listdir(state_estm_path) if isfile(join(state_estm_path, f)) and state_estm_prefix.split('/')[-1] in f \
        and '-'.join(map(str, raw_obs.file_names)) in f] if os.path.exists(state_estm_path) else []

    if len(se_filenames) == 0:
        if not os.path.exists(state_estm_path):
            os.makedirs(state_estm_path)
        filter_ = State_Estimation_Filter(observation = raw_obs) # This should just be the whale_locs_t0, whale_ups
        f = open(state_estm_prefix + '-'.join(map(str, raw_obs.file_names)), 'wb')
        pickle.dump(filter_, f)
        f.close()
    else:
        f = open(state_estm_path + se_filenames[0], 'rb')
        filter_ = pickle.load(f)
        f.close()
    return filter_

def set_random_seed(parameter: Global_knowledge, run_id = 0, seed = None):
    gt_path = parameter.get_system_seed_txyz_fnPrefix_path(run_id)
        
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    seed_filename = join(gt_path, 'seed.txt')
    if isfile(seed_filename):
        with open(seed_filename, 'r') as s_f:
            seed_for_run_id = int(s_f.readlines()[0].strip('\n'))
    else:
        if seed is None:
            seed_for_run_id = (os.getpid() * int(time.time())) % 123456789
        else:
            seed_for_run_id = seed
        with open(seed_filename, 'w') as s_f:
            s_f.write(str(seed_for_run_id))
    np.random.seed(seed_for_run_id)

def run_thread(parameter: Global_knowledge, run_id = 0, debug = False, dont_run_policy = False, seed = None, rollout_time_dist = False, policy_name = None):
    
    try:
        set_random_seed(parameter, run_id, seed)
        gt = System_state(parameters = parameter, run_id = run_id) 
        # gt : System_state = get_ground_truth(parameter, run_id)
        evalObject = EvalObjectClass(gt = gt)
        # raw_obs: ObservationClass = get_observation(gt)
        raw_obs = ObservationClass(gt = gt)
        # localization_filter: State_Estimation_Filter = get_filter(raw_obs)
        localization_filter = State_Estimation_Filter(observation = raw_obs) # This should just be the whale_locs_t0, whale_ups
        belief_state = localization_filter.belief_state0
        
        if policy_name is None:
            policy = MA_Rollout_science(knowledge = parameter, state = belief_state, \
                CE = False, commitment = False, direction = False, rollout_time_dist = rollout_time_dist)
            # policy = BLE(knowledge = parameter, state = belief_state)
            # policy = VRP_TW(knowledge=parameter, state = belief_state)
            # policy = Policy_base()
        else:
            if policy_name == 'MA_rollout':
                policy = MA_Rollout_science(knowledge = parameter, state = belief_state, \
                    CE = False, commitment = False, direction = False, rollout_time_dist = rollout_time_dist)
            elif policy_name == 'BLE':
                policy = BLE(knowledge = parameter, state = belief_state)
            elif policy_name == 'VRP_TW_no_replan_CE':
                policy = VRP_TW(knowledge=parameter, state = belief_state)
            else:
                print('Provide a valid policy_name')
                exit()
        folder_ = parameter.base_output_path + 'Run_' + str(run_id) + '/' + str(policy.policy_name) + '/'
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        log_filename = folder_ + 'state_log.csv'
    
        log_file = open(log_filename, "w")


        # observations_x_y_v_theta_up, Pcov = localization_filter.get_next_estimation(observation = raw_obs)
        # evalObject.update_logs(raw_obs, observations_x_y_v_theta_up, Pcov, plot = debug, folder_ = folder_)
        

        log_file.write(belief_state.state_string())
        log_file.flush()

        if run_id == -1 or debug == True:
            belief_state.plot_state(path = folder_)

        for time_step in range(int(parameter.n_horizon_for_evaluation/parameter.observations_per_minute) - 1):
        
            
            
            if 1==1:
                for sec in range(parameter.observations_per_minute):
                    # scaled_control = Boat_Control(b_theta = control.b_theta, b_v = control.b_v / parameter.observations_per_minute)
                    scaled_control = policy.get_control(belief_state)
                    gt.step(scaled_control)
                    
                    b_xys = np.array([convert_longlat_to_xy_in_meters(gt.current_agent_xs[bid], gt.current_agent_ys[bid]) \
                        for bid in range(parameter.number_of_agents)])
                    c_b_x = np.array([b_xy[0] for b_xy in b_xys])
                    c_b_y = np.array([b_xy[1] for b_xy in b_xys])


                    
                    
                    raw_obs.set_observation_t(gt, time_step * parameter.observations_per_minute + sec + 1)
                    observations_x_y_v_theta_up, Pcov = localization_filter.get_next_estimation(observation = raw_obs)

                    
                    if parameter.experiment_type in ['Benchmark_Shane_Data','Feb24_Dominica_Data', 'Combined_Dominica_Data']:
                        g_w_xys = {wid : None for wid in range(parameter.number_of_whales)}
                        for wid in range(parameter.number_of_whales):
                            if raw_obs.gt_for_eval[wid][0] is None:
                                raw_obs.gt_for_eval[wid, 0] = observations_x_y_v_theta_up[wid, 5]
                                raw_obs.gt_for_eval[wid, 1] = observations_x_y_v_theta_up[wid, 6]
                                

                                # for wid in range(parameter.number_of_whales):
                                if gt.current_whale_xs[wid] is None:
                                    g_w_xys[wid] = None
                                    continue
                                g_w_xys[wid] = (observations_x_y_v_theta_up[wid, 0], observations_x_y_v_theta_up[wid,1])

                                gt.current_whale_assigned[wid] = gt.current_whale_assigned[wid] or (gt.current_whale_up[wid] and \
                                    any([get_distance_from_latLon_to_meter(gt.current_agent_ys[bid], gt.current_agent_xs[bid], \
                                        raw_obs.gt_for_eval[wid, 1], raw_obs.gt_for_eval[wid, 0]) <= parameter.tagging_distance \
                                            for bid in range(parameter.number_of_agents)]))
                                raw_obs.current_whale_assigned[wid] = gt.current_whale_assigned[wid]



                            else:
                                g_w_xys[wid] = convert_longlat_to_xy_in_meters(raw_obs.gt_for_eval[wid][0], raw_obs.gt_for_eval[wid][1])
                    else:
                        g_w_xys = {wid : convert_longlat_to_xy_in_meters(raw_obs.gt_for_eval[wid][0], raw_obs.gt_for_eval[wid][1]) \
                            for wid in range(parameter.number_of_whales)}

                    evalObject.update_logs(raw_obs, observations_x_y_v_theta_up, Pcov, plot = debug if sec == 0 else False, folder_ = folder_)

                    
                    # prev_assigned = len(belief_state.assigned_whales)
                    belief_state.next_state(scaled_control, \
                        observations_x_y_v_theta_up = observations_x_y_v_theta_up, Pcov = Pcov, \
                            ground_truth_for_evaluation = g_w_xys, \
                                b_xys = [c_b_x, c_b_y], w_assigned = raw_obs.current_whale_assigned)
                            
                    
                    # if prev_assigned != len(belief_state.assigned_whales):
                    #     print('h')
                    
                    # if sec == 0 and (run_id == -1 or debug == True):
                    #     belief_state.plot_state(path = folder_)
                    
                    
                    log_file.write(belief_state.state_string())
                    log_file.flush()
                    if raw_obs.terminal():
                        # print('terminal', raw_obs.current_whale_assigned)
                        break
                if run_id == -1 or debug == True:
                    belief_state.plot_state(path = folder_)
                if raw_obs.terminal():
                    # print('terminal', raw_obs.current_whale_assigned)
                    break
                    # gt.visualize_whales(folder_ = folder_, time_start_end = [0, (time_step + 1) * parameter.observations_per_minute ])
            # if raw_obs.current_time >=59:
            #     print('')

            
            

        log_file.close()

        loc_error_filename = folder_ + 'loc_error.csv'

        for wid in range(parameter.number_of_whales):
            plt.scatter(np.arange(len(evalObject.localization_error[wid])), evalObject.localization_error[wid])
            mean_err = round(np.mean(evalObject.localization_error[wid]),2)
            std_err = round(np.std(evalObject.localization_error[wid]),2)
            plt.title('Localization error mean:' + str(mean_err) + ' std:' +str(std_err))
            plt.savefig(folder_ + 'loc_error'+str(wid)+'.png')
            plt.close()
        with open(loc_error_filename, "w") as loc_error_file:
            allw_loc_error = sum([evalObject.localization_error[wid] for wid in range(parameter.number_of_whales)], [])
            loc_error_file.write(str(np.mean(allw_loc_error)) +','+ str(np.std(allw_loc_error))+','+ str(len(allw_loc_error))+'\n')


    except Exception as e:
        print('Exception: ', e, 'run_id: ', run_id)
        full_traceback = traceback.extract_tb(e.__traceback__)
        filename, lineno, funcname, text = full_traceback[-1]
        for traceback_ in full_traceback:
            print(traceback_)
        print("Error in ", filename, " line:", lineno, " func:", funcname, ", exception:", e)
        

if __name__ == '__main__':
    if len(sys.argv) > 1:
        f = open(sys.argv[1], "r")
    else:
        # f = open("src/configs/config_Dominica_Feb24.json", "r")
        # f = open("src/configs/config_Dominica_Nov23.json", "r")
        f = open("src/configs/config_Benchmark.json", "r")
        # f = open('src/configs/rebuttal_runs/config_DSWP_r500_Acoustic_xy_VHF_xy_w4_a2.json')
    
    policy_name = None
    if len(sys.argv) > 2:
        policy_name = sys.argv[2]

    knowledge = Global_knowledge()
    knowledge.set_attr(json.load(f))

    if len(sys.argv) == 4:
        num_processes = knowledge.average_num_runs
        pool = multiprocessing.Pool(num_processes)
        # parameter, run_id, debug, dont_run_policy, seed, rollout_time_dist
        processes = [pool.apply_async(run_thread, args = (knowledge, run_id, False, False, None, False, policy_name)) \
            for run_id in range(num_processes)]
        results = [p.get() for p in processes]
        x = results
        pool.close()
        pool.join()
    else:
        # for i in range(100):
        # seed_for_run_id = (os.getpid() * int(time.time())) % 123456789
        seed_for_run_id = None
        run_thread(parameter = knowledge, run_id = 0, debug = False, dont_run_policy = False, seed = seed_for_run_id, \
            rollout_time_dist = False)


