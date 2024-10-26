from os import listdir
from os.path import isfile, join
import json, pickle, copy,csv, sys
from src.belief_state import Belief_State
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from src.rebuttal_runs.pretty_plot_dswp import plot_results_dswp, plot_results_engg_real, plot_miss_freq_results_engg_real
metric_result_keys = ['successful_opportunity','missed_opportunity', 'mission_time', 'whale_not_visited', 'time_bw_successive_rendezvous'] +\
    ['whale_missed_prob']




def read_metric_for_states(observation_type, num_agent, num_whale, tagging_radius, policy_name, output_base, date_combi):
    if expedition == 'Benchmark':
        suffix = '_' + observation_type+ '_r' + str(tagging_radius)  + '_w'+ str(num_whale) + '_a' + str(num_agent)
    else:
        suffix = '_r' + str(tagging_radius) + '_w'+ '_'.join(date_combi) + '_a' + str(num_agent)
    output_base_suffix = output_base + suffix + '/'
    runs = [f for f in listdir(output_base_suffix) \
        if not isfile(join(output_base_suffix, f)) and 'Run_' in f]

    metric_result = {metric_name:[] for metric_name in metric_result_keys}
    metric_result['whale_missed_prob'] = [0] * (num_whale+1)
    for run_id in runs:
        state_filename = output_base_suffix + run_id + '/'+ policy_name +'/state_log.csv'
        if expedition == 'Nov23':
            suffix2 = '_r' + str(tagging_radius) + '_w'+str(num_whale)+ '_'.join(date_combi) + '_a' + str(num_agent)
            config_filename = 'src/configs/rebuttal_runs/config_Dominica_Nov23' + suffix2 + '.json'
        elif expedition == 'Feb24':
            config_filename = 'src/configs/rebuttal_runs/config_Dominica_Feb24' + suffix + '.json'
        else:
            suffix2 = '_r' + str(tagging_radius) + '_' + observation_type + '_w'+str(num_whale) + '_a' + str(num_agent)
            config_filename = 'src/configs/rebuttal_runs/config_DSWP' + suffix2 + '.json'
        with open(config_filename, 'r') as file:
            config_obj = json.load(file)
        with open(state_filename, 'r') as f:
            lines = f.readlines()
            missed_opportunity_num = 0
            for lno, line in enumerate(lines):
                st = Belief_State(knowledge = config_obj, state_str = line)
                if lno == 0:
                    start_time = st.time
                    rendezvous = [st.w_last_surface_end_time[wid] if st.w_last_surface_end_time[wid] else 0 for wid in range(st.number_of_whales)]
                else:
                    missed_opportunity_num += np.sum([ 1 for wid in range(st.number_of_whales) \
                    if whale_prev_up[wid] and not st.whale_up2[wid] and \
                        wid not in st.assigned_whales])
                        
                    for wid in range(st.number_of_whales):
                        if wid not in previous_assigned and wid in st.assigned_whales:
                            rendezvous[wid] = (st.time - rendezvous[wid])/60
                if len(st.assigned_whales) == st.number_of_whales:
                    break
                whale_prev_up = st.whale_up2
                previous_assigned = copy.deepcopy(st.assigned_whales)
                previous_bxs = st.b_x
                previous_bys = st.b_y

            for wid in range(st.number_of_whales):
                if wid not in st.assigned_whales:
                    rendezvous[wid] = (st.time + 30 - rendezvous[wid])/60

            mission_time = (st.time - start_time)/60

            whale_not_visited = st.number_of_whales - len(st.assigned_whales)

            metric_result['whale_missed_prob'][whale_not_visited] += 1

            whale_not_visited_percentage = 100* whale_not_visited /st.number_of_whales

            missed_opportunity_denom = missed_opportunity_num + len(st.assigned_whales)
            missed_opportunity = 0 if (missed_opportunity_num == 0 and missed_opportunity_denom == 0) \
                else 100 * missed_opportunity_num / missed_opportunity_denom

            success_opportunity = 0 if (missed_opportunity_num == 0 and missed_opportunity_denom == 0) \
                else 100 * len(st.assigned_whales) / missed_opportunity_denom

            metric_result['mission_time'].append(mission_time)
            metric_result['missed_opportunity'].append(missed_opportunity)
            metric_result['successful_opportunity'].append(success_opportunity)
            metric_result['whale_not_visited'].append(whale_not_visited_percentage)
            for wid in range(st.number_of_whales):
                metric_result['time_bw_successive_rendezvous'].append(rendezvous[wid])
    line = observation_type +','+str(num_agent) + ',' + str(num_whale) + ','+ str(tagging_radius) + ','+ \
        ','.join([str(round(np.mean(mval),2)) + ',' + str(round(np.std(mval),2)) \
            for mkey, mval in metric_result.items() if mkey != 'whale_missed_prob']) + ',' + \
                ','.join([str(mv) for mv in metric_result['whale_missed_prob']])
    print(line)
    
    return metric_result, line




if __name__ == '__main__':
   
    
    expedition = ''
    if len(sys.argv) > 1:
        expedition = sys.argv[1]
    if expedition not in ['Benchmark', 'Feb24', 'Nov23']:
        expedition = 'Benchmark'

    policy_name = ''
    if len(sys.argv) > 2:
        policy_name = sys.argv[2]
    if policy_name not in ['MA_rollout', 'BLE', 'VRP_TW_no_replan_CE']:
        policy_name = 'MA_rollout'

    
    if expedition == 'Nov23':
        # num_agents = [2, 3]
        num_agents = [2,3]
        num_whales = [4,5]
        nagent_nwhales_combo = [[2,4], [2,5], [3,5]]
        # num_whales = [4, 5, 6]
        tagging_radii = [200, 300, 500]
        metric_output = 'output_Engineered_whale/' 
        output_base = metric_output + 'Combined_Dominica_Data'
        all_possible_dates = ["2023-11-21_trace0", "2023-11-21_trace1", "2023-11-22_trace2", "2023-11-22_trace3", "2023-11-23_trace5", "2023-11-23_trace4"]
        combinations_of_dates = {num_whale: [all_possible_dates[:num_whale]] for num_whale in num_whales}
        observation_types = ['Acoustic_AOA_VHF_AOA']
    elif expedition == 'Feb24':
        num_agents = [2,3]
        num_whales = [3,4]
        nagent_nwhales_combo = [[2,3], [2,4], [3,4]]
        tagging_radii = [500, 1000, 1500] 
        all_possible_dates = ["2024-02-29", "2024-03-01", "2024-03-02", "2024-03-04"]
        combinations_of_dates = {num_whale: list(combinations(all_possible_dates, num_whale)) for num_whale in num_whales}
        metric_output = 'output_sperm_whale/'
        output_base = metric_output + 'Feb24_Dominica_Data'
        observation_types = ['Acoustic_AOA_no_VHF']
    else:
        num_agents = [2]
        num_whales = [4]
        nagent_nwhales_combo = [[2,4]]
        tagging_radii = [200, 300, 500]
        combinations_of_dates = {num_whale: [num_whale] for num_whale in num_whales}
        metric_output = 'output_ablation_dswp/'
        output_base = metric_output + 'Benchmark_Shane_Data'
        observation_types = ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy', 'Acoustic_AOA_no_VHF','Acoustic_AOA_VHF_AOA'] 

    print(metric_output, output_base)
    

    
    all_metric = {mname: np.zeros((len(observation_types), len(num_agents), len(num_whales), \
        max([len(combinations_of_dates[wid]) for wid in num_whales]), \
        len(tagging_radii),2)) \
        for mname in metric_result_keys}
    all_metric_summarized = {}
    
    line = 'observation_type,num_agent,num_whale,dates,tagging_radius,' \
                + ','.join([mname + '_mean,'+ mname + '_std' for mname in metric_result_keys])
    print(line)
    all_metric_filename = expedition+'_'+policy_name+'_same_rejection_metrics'
    all_metric_summarized_filename = expedition+'_'+policy_name+'_same_rejection_summary_metrics'
    for no,observation_type in enumerate(observation_types):
        for na,num_agent in enumerate(num_agents):
            for nw,num_whale in enumerate(num_whales):
                if [num_agent, num_whale] not in nagent_nwhales_combo:
                    continue
                for date_combi_id, date_combi in enumerate(combinations_of_dates[num_whale]):
                    for nt, tagging_radius in enumerate(tagging_radii):
                        
                        metric, line = read_metric_for_states(observation_type, num_agent, num_whale, tagging_radius, policy_name, output_base, date_combi)
                        for mname in metric_result_keys:
                            if mname != 'whale_missed_prob':
                                all_metric[mname][no, na,nw, date_combi_id, nt,:] = [np.mean(metric[mname]),np.std(metric[mname])]
                        
                        for mname in metric_result_keys:
                            if (mname, observation_type, num_agent, num_whale, tagging_radius) not in all_metric_summarized.keys():
                                if mname != 'whale_missed_prob':
                                    all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)] = metric[mname]
                                else:
                                    all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)] = metric[mname]
                            else:
                                if mname != 'whale_missed_prob':
                                    all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)].extend(metric[mname])
                                else:
                                    all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)] = \
                                        [metric[mname][i] + all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)][i] \
                                            for i in range(num_whale+1)]
                               
    for dict_key in all_metric_summarized.keys():
        if dict_key[0] != 'whale_missed_prob':
            all_metric_summarized[dict_key] = [np.mean(all_metric_summarized[dict_key]), np.std(all_metric_summarized[dict_key])]
        else:
            all_metric_summarized[dict_key] = [all_metric_summarized[dict_key][i]/ sum(all_metric_summarized[dict_key]) \
                for i in range(dict_key[3]+1)]
                
    print()
    with open(metric_output + all_metric_filename, 'wb') as f:
        pickle.dump(all_metric, f)
        
    with open(metric_output + all_metric_summarized_filename, 'wb') as f:
        pickle.dump(all_metric_summarized, f)
        
    
    string_for_nice_print = ''
        
    metric_name_str = 'observation_type,num_agents,num_whales,tagging_radius,' + \
        ','.join([m + '_mean,' + m +'_std' for m in ['successful_opportunity','mission_time']]) + ','+\
            'missing_1_or_more_whales_prob,' +  ','.join(['missed_'+str(w)+'whales_prob' for w in range(max(num_whales)+1)])
    print('\n'+metric_name_str)
    string_for_nice_print += metric_name_str + '\n'

    for no, observation_type in enumerate(observation_types):
        for na,num_agent in enumerate(num_agents):
            for nw,num_whale in enumerate(num_whales):
                if num_whale <= num_agent or (expedition == 'Nov23' and num_whale == 4 and num_agent != 2):
                    continue
                for nt, tagging_radius in enumerate(tagging_radii):
                
                    line =observation_type+ ','+str(num_agent) + ',' + str(num_whale) + ',' + str(tagging_radius)
                    for mname in ['successful_opportunity','mission_time']:
                        m = all_metric_summarized[(mname, observation_type, num_agent, num_whale, tagging_radius)]
                        line += ',' + str(m[0]) + ',' + str(m[1])
                    ms = all_metric_summarized[('whale_missed_prob', observation_type, num_agent, num_whale, tagging_radius)]
                    line += ',' + str(1-ms[0])
                    line += ',' + ','.join([str(m) for m in ms])
                    print(line)
                    string_for_nice_print += line + '\n'

    run_filename = 'src/rebuttal_runs/'+expedition+'_runs.csv'
    with open(run_filename, 'w') as run_file:
        run_file.write(string_for_nice_print)
    if expedition == 'Benchmark':
        plot_results_dswp(run_filename, metric_name='successful_opportunity', output_foldername = metric_output)
        plot_results_dswp(run_filename, metric_name='mission_time', output_foldername = metric_output)
    
    elif expedition in ['Nov23', 'Feb24']:
        plot_results_engg_real(run_filename, metric_name='successful_opportunity', output_foldername = metric_output)
        plot_results_engg_real(run_filename, metric_name='mission_time', output_foldername = metric_output)
        plot_miss_freq_results_engg_real(run_filename, output_foldername = metric_output)

    
      
