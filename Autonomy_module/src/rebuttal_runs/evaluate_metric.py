from os import listdir
from os.path import isfile, join
import json, pickle, copy,csv, sys
from src.belief_state import Belief_State
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
metric_result_keys = ['successful_opportunity','missed_opportunity', 'mission_time', 'whale_not_visited', 'time_bw_successive_rendezvous'] +\
    ['whale_missed_prob']


def plot_run_id(num_agent, num_whale, tagging_radius, policy_name, output_base, run_id, date_combi):
    # suffix = '_r' + str(tagging_radius) + '_w' + str(num_whale) + '_a' + str(num_agent)# + '_i' + str(run_id)
    if expedition == 'Benchmark':
        suffix = '_' + observation_type+ '_r' + str(tagging_radius)  + '_w'+ str(num_whale) + '_a' + str(num_agent)
    else:
        suffix = '_r' + str(tagging_radius) + '_w' + '_'.join(date_combi) + '_a' + str(num_agent)
    # suffix2 = suffix + '_i' + str(run_id)
    output_base_suffix = output_base + suffix + '/'
    state_filename = output_base_suffix + 'Run_' + str(run_id) + '/' + policy_name + '/state_log.csv'
    png_filename = output_base_suffix + 'Run_' + str(run_id) + '/' + policy_name + '/states.png'


    if expedition == 'Nov23':
        suffix2 = '_r' + str(tagging_radius) + '_w'+str(num_whale)+ '_'.join(date_combi) + '_a' + str(num_agent)
        config_filename = 'src/configs/rebuttal_runs/config_Dominica_Nov23' + suffix2 + '.json'
    elif expedition == 'Feb24':
        config_filename = 'src/configs/rebuttal_runs/config_Dominica_Feb24' + suffix + '.json'
    else:
        suffix2 = '_r' + str(tagging_radius) + '_' + observation_type + '_w'+str(num_whale) + '_a' + str(num_agent)
        config_filename = 'src/configs/rebuttal_runs/config_DSWP' + suffix2 + '.json'

    # config_filename = 'src/configs/rebuttal_runs/config_Dominica_Feb24' + suffix + '.json'
    # config_filename = 'src/configs/rebuttal_runs/config_Dominica_Nov23' + suffix + '.json'
    with open(config_filename, 'r') as file:
        config_obj = json.load(file)

    history = {'w': {wid : {'x': [], 'y':[], 'up': []} for wid in range(num_whale)} , \
        'a' : {aid : {'x': [], 'y':[]} for aid in range(num_agent)}}
    
    with open(state_filename, 'r') as f:
        lines = f.readlines()
        for lno, line in enumerate(lines):
            st = Belief_State(knowledge = config_obj, state_str = line)
            for wid in range(num_whale):
                history['w'][wid]['x'].append(st.w_x[wid])
                history['w'][wid]['y'].append(st.w_y[wid])
                history['w'][wid]['up'].append([1,0,0] if wid in st.assigned_whales \
                    else [0,0,1] if st.whale_up2[wid] else [0,0,0])
            for bid in range(num_agent):
                history['a'][bid]['x'].append(st.b_x[bid])
                history['a'][bid]['y'].append(st.b_y[bid])

            if len(st.assigned_whales) == st.number_of_whales:
                break
        for wid in range(num_whale):
            plt.scatter(history['w'][wid]['x'], history['w'][wid]['y'], c = history['w'][wid]['up'], s = 2)
            plt.scatter(history['w'][wid]['x'][0], history['w'][wid]['y'][0], s = 20)
        for bid in range(num_agent):
            
            plt.scatter(history['a'][bid]['x'][:], history['a'][bid]['y'][:], label = 'bid_'+str(bid), s =0.1)
            plt.scatter(history['a'][bid]['x'][0], history['a'][bid]['y'][0], s = 20, label = 'bid0_'+str(bid))
        # print(history['a'][0]['x'][:10], history['a'][0]['x'][0])

        plt.legend()
        print(png_filename)
        plt.savefig(png_filename)
        # plt.savefig(output_base_suffix + 'Run_' + str(run_id) + '/MA_rollout/' + suffix2 + ".png")
        plt.close()


def read_metric_for_states(observation_type, num_agent, num_whale, tagging_radius, policy_name, output_base, date_combi):
    # suffix = '_r' + str(tagging_radius) + '_w' + str(num_whale) + '_a' + str(num_agent)
    # if expedition == 'Nov23':
    #     suffix = '_r' + str(tagging_radius) + '_w'+ str(num_whale) + '_'.join(date_combi) + '_a' + str(num_agent)
    # else:
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
                        # start_time <= st.w_last_surface_start_time[wid] and st.w_last_surface_end_time[wid] < st.time \
                        #     and wid not in st.assigned_whales])
                    # distance_moved += np.sum([np.sqrt((previous_bxs[bid] - st.b_x[bid])**2 + (previous_bys[bid] - st.b_y[bid])**2)/1000 \
                    #     for bid in range(st.number_of_boats)])
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
    # with("rebuttal_output_Feb24/results.csv", "w") as f:
        
    #     f.write(line + '\n')
    return metric_result, line




if __name__ == '__main__':#evalmetric':
    # policy_name = 'MA_rollout_time_dist'
    
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
        output_base = metric_output + 'Combined_Dominica_Data'#'Nov23_Dominica_Data'
        all_possible_dates = ["2023-11-21_trace0", "2023-11-21_trace1", "2023-11-22_trace2", "2023-11-22_trace3", "2023-11-23_trace5", "2023-11-23_trace4"]
        combinations_of_dates = {num_whale: [all_possible_dates[:num_whale]] for num_whale in num_whales}
        observation_types = ['Acoustic_AOA_VHF_AOA']
    elif expedition == 'Feb24':
        num_agents = [2,3]
        num_whales = [3,4]
        nagent_nwhales_combo = [[2,3], [2,4], [3,4]]
        tagging_radii = [500, 1000, 1500] #[1500, 1000, 500 ]
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
        observation_types = ['Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy', 'Acoustic_AOA_no_VHF','Acoustic_AOA_VHF_AOA'] #,'Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']

    print(metric_output, output_base)
    
    # for no,observation_type in enumerate(observation_types):
    #     for na,num_agent in enumerate(num_agents):
    #         for nw,num_whale in enumerate(num_whales):
    #             if num_whale <= num_agent:
    #                 continue
    #             for date_combi_id, date_combi in enumerate(combinations_of_dates[num_whale]):
    #                 for nt, tagging_radius in enumerate(tagging_radii):
    #                     for run_id in range(50):
    #                         plot_run_id(num_agent = num_agent, num_whale = num_whale, tagging_radius = tagging_radius, policy_name = policy_name, \
    #                             output_base = output_base, run_id = run_id, date_combi = date_combi)
    # exit()
    
    
    
        
    

    
    all_metric = {mname: np.zeros((len(observation_types), len(num_agents), len(num_whales), \
        max([len(combinations_of_dates[wid]) for wid in num_whales]), \
        len(tagging_radii),2)) \
        for mname in metric_result_keys}
    all_metric_summarized = {}
    # all_metric_summarized = {mname: np.zeros((len(num_agents), len(num_whales), \
    #     num_whales, \
    #     len(tagging_radii),2)) \
    #     for mname in metric_result_keys}
    line = 'observation_type,num_agent,num_whale,dates,tagging_radius,' \
                + ','.join([mname + '_mean,'+ mname + '_std' for mname in metric_result_keys])
    print(line)
    all_metric_filename = expedition+'_'+policy_name+'_same_rejection_metrics'
    all_metric_summarized_filename = expedition+'_'+policy_name+'_same_rejection_summary_metrics'
    if not isfile(metric_output + all_metric_filename) or not isfile(metric_output + all_metric_summarized_filename):
        for no,observation_type in enumerate(observation_types):
            for na,num_agent in enumerate(num_agents):
                for nw,num_whale in enumerate(num_whales):
                    # if num_whale <= num_agent or (expedition == 'Nov23' and num_whale == 4 and num_agent != 2):
                    if [num_agent, num_whale] not in nagent_nwhales_combo:
                        continue
                    for date_combi_id, date_combi in enumerate(combinations_of_dates[num_whale]):
                        for nt, tagging_radius in enumerate(tagging_radii):
                            # if num_whale == 4 and num_agent != 2:
                            #     continue
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
                                # if mname == 'whale_missed_prob':
                                #     print(all_metric_summarized[(mname, num_agent, num_whale, tagging_radius)])
        for dict_key in all_metric_summarized.keys():
            if dict_key[0] != 'whale_missed_prob':
                all_metric_summarized[dict_key] = [np.mean(all_metric_summarized[dict_key]), np.std(all_metric_summarized[dict_key])]
            else:
                all_metric_summarized[dict_key] = [all_metric_summarized[dict_key][i]/ sum(all_metric_summarized[dict_key]) \
                    for i in range(dict_key[3]+1)]
                # print(all_metric_summarized[dict_key], sum(all_metric_summarized[dict_key]), all_metric_summarized[dict_key][0])
        print()
        with open(metric_output + all_metric_filename, 'wb') as f:
            pickle.dump(all_metric, f)
        
        with open(metric_output + all_metric_summarized_filename, 'wb') as f:
            pickle.dump(all_metric_summarized, f)
        
            

    else:
        with open(metric_output + all_metric_filename, 'rb') as f:
            
            all_metric = pickle.load(f)
            for no,observation_type in enumerate(observation_types):
                for na,num_agent in enumerate(num_agents):
                    for nw,num_whale in enumerate(num_whales):
                        if num_whale <= num_agent or (expedition == 'Nov23' and num_whale == 4 and num_agent != 2):
                            continue
                        for date_combi_id, date_combi in enumerate(combinations_of_dates[num_whale]):
                            for nt, tagging_radius in enumerate(tagging_radii):
                                line =observation_type + ','+ str(num_agent) + ',' + str(num_whale) + ','+ str(tagging_radius) + ','+\
                                    ','.join([str(round(all_metric[mname][no, na,nw, date_combi_id, nt,0],2)) \
                                        + ',' + str(round(all_metric[mname][no,na,nw, date_combi_id, nt,1],2)) \
                                            for mname in metric_result_keys if mname != 'whale_missed_prob'])
                                print(line)

        with open(metric_output + all_metric_summarized_filename, 'rb') as f:
            all_metric_summarized = pickle.load(f)
    

    
    def plot_summary():
        metrics_to_plot = ['successful_opportunity','mission_time', 'whale_missed_prob']
        
        metric_name_str = 'observation,num_agents,num_whales,radius,' + \
            ','.join([m + '_mean,' + m +'_std' for m in ['successful_opportunity','mission_time']]) + ','+\
                'missing_1_or_more_whales_prob,' +  ','.join(['missed_'+str(w)+'whales_prob' for w in range(max(num_whales)+1)])
        print('\n'+metric_name_str)
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
                        # print(ms)
                        line += ',' + str(1-ms[0])
                        line += ',' + ','.join([str(m) for m in ms])
                        print(line)

        for no, observation_type in enumerate(observation_types):
            for mname in metrics_to_plot:
                xticks = np.unique(['a' + str(mkey[2]) + '_w' + str(mkey[3]) for mkey in all_metric_summarized.keys()])
                min_y = 0
                max_y = None
                if mname not in metrics_to_plot:
                    continue
                elif mname == 'successful_opportunity':
                    max_y = 100
                elif mname == 'whale_missed_prob':
                    max_y = 2
            
                fig = plt.figure()
                ax = fig.add_subplot(111)    
                width = 0.1
                for nt, tagging_radius in enumerate(tagging_radii):
               
                
                    if mname in ['successful_opportunity','mission_time']:
                        m_mean = [mval[0] for mkey, mval in all_metric_summarized.items() if mkey[0]==mname and mkey[3]==tagging_radius] 
                        m_std = [mval[1] for mkey, mval in all_metric_summarized.items() if mkey[0]==mname and mkey[3]==tagging_radius] 
                    else:
                        m_mean = [sum(mval[1:]) for mkey, mval in all_metric_summarized.items() if mkey[0]==mname and mkey[3]==tagging_radius] 
                        m_std = None
                
                    if m_std is not None:
                        bars = ax.bar(np.arange(len(m_mean)) + width * nt, m_mean, width, yerr = m_std, label = 'r_'+str(tagging_radius))
                    else:
                        bars = ax.bar(np.arange(len(m_mean)) + width * nt, m_mean, width, label = 'r_'+str(tagging_radius))
                    # print(m_std)
                    for bid, bar in enumerate(bars):
                        height = bar.get_height()
                        if m_std is not None:
                        
                            height += m_std[bid]
                            ax.text(bar.get_x() + bar.get_width() / 2, height, str(round(m_mean[bid],1)) + r"$\pm$" + str(round(m_std[bid],1)), ha='center', va='bottom', rotation = 45)
                        else:
                            print('whale_missed_prob', m_mean[bid])
                            ax.text(bar.get_x() + bar.get_width() / 2, height, str(round(m_mean[bid],3)), ha='center', va='bottom', rotation = 45)

                        if mname == 'mission_time':
                            min_y = min(min_y, height)
                            max_y = max(max_y, height) if max_y is not None else height
        
        
            
                ax.set_xticks(np.arange(len(xticks)))
                ax.set_xticklabels(xticks)
            
                ax.set_ylim(min_y, max_y)
                plt.legend(ncol = len(tagging_radii))
                plt.title(mname+' '+observation_type)
                plt.savefig(metric_output + "results_summarized_" + observation_type + '_' + mname + ".png")
                plt.close()

    plot_summary()