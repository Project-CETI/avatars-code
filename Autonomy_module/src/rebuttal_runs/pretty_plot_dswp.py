from os import listdir
from os.path import isfile, join
import os,json, copy, sys
import numpy as np
from src.belief_state import Belief_State
import traceback
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import inflect
import matplotlib as mpl


def print_dswp_results(policy_names = ['MA_rollout', 'BLE', 'VRP_TW_no_replan_CE']):
    try:
        line = ""
        metric_output = 'rebuttal2_output_dswp_Sep29/'
        output_base = metric_output + 'Benchmark_Shane_Data'
        observation_types = ['Acoustic_AOA_no_VHF', 'Acoustic_AOA_VHF_AOA', 'Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']
        num_whale = 4
        num_agent = 2
        # policy_names = ['MA_rollout'] #['MA_rollout', 'BLE', 'VRP_TW_no_replan_CE']
        with open('src/configs/config_Benchmark.json', 'r') as file:
            config_obj = json.load(file)
        metric_result_keys = ['successful_opportunity','missed_opportunity', 'mission_time', 'whale_not_visited', 'time_bw_successive_rendezvous'] +\
            ['whale_missed_prob']
        metric_result = {metric_name:[] for metric_name in metric_result_keys}
        metric_result['whale_missed_prob'] = [0] * (num_whale+1)

        print('policy_name,observation_type,num_agent,num_whale,tagging_radius,' + ','.join([mkey+'_mean,' + mkey + '_std' for mkey, mval in metric_result.items() if mkey !=   'whale_missed_prob']) + ',' + \
            ','.join(['whale_missed_prob_' + str(mk) for mk,mv in enumerate(metric_result['whale_missed_prob']) ]))
    
        for policy_name in policy_names:
            for observation_type in observation_types:
                for tagging_radius in [200, 300, 500]:
                    suffix = '_' + observation_type+ '_r' + str(tagging_radius)  + '_w'+ str(num_whale) + '_a' + str(num_agent)
                    output_base_suffix = output_base + suffix + '/'
                    if not os.path.exists(output_base_suffix):
                        print(output_base_suffix + ' does not exists')
                        continue
                    runs = [f for f in listdir(output_base_suffix) \
                        if not isfile(join(output_base_suffix, f)) and 'Run_' in f]
                
                    state_filenames = [output_base_suffix + run_id + '/'+ policy_name +'/state_log.csv' for run_id in runs]
                    if any([not os.path.exists(state_filename) for state_filename in state_filenames]):
                        print('At least one file of combination ', policy_name, observation_type, 
                        tagging_radius, ' does not exists')
                        continue

                    metric_result = {metric_name:[] for metric_name in metric_result_keys}
                    metric_result['whale_missed_prob'] = [0] * (num_whale+1)
                        
                    for run_id in runs:
                        state_filename = output_base_suffix + run_id + '/'+ policy_name +'/state_log.csv'
                        if not os.path.exists(state_filename):
                            # print(state_filename + ' does not exists')
                            continue
                        previous_assigned = []
                        with open(state_filename, 'r') as f:
                            lines = f.readlines()
                            missed_opportunity_num = 0
                            for lno, line in enumerate(lines):
                                try:
                                    st = Belief_State(knowledge = config_obj, state_str = line)
                                except Exception as e:
                                    print(line, ' has an exception')
                                    print(e)
                                    full_traceback = traceback.extract_tb(e.__traceback__)
                                    print(full_traceback)
                                    # st = Belief_State(knowledge = config_obj, state_str = line, debug = True)
                                    break
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
                                whale_prev_up = copy.deepcopy(st.whale_up2)
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
                    line = policy_name + ','+ observation_type +','+str(num_agent) + ',' + str(num_whale) + ','+ str(tagging_radius) + ','+ \
                        ','.join([str(round(np.mean(mval),2)) + ',' + str(round(np.std(mval),2)) \
                            for mkey, mval in metric_result.items() if mkey != 'whale_missed_prob']) + ',' + \
                                ','.join([str(mv) for mv in metric_result['whale_missed_prob']])
                    print(line)

    except Exception as e:
        print(e)
        full_traceback = traceback.extract_tb(e.__traceback__)
        print(full_traceback)
        print(line)




def plot_results_dswp(filename, metric_name = 'successful_opportunity', color_palette = 'rocket_r'):
    
    color_palette = sns.color_palette('flare', 3)
    

    df = pd.read_csv(filename,  header=0)
    # print(df)
    custom_order = ['Acoustic_AOA_no_VHF','Acoustic_AOA_VHF_AOA', 'Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']
    df['observation_type'] = pd.Categorical(df['observation_type'], categories=custom_order, ordered=True)
    
    
    df['observation_type_spaced'] = df['observation_type'].replace({
        'Acoustic_AOA_no_VHF': 'Acoustic\nAOA',
        'Acoustic_AOA_VHF_AOA': 'Acoustic +\nVHF AOA',
        'Acoustic_xy_no_VHF': 'Acoustic\npositioning',
        'Acoustic_xy_VHF_xy': 'Acoustic\npositioning + GPS'
    })

    policy_names = df['policy_name'].unique()
    for policy_name in policy_names:
        df_policy = df[df['policy_name']==policy_name]
        df_policy.reset_index(drop=True, inplace=True)
        # print(df_policy.to_string())
        # exit()
        plt.figure(figsize=(24, 8))
        print(metric_name + '_mean', df_policy[ metric_name + '_mean'].values)
        print(metric_name + '_std' , df_policy[ metric_name + '_std'].values)
        ax = sns.barplot(
            x='observation_type_spaced',  # X-axis (grouped by observation type)
            y= metric_name + '_mean',  # Y-axis (the values we want to plot)
            hue='tagging_radius',  # Grouping based on tagging_radius
            data=df_policy,
            palette=color_palette  # Set a color palette
        )
        if 1==2:
            for container in ax.containers:
                ax.bar_label(container)
        xy_coords = {p.get_x() + 0.5 * p.get_width():p.get_y() + p.get_height() for p in ax.patches if p.get_height() !=0}
        xy_coords = dict(sorted(xy_coords.items()))
        df_policy = df_policy.sort_values(by =['observation_type', 'tagging_radius'])
        # print('x_coords:', x_coords, ', y_coords:', y_coords)
        for i, row in df_policy.iterrows():
            # print(i)
            print(i, list(xy_coords.keys())[i], list(xy_coords.values())[i], row[metric_name + '_mean'], row[metric_name + '_std'])
            
            plt.errorbar(
                x=list(xy_coords.keys())[i],  # X position
                y=list(xy_coords.values())[i],  # Y position
                yerr=row[metric_name + '_std'],  # Error values
                fmt='none',  # No plot markers
                capsize=5,  # Cap size
                color='red',  # Color of error bars
                label=None  # Avoid duplicate legend entries
            )
            # plt.text(x=list(xy_coords.keys())[i], y=list(xy_coords.values())[i], s=str(round(row[metric_name + '_mean'],2)) +' '+ str(round(row[metric_name + '_std'],2)))
        
        plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjust bbox_to_anchor as needed
        if metric_name == 'successful_opportunity':
            plt.ylim(0,100)
        elif metric_name == 'mission_time':
            plt.ylim(0,200)
        plt.grid()
        exp_type = filename.split('/')[-1].split('_')[0]
        plt.savefig('temp_' + exp_type + '_' + policy_name + '_' + metric_name + '.png')
        plt.close()
        plt.clf()


def plot_results_engg_real(filename, metric_name = 'successful_opportunity', color_palette = 'dark:salmon_r'):
    
    exp_type = filename.split('/')[-1].split('_')[0]
    if exp_type == 'Feb24':
        color_palette = sns.color_palette('Blues', 3)
    else:
        color_palette = sns.color_palette('YlOrBr', 3)

    df = pd.read_csv(filename,  header=0)
    print(df)
    # custom_order = ['Acoustic_AOA_no_VHF','Acoustic_AOA_VHF_AOA', 'Acoustic_xy_no_VHF', 'Acoustic_xy_VHF_xy']
    # df['observation_type'] = pd.Categorical(df['observation_type'], categories=custom_order, ordered=True)
    # df = df.sort_values(by='observation_type')
    
    df['config'] = df['num_agents'].astype(str) +' robots,' + df['num_whales'].astype(str) + ' whales' 
    # df['observation_type_spaced'] = df['observation_type'].replace({
    #     'Acoustic_AOA_no_VHF': 'Acoustic\nAOA',
    #     'Acoustic_AOA_VHF_AOA': 'Acoustic +\nVHF AOA',
    #     'Acoustic_xy_no_VHF': 'Acoustic\npositioning',
    #     'Acoustic_xy_VHF_xy': 'Acoustic\npositioning + GPS'
    # })

    plt.figure(figsize=(24, 8))
    # print(df.to_string())
    ax = sns.barplot(
        x='config',  # X-axis (grouped by observation type)
        y= metric_name + '_mean',  # Y-axis (the values we want to plot)
        hue='tagging_radius',  # Grouping based on tagging_radius
        data=df,
        palette=color_palette  # Set a color palette
    )
    xy_coords = {p.get_x() + 0.5 * p.get_width():p.get_y() + p.get_height() for p in ax.patches if p.get_height() !=0}
    xy_coords = dict(sorted(xy_coords.items()))
    df = df.sort_values(by =['config', 'tagging_radius'])
    
    # x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    # y_coords = [p.get_height() for p in ax.patches]
    for i, row in df.iterrows():
        print(list(xy_coords.keys())[i], row[metric_name + '_std'])
        plt.errorbar(
            x=list(xy_coords.keys())[i],  # X position
            y=list(xy_coords.values())[i],  # Y position
            yerr=row[metric_name + '_std'],  # Error values
            fmt='none',  # No plot markers
            capsize=5,  # Cap size
            color='red',  # Color of error bars
            label=None  # Avoid duplicate legend entries
        )
        # plt.text(x=list(xy_coords.keys())[i], y=list(xy_coords.values())[i], s=str(round(row[metric_name + '_mean'],2)) +' '+ str(round(row[metric_name + '_std'],2)))

    # plt.title('Grouped Bar Plot of Observation Type vs. Successful Opportunity Mean')
    # plt.xlabel('Observation Type')
    # plt.ylabel('Successful Opportunity Mean')

    plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjust bbox_to_anchor as needed
    
    if metric_name == 'successful_opportunity':
        plt.ylim(0,100)
    elif metric_name == 'mission_time':
        plt.ylim(0,200)
    plt.grid()
    exp_type = filename.split('/')[-1].split('_')[0]
    plt.savefig('temp_' + exp_type + '_' + metric_name + '.png')

def plot_miss_freq_results_engg_real(filename, color_palette = 'dark:salmon_r'):
    
    exp_type = filename.split('/')[-1].split('_')[0]
    if exp_type == 'Feb24':
        color_palette = sns.color_palette('Blues', 3)
    else:
        color_palette = sns.color_palette('YlOrBr', 3)
    df = pd.read_csv(filename, header=0)
    print(df)
    
    df['config'] = 'a'+df['num_agents'].astype(str) +'_w'+ df['num_whales'].astype(str)
    # df['config'] = df['num_agents'] + df['num_whales']
    
    df = df.drop(columns = ['observation_type', 'num_agents','successful_opportunity_mean', 'successful_opportunity_std', 'mission_time_mean','mission_time_std', 'missing_1_or_more_whales_prob'])

    p = inflect.engine()
    

    
    configs = df['config'].unique()
    for config in configs:
        df_config = df[df['config']==config]
        num_whale = df_config['num_whales'].values[0]
        df_config = df_config.drop(columns = ['num_whales'])

        df_melted = pd.melt(df_config, id_vars=['tagging_radius'], value_vars=['missed_'+str(i)+'whales_prob' for i in range(num_whale + 1)])

        df_melted['variable'] = df_melted['variable'].replace({
        'missed_'+str(i)+'whales_prob': p.number_to_words(i) + ' missed\nwhale'+ ('s' if i>1 else '') for i in range(num_whale + 1)})

        # print(df_melted.columns)
        # exit()
        plt.figure(figsize=(24, 8))
        # print(df.to_string())
        ax = sns.barplot(
            x='variable',  # X-axis (grouped by observation type)
            y= 'value',  # Y-axis (the values we want to plot)
            hue='tagging_radius',  # Grouping based on tagging_radius
            data=df_melted,
            palette=color_palette  # Set a color palette
        )
    
        if 1==2:
            for container in ax.containers:
                ax.bar_label(container)

        # plt.title('Grouped Bar Plot of Observation Type vs. Successful Opportunity Mean')
        # plt.xlabel('Observation Type')
        # plt.ylabel('Successful Opportunity Mean')

        plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjust bbox_to_anchor as needed
    
    
        plt.ylim(0,1.2)
        plt.grid()
        plt.savefig('temp_' + exp_type + '_missed_freq_'+str(config)+'.png')



if __name__ == '__main__':
    exp_name = ''
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    if exp_name not in ['benchmark', 'Feb24', 'Nov23']:
        exp_name = 'Feb24'

    if exp_name == 'benchmark':
        plot_results_dswp('src/rebuttal_runs/dswp_50_runs.csv', metric_name='successful_opportunity')
        plot_results_dswp('src/rebuttal_runs/dswp_50_runs.csv', metric_name='mission_time')
    
    if exp_name == 'Nov23':
        plot_results_engg_real('src/rebuttal_runs/Nov23_50_runs.csv', metric_name='successful_opportunity')
        plot_results_engg_real('src/rebuttal_runs/Nov23_50_runs.csv', metric_name='mission_time')
        plot_miss_freq_results_engg_real('src/rebuttal_runs/Nov23_50_runs.csv')
    
    if exp_name == 'Feb24':
        plot_results_engg_real('src/rebuttal_runs/Feb24_50_runs.csv', metric_name='successful_opportunity')
        plot_results_engg_real('src/rebuttal_runs/Feb24_50_runs.csv', metric_name='mission_time')
        plot_miss_freq_results_engg_real('src/rebuttal_runs/Feb24_50_runs.csv')
    
    
    