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





def plot_results_dswp(filename, metric_name = 'successful_opportunity', color_palette = 'rocket_r', output_foldername = None):
    
    color_palette = sns.color_palette('flare', 3)
    

    df = pd.read_csv(filename,  header=0)
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
        
        xy_coords = {p.get_x() + 0.5 * p.get_width():p.get_y() + p.get_height() for p in ax.patches if p.get_height() !=0}
        xy_coords = dict(sorted(xy_coords.items()))
        df_policy = df_policy.sort_values(by =['observation_type', 'tagging_radius'])
        for i, row in df_policy.iterrows():
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
            
        plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjust bbox_to_anchor as needed
        if metric_name == 'successful_opportunity':
            plt.ylim(0,100)
        elif metric_name == 'mission_time':
            plt.ylim(0,200)
        plt.grid()
        exp_type = filename.split('/')[-1].split('_')[0]
        if output_foldername is None:
            plt.savefig('temp_' + exp_type + '_' + policy_name + '_' + metric_name + '.png')
        else:
            plt.savefig(output_foldername + 'results_' + exp_type + '_' + policy_name + '_' + metric_name + '.png')
        plt.close()
        plt.clf()


def plot_results_engg_real(filename, metric_name = 'successful_opportunity', color_palette = 'dark:salmon_r', output_foldername = None):
    
    exp_type = filename.split('/')[-1].split('_')[0]
    if exp_type == 'Feb24':
        color_palette = sns.color_palette('Blues', 3)
    else:
        color_palette = sns.color_palette('YlOrBr', 3)

    df = pd.read_csv(filename,  header=0)
    print(df)
    
    df['config'] = df['num_agents'].astype(str) +' robots,' + df['num_whales'].astype(str) + ' whales' 
    
    plt.figure(figsize=(24, 8))
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
        

    plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  # Adjust bbox_to_anchor as needed
    
    if metric_name == 'successful_opportunity':
        plt.ylim(0,100)
    elif metric_name == 'mission_time':
        plt.ylim(0,200)
    plt.grid()
    exp_type = filename.split('/')[-1].split('_')[0]

    if output_foldername is None:
        plt.savefig('temp_' + exp_type + '_' + metric_name + '.png')
    else:
        plt.savefig(output_foldername + 'results_' + exp_type + '_' + metric_name + '.png')
    

def plot_miss_freq_results_engg_real(filename, color_palette = 'dark:salmon_r', output_foldername = None):
    
    exp_type = filename.split('/')[-1].split('_')[0]
    if exp_type == 'Feb24':
        color_palette = sns.color_palette('Blues', 3)
    else:
        color_palette = sns.color_palette('YlOrBr', 3)
    df = pd.read_csv(filename, header=0)
    print(df)
    
    df['config'] = 'a'+df['num_agents'].astype(str) +'_w'+ df['num_whales'].astype(str)
    
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

        
        plt.figure(figsize=(24, 8))
        ax = sns.barplot(
            x='variable',  # X-axis (grouped by observation type)
            y= 'value',  # Y-axis (the values we want to plot)
            hue='tagging_radius',  # Grouping based on tagging_radius
            data=df_melted,
            palette=color_palette  # Set a color palette
        )
    
        

        plt.legend(title='Tagging Radius', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)  
    
    
        plt.ylim(0,1.2)
        plt.grid()

        if output_foldername is None:
            plt.savefig('temp_' + exp_type + '_missed_freq_'+str(config)+'.png')
        else:
            plt.savefig(output_foldername + 'results_' + exp_type + '_missed_freq_'+str(config)+'.png')

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
    
    
    