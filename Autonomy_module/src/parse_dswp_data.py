from src.global_knowledge import Global_knowledge

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib
import pickle, math, os, json
import matplotlib.pyplot as plt
import traceback
from src.configs.constants import *
from scipy.stats import truncnorm 

def parse_dswp_data(knowledge: Global_knowledge, date_list = None, idn_list = None):
    
    date_cols = ['DateTime']

    df = pd.read_csv('Shanes_GPS_20230127.csv', \
        sep = ',', names = ['Roll', 'Filename', 'DateTime', 'Encounter', 'Cluster', 'LAT', 'LONG', 'Males', \
            'Adults', 'Calves', 'Part', 'Class', 'Q', 'IDN', 'Grouping'], \
                header=0, parse_dates=date_cols, dayfirst=True)
    df = df[df.DateTime.notnull()] 
    df['DateTime'] = df['DateTime'].apply(pd.to_datetime, dayfirst=True)
    df['date'] = df['DateTime'].apply(lambda x: "%d-%d-%d" % (x.year, x.month, x.day))
    df['seconds'] = df['DateTime'].apply(lambda x: (x.hour*3600 + x.minute*60 + x.second))
    df['minutes'] = df['seconds'].apply(lambda x: int(x/60))
    
    df = df[df['Males']=='0']
    df = df[df['Adults'] == '1']
    
    df = df.drop_duplicates(subset=['date','LAT','LONG', 'IDN'])
    
    df['hour'] = df['minutes'].apply(lambda x: int(x/60))
    

    

    df = df.sort_values(by=['seconds'], ascending = True)

    units = df.groupby(['date', 'IDN']).filter(lambda x : len(x)>2)

    units = np.array(units[['date', 'IDN']].drop_duplicates())
    
    if date_list is None:
       date_list = [u0 for (u0,u1) in list(zip(units[:,0], units[:,1]))]
    if idn_list is None:
        idn_list = [u1 for (u0,u1) in list(zip(units[:,0], units[:,1]))]
        plot_not_save = False
    else:
        plot_not_save = True

    df_date_idn = None
    data_dict = {}
    for l in range(len(idn_list)):
        df_date_idn_ = df[(df['date']==date_list[l]) & (df['IDN']==idn_list[l])]
        if df_date_idn is None:
            df_date_idn = df_date_idn_
        else:
            df_date_idn = pd.concat([df_date_idn, df_date_idn_])
        data_dict[l] = df_date_idn_
   
    append_missing_surfacing_and_fill_gap_in_trajectory(data_dict, date_list, idn_list, knowledge, plot_not_save)
    # find_missing_surfacings(data_dict, date_list, idn_list, knowledge, plot_not_save)


def find_missing_surfacings(data_dict, date_list, idn_list, knowledge: Global_knowledge, plot_not_save):
    dist = {}
    for l in range( len(idn_list)):
        data_dict_l = data_dict[l]
        
        # xys = [long_lat for long_lat in list(zip(data_dict_l.LONG, data_dict_l.LAT))]
        # xys = [[xy[0], xy[1]] for xy in xys]
        min_sec = min(data_dict_l['seconds'].values)
        ts = [t - min_sec for t in  data_dict_l['seconds'].values]

        for index in range(1, len(ts)):
            k = int((ts[index] - ts[index - 1])/60)
            if k not in dist.keys():
                dist[k] = 1
            else:
                dist[k] += 1

    plt.scatter(dist.keys(), dist.values())
    max_key = max(dist.keys())
    max_val = max(dist.values())
    tot_mean = (knowledge.down_time_mean + knowledge.surface_time_mean) 
    # + \
    #     (knowledge.down_time_var + knowledge.surface_time_var)
    tot_v = (knowledge.down_time_var + knowledge.surface_time_var)
    for i in range(int(np.ceil(max_key/tot_mean)) + 1):
        plt.plot([tot_mean * i , tot_mean * i], [0, max_val], c = 'b')
        plt.plot([tot_mean * i - tot_v , tot_mean * i - tot_v], [0, max_val], c = 'grey', alpha = 0.5)
        plt.plot([tot_mean * i + tot_v , tot_mean * i + tot_v], [0, max_val], c = 'brown', alpha = 0.5)
    plt.show()
    print('')

def append_missing_surfacing_and_fill_gap_in_trajectory(data_dict, date_list, idn_list, knowledge: Global_knowledge, plot_not_save):
    colors = matplotlib.colors.cnames

    small_colvals = np.random.choice(list(colors.values()), len(idn_list)) 
  
    for l in range( len(idn_list)):
        data_dict_l = data_dict[l]
        
        xys = [long_lat for long_lat in list(zip(data_dict_l.LONG, data_dict_l.LAT))]
        # else:
        #     xys = [convert_longlat_to_xy_in_meters(long_lat[0], long_lat[1]) for long_lat in list(zip(data_dict_l.LONG, data_dict_l.LAT))]
        xys = [[xy[0], xy[1]] for xy in xys]
        # df_minutes = data_dict_l['DateTime'].apply(lambda x: (x.hour*60 + x.minute))
        min_sec = min(data_dict_l['seconds'].values)
        ts = [t - min_sec for t in  data_dict_l['seconds'].values]

        new_ts = [ts[0]]
        new_xs = [xys[0][0]]
        new_ys = [xys[0][1]]
        for index in range(1, len(ts)):
           
            if ts[index] - ts[index-1] >= (knowledge.down_time_mean * 60 + knowledge.surface_time_mean * 60):
                
                num_new_points = math.floor((ts[index] - ts[index-1]) / (knowledge.down_time_mean * 60 + knowledge.surface_time_mean * 60))

                # if num_new_points > 1:
                #     print('More than 1 sightings to be added')
                # elif num_new_points == 1:
                #     print('1 sighting to be added')
                #     mu = knowledge.down_time_mean * 60 + knowledge.surface_time_mean * 60
                #     sigma = knowledge.down_time_var * 60 + knowledge.surface_time_var * 60
                    # interval_duration = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc = mu, scale = sigma, size = 1)[0]

                new_times = np.maximum(knowledge.down_time_var * 60 + knowledge.surface_time_var * 60, \
                    np.random.normal(knowledge.down_time_mean * 60 + knowledge.surface_time_mean * 60, knowledge.down_time_var * 60 + knowledge.surface_time_var * 60, \
                        num_new_points).astype(int))

                last_time = ts[index - 1]
                for ni in range(num_new_points):
                    last_time += new_times[ni]
                    if last_time > ts[index] - knowledge.down_time_var * 60 - knowledge.surface_time_var * 60:
                        continue
                    
                    new_ts.append(last_time)
                    t_frac = (last_time - ts[index-1])/ (ts[index] - ts[index-1])
                    new_xs.append(xys[index-1][0] +  (xys[index][0] - xys[index-1][0]) * t_frac )
                    new_ys.append(xys[index-1][1] +  (xys[index][1] - xys[index-1][1]) * t_frac )
                    # new_ts.append(int(ts[index-1] + (ts[index] - ts[index-1]) * (1+ni)/(1+num_new_points)))
                    # new_xs.append(xys[index-1][0] + (xys[index][0] - xys[index-1][0]) * (1+ni)/(1+num_new_points))   
                    # new_ys.append(xys[index-1][1] + (xys[index][1] - xys[index-1][1]) * (1+ni)/(1+num_new_points))    
            
            new_ts.append(ts[index])
            new_xs.append(xys[index][0])
            new_ys.append(xys[index][1])
        
        
        if len(new_ts) == 1:
            continue
        last_seen_points = {'xs': new_xs, 'ys': new_ys, 'ts': new_ts}
       
        whales_states, scenarios = generate_trajectories_v2(last_seen_points, \
            knowledge, plot_not_save = plot_not_save)

        if plot_not_save:
            plt.show()
        else:
            main_dir = knowledge.parsed_whale_data_output
            if not os.path.exists(main_dir):
                os.mkdir(main_dir)

            fout_acoustic_stop_start = open(main_dir + str(date_list[l]) + '_' + str(idn_list[l]) + 'acoustic_end_start.csv', 'w')
            fout_surface_interval = open(main_dir + str(date_list[l]) + '_' + str(idn_list[l]) + 'surface_interval.csv', 'w')
            for ss_se in scenarios:
                w_now = whales_states[ss_se[1]][0], whales_states[ss_se[1]][2]
                if ss_se[1] + 1 in whales_states.keys():
                    w_next = whales_states[ss_se[1] + 1][0], whales_states[ss_se[1] + 1][2]
                    angle_pid = 90 - 180 * np.arctan2(w_next[1] - w_now[1], w_next[0] - w_now[0]) / np.pi
                else:
                    angle_pid = None
                fout_acoustic_stop_start.write(str(ss_se[0]) + ',' + str(ss_se[1]) + ',' + str(angle_pid)+ '\n')
                fout_surface_interval.write(str(ss_se[0]) + ',' + str(ss_se[1]) + ',' + str(angle_pid)+ '\n')
            fout_acoustic_stop_start.close()
            fout_surface_interval.close()
            whale_xs = [w[0] for w in whales_states.values()]
            whale_ys = [w[2] for w in whales_states.values()]
            whale_ts = [t for t in whales_states.keys()]
            ground_truth_df = pd.DataFrame({'sec': whale_ts, \
                'gt_fluke_long': whale_xs, 'gt_fluke_lat': whale_ys, \
                    'camera_lon': [None] *len(whales_states), 'camera_lat': [None] *len(whales_states), \
                        'fluke_angle': [None] *len(whales_states) })
            ground_truth_df.to_csv(main_dir + str(date_list[l]) + '_' + str(idn_list[l]) + 'ground_truth.csv', \
                sep = ',', index = False, header= False)


            vhf_x0_gps, vhf_y0_gps = get_gps_from_start_vel_bearing(Roseau_long, Roseau_lat, 2* 3250, -90) # ~2 miles off the coast of Roseau
            towed_array_x0_gps, towed_array_y0_gps = get_gps_from_start_vel_bearing(Roseau_long, Roseau_lat, 4*6500, -90) # ~4 miles off the coast of Roseau
            towed_array_speed_mtps = 2.57
            vhf_speed_mtps = 2.57
            towed_array_heading_angle = 0
            vhf_heading_angle = 0

            start_t = list(whales_states.keys())[0]
            end_t = list(whales_states.keys())[-1]
            towed_array_xT_gps, towed_array_yT_gps = get_gps_from_start_vel_bearing(towed_array_x0_gps, towed_array_y0_gps, \
                towed_array_speed_mtps * (end_t - start_t), towed_array_heading_angle)
            k_ = 1
            x_tck = interpolate.splrep([start_t, end_t], [towed_array_x0_gps, towed_array_xT_gps], k = k_)
            ta_x_coords = interpolate.splev(np.arange(start_t, end_t + 1), x_tck)
            y_tck = interpolate.splrep([start_t, end_t], [towed_array_y0_gps, towed_array_yT_gps], k = k_)
            ta_y_coords = interpolate.splev(np.arange(start_t, end_t + 1), y_tck)

            vhf_xT_gps, vhf_yT_gps = get_gps_from_start_vel_bearing(vhf_x0_gps, vhf_y0_gps, vhf_speed_mtps *  (end_t - start_t), vhf_heading_angle)
            x_tck = interpolate.splrep([start_t, end_t], [vhf_x0_gps, vhf_xT_gps], k = k_)
            vhf_x_coords = interpolate.splev(np.arange(start_t, end_t + 1), x_tck)
            y_tck = interpolate.splrep([start_t, end_t], [vhf_y0_gps, vhf_yT_gps], k = k_)
            vhf_y_coords = interpolate.splev(np.arange(start_t, end_t + 1), y_tck)

            a_whale_xs = [w[0] for w in whales_states.values() if w[4]==0]
            a_whale_ys = [w[2] for w in whales_states.values() if w[4]==0]
            a_whale_ts = [t for t,w in whales_states.items() if w[4]==0]
            a_sensor_long = [ax for w, ax in list(zip(whales_states.values(), ta_x_coords)) if w[4] == 0]
            a_sensor_lat = [ay for w, ay in list(zip(whales_states.values(), ta_y_coords)) if w[4] == 0]

            acoustic_aoas = np.array([get_bearing_from_p1_p2(tax, tay, ax, ay) * Radian_to_degree \
                for (tax, tay, ax,ay) in list(zip(a_sensor_long, a_sensor_lat, a_whale_xs, a_whale_ys))]) \
                    + np.random.normal(0, knowledge.Acoustic_AOA_obs_error_std_degree, size = len(a_whale_ts))
            
            
            # a_sensor_long = [bouy1_long]* len(a_whale_xs)
            # a_sensor_lat = [bouy1_lat]* len(a_whale_xs)

            v_whale_xs = [w[0] for w in whales_states.values() if w[4]==1]
            v_whale_ys = [w[2] for w in whales_states.values() if w[4]==1]
            v_whale_ts = [t for t,w in whales_states.items() if w[4]==1]
            v_sensor_long = [vx for w, vx in zip(whales_states.values(), vhf_x_coords) if w[4] == 1]
            v_sensor_lat = [vy for w, vy in zip(whales_states.values(), vhf_y_coords) if w[4] == 1]
            vhf_aoas = np.array([get_bearing_from_p1_p2(sx, sy, vx, vy)* Radian_to_degree \
                for (sx, sy, vx,vy) in list(zip(v_sensor_long, v_sensor_lat, v_whale_xs, v_whale_ys))]) \
                    + np.random.normal(0, knowledge.Vhf_AOA_obs_error_std_degree, size = len(v_whale_ts))
            # v_sensor_long = [bouy2_long]* len(v_whale_xs)
            # v_sensor_lat = [bouy2_lat]* len(v_whale_xs)
            

            
            sensor_df = pd.DataFrame({'sensor_sec': a_whale_ts + v_whale_ts, \
                'sensor_lon': a_sensor_long + v_sensor_long, \
                    'sensor_lat': a_sensor_lat + v_sensor_lat, \
                        'aoa': [None]*len(a_whale_ts) + [None]*len(v_whale_ts) , \
                            'aoa1': acoustic_aoas.tolist() + vhf_aoas.tolist(), \
                                'aoa2': acoustic_aoas.tolist() + vhf_aoas.tolist(), \
                                    'std_error':[knowledge.Acoustic_AOA_obs_error_std_degree]*len(a_whale_ts) + \
                                        [knowledge.Vhf_AOA_obs_error_std_degree]*len(v_whale_ts), \
                                            'sensor_name':['A']*len(a_whale_ts) + ['V']*len(v_whale_ts)})
            # sensor_up = sensor_df[(sensor_df['sensor_name']!='A') & (sensor_df['sensor_sec']>=1500)]
            # print(sensor_up)
            
            sensor_df.to_csv(main_dir + str(date_list[l]) + '_' + str(idn_list[l]) +'aoa.csv', \
                sep = ',', index = False, header= False)

            b_whale_ts = [t for t,w in whales_states.items() if w[4]==0]
            g_whale_ts = [t for t,w in whales_states.items() if w[4]==1]
            w_loc = np.transpose(np.array([a_whale_xs, a_whale_ys]).reshape(2, len(a_whale_ts)))
            eta_d = np.random.normal(0, np.sqrt(knowledge.Acoustic_XY_obs_error_cov_m2[0]), size = len(a_whale_ts))
            eta_a = np.random.uniform(-np.pi, np.pi, size = len(a_whale_ts))
            bw_longs_lats = [get_gps_from_start_vel_bearing(w_loc[i][0], w_loc[i][1], eta_d[i], eta_a[i]) \
                for i in range(len(a_whale_ts))]
            b_whale_longs = [l[0] for l in bw_longs_lats]
            b_whale_lats = [l[1] for l in bw_longs_lats]
            w_loc = np.transpose(np.array([v_whale_xs, v_whale_ys]).reshape(2, len(v_whale_ts)))
            eta_d = np.random.normal(0, np.sqrt(knowledge.Vhf_XY_obs_error_cov_m2[0]), size = len(v_whale_ts))
            eta_a = np.random.uniform(-np.pi, np.pi, size = len(v_whale_ts))
            gw_longs_lats = [get_gps_from_start_vel_bearing(w_loc[i][0], w_loc[i][1], eta_d[i], eta_a[i]) \
                for i in range(len(v_whale_ts))]
            g_whale_longs = [l[0] for l in gw_longs_lats]
            g_whale_lats = [l[1] for l in gw_longs_lats]
            xy_sensor_df = pd.DataFrame({'sensor_sec': b_whale_ts + g_whale_ts, \
                'w_long': b_whale_longs + g_whale_longs, \
                    'w_lat': b_whale_lats + g_whale_lats, \
                        'long_std': [np.sqrt(knowledge.Acoustic_XY_obs_error_cov_m2[0])] * len(b_whale_ts) + [np.sqrt(knowledge.Vhf_XY_obs_error_cov_m2[0])] * len(g_whale_ts), \
                            'lat_std': [np.sqrt(knowledge.Acoustic_XY_obs_error_cov_m2[1])] * len(b_whale_ts) + [np.sqrt(knowledge.Vhf_XY_obs_error_cov_m2[1])] * len(g_whale_ts), \
                                'sensor_name':['U']*len(b_whale_ts) + ['G']*len(g_whale_ts)})

            xy_sensor_df.to_csv(main_dir + str(date_list[l]) + '_' + str(idn_list[l]) +'xy.csv', \
                sep = ',', index = False, header= False)

            
             

            # data = {'scenarios': scenarios[l], 'whales_states': whales_states[l]}
            # f = open(main_dir + 'whale_' + str(date_list[l]) + '_' + str(idn_list[l]), 'wb')
            # pickle.dump(data, f)
            f.close()
            plt.savefig(main_dir + 'whale_track_' + str(date_list[l]) + '_' + str(idn_list[l]) + '.png')
            plt.cla()
        
            
        

    
from scipy import interpolate
import itertools
def generate_trajectories_v2(last_seen_points, knowledge: Global_knowledge, plot_not_save = False):
    whale_states = {}
    
    
    surface_interval_durations = np.maximum(knowledge.surface_time_var * 60, \
        np.random.normal(knowledge.surface_time_mean * 60, knowledge.surface_time_var * 60, len(last_seen_points['ts'])).astype(int))

    scenario = [[last_seen_points['ts'][t] - surface_interval_durations[t], last_seen_points['ts'][t]] \
        for t in range(len(last_seen_points['ts']))]


    s_d  = [(scenario[i][1]-scenario[i][0]+1)*[1] + \
        ((scenario[i+1][0] - scenario[i][1]-1)*[0] if i<len(scenario)-1 else [0])  \
            for i in range(len(scenario))]
    whale_ups = list(itertools.chain(*s_d)) 

    # whale_ups2 = []
    # for i in range(len(scenario)):
    #     up_s = (scenario[i][1]-scenario[i][0]+1)*[1]
    #     whale_ups2.extend(up_s)
    #     if i<len(scenario)-1:
    #         down_s = (scenario[i+1][0] - scenario[i][1]-1)*[0]
    #     else:
    #         down_s = [0]
    #     whale_ups2.extend(down_s)   
    # print(scenario, whale_ups, whale_ups2)

    times = np.arange(scenario[0][0], scenario[-1][1]+1)
    k_ = 2
    if (len(last_seen_points['ts']) <= 3):
        k_ = len(last_seen_points['ts']) - 1
    x_tck = interpolate.splrep(last_seen_points['ts'], last_seen_points['xs'], k = k_)
    x_coords = interpolate.splev(times, x_tck)
    y_tck = interpolate.splrep(last_seen_points['ts'], last_seen_points['ys'], k = k_)
    y_coords = interpolate.splev(times, y_tck)

    dot_xs = [t - s for s, t in zip(x_coords, x_coords[1:])] + [0]
    dot_ys = [t - s for s, t in zip(y_coords, y_coords[1:])] + [0]

    # dist0 = knowledge.get_distance_from_latLon_to_meter(y_coords[0], x_coords[0], y_coords[1], x_coords[1])
    for t1 in times:
        t2 = t1 - times[0]
        whale_states[t1] = [x_coords[t2], dot_xs[t2], y_coords[t2], dot_ys[t2], whale_ups[t2]]
        
    xs = [w[0] for w in whale_states.values()]
    ys = [w[2] for w in whale_states.values()]
    cs = ['red' if w[4] else 'black' for w in whale_states.values()]
    plt.scatter(xs, ys, c = cs, s = 10)
    

    plt.scatter(xs[0], ys[0], c = cs[0], s = 10)
        
    if not plot_not_save:
        for (xe,ye,t) in list(zip(last_seen_points['xs'], last_seen_points['ys'], last_seen_points['ts'])):
            plt.text(xe, ye, s = str(int(t/3600))+':'+str(int(time_str/60) -  int(t/3600) * 60), rotation = 90, c = 'blue')
        
    return whale_states, scenario


if __name__ == '__main__':
    try:
        f = open("src/configs/config_Benchmark.json", "r")
        knowledge = Global_knowledge()
        knowledge.set_attr(json.load(f))
        parse_dswp_data(knowledge)
        # parse_dswp_data(knowledge, date_list=['2005-1-23'], idn_list=[5722])
    except Exception as e:
        print('Exception: ', e)
        full_traceback = traceback.extract_tb(e.__traceback__)
        filename, lineno, funcname, text = full_traceback[-1]
        print("Error in ", filename, " line:", lineno, " func:", funcname, ", exception:", e)


if __name__ == '__main__generate_whale_trace_pngs':

    directory = 'dswp_parsed_data_moving_sensors/'
    for fn in os.listdir(directory):
        xy_filename = os.path.join(directory, fn)
        if os.path.isfile(xy_filename) and 'ground_truth.csv' not in xy_filename:
            continue
        whale_trace_old = pd.read_csv(xy_filename, \
            names=['sensor_sec', 'w_long', 'w_lat', 'c_long', 'c_lat', 'fluke_angle'], header=None)
        whale_trace_old = whale_trace_old.sort_values(by=['sensor_sec'])

        k_ = 2
        t_min = min(whale_trace_old['sensor_sec'].values)
        t_max = max(whale_trace_old['sensor_sec'].values)
        time_series = np.arange(t_min, t_max + 1)
        x_tck = interpolate.splrep(whale_trace_old['sensor_sec'].values, whale_trace_old['w_long'].values, k = k_)
        x_coords = interpolate.splev(time_series, x_tck)
        y_tck = interpolate.splrep(whale_trace_old['sensor_sec'].values, whale_trace_old['w_lat'].values, k = k_)
        y_coords = interpolate.splev(time_series, y_tck)
        whale_trace = pd.DataFrame({'sensor_sec': time_series, 'w_long': x_coords, 'w_lat': y_coords})

        xs = whale_trace['w_long'].values
        ys = whale_trace['w_lat'].values
        # cs = ['red' if w == 'G' else 'black' for w in whale_trace['sensor_name'].values]
        plt.scatter(xs, ys, c = 'black', s = 1, label = 'underwater phase')
            
        surface_time_filename = directory + fn.replace('ground_truth.csv', 'surface_interval.csv')
        surface_time_df = pd.read_csv(surface_time_filename, \
            names=['surface_start', 'surface_stop', 'fluke_camera_aoa'], header=None)
        start_time_rows = surface_time_df.merge(whale_trace, left_on='surface_start', right_on='sensor_sec')
        end_time_rows = surface_time_df.merge(whale_trace, left_on='surface_stop', right_on='sensor_sec')

        f_row = 0
        for _, surface_row in surface_time_df.iterrows():
            whale_trace_sur = whale_trace[((whale_trace['sensor_sec']>=surface_row['surface_start']) & \
                (whale_trace['sensor_sec']<=surface_row['surface_stop'])) ]
            
            if f_row > 0:
                plt.scatter(whale_trace_sur['w_long'].values, whale_trace_sur['w_lat'].values, c = 'red', s = 1)
            else:
                plt.scatter(whale_trace_sur['w_long'].values, whale_trace_sur['w_lat'].values, c = 'red', s = 1, label = 'surface phase')
            f_row += 1
        for _, start_time_row in start_time_rows.iterrows():
            time_str = start_time_row['sensor_sec']
            hour = int(time_str/3600)
            minute = int(time_str/60) -  hour * 60
            if hour < 0 or minute < 0:
                continue  
            time_str = str(hour) + ':' + str(minute)
            plt.text(start_time_row['w_long'], start_time_row['w_lat'], s = time_str, rotation = 90, c = 'blue')

        for _, end_time_row in end_time_rows.iterrows():
            time_str = end_time_row['sensor_sec']
            hour = int(time_str/3600)
            minute = int(time_str/60) -  hour * 60
            if hour < 0 or minute < 0:
                continue  
            time_str = str(hour) + ':' + str(minute)
            plt.text(end_time_row['w_long'], end_time_row['w_lat'], s = time_str, rotation = 90, c = 'blue')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.yticks(rotation = 45)
        plt.legend()
        plt.savefig(directory + 'whale_track_' + fn.replace('ground_truth.csv', '') + '.png')
        plt.cla()