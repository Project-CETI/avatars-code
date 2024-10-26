import traceback, os,sys
from src.belief_state import Belief_State
import typing as t
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from src.global_knowledge import Global_knowledge
import collections, json, copy
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from scipy.interpolate import splev, splrep
from PIL import Image

import imageio.v2 as imageio
from os import listdir
from os.path import isfile, join

import matplotlib.animation as animation
plt.rcParams.update({'font.size': 36})

Vis_state = collections.namedtuple('Vis_state', ['num_boats', 'b_xs', 'b_ys', 'num_whales', 'w_xs', 'w_ys', 'w_as', 'w_ups'])

class Visualize_file:

    def __init__(self, output_foldername, filename, tagging_radius, process_flag = True, number_of_observations_per_minute = 1, paper_view = True, legacy = False):
        self.output_folder = output_foldername +'figure_frames/' 
        if process_flag == False:
            return
        
        self.paper_view = paper_view
        self.data : t.Dict[int, Vis_state] = {}
        self.tagging_radius = tagging_radius
        self.boundary_x = [np.inf, -np.inf]
        self.boundary_y = [np.inf, -np.inf]
        if self.paper_view:
            f = open("src/configs/config_Benchmark.json", "r")
            # f = open("src/configs/config_Dominica_Nov23.json", "r")
            # f = open("src/configs/config_Dominica_Feb24.json", "r")
            knowledge = Global_knowledge()
            knowledge.set_attr(json.load(f))
        with open(filename, 'r') as f:
            if self.paper_view:
                lines = f.readlines()
            else:
                lines = f.readlines()[1:]
            for lno, line in enumerate(lines):

                
                if lno % number_of_observations_per_minute != 0 and lno != len(lines) -1:
                    continue
                
                if self.paper_view:
                    
                    st = Belief_State(knowledge=knowledge, state_str=line)
                    t_ = int(st.time / st.knowledge.observations_per_minute)
                    self.num_boats = st.number_of_boats
                    self.num_whales = st.number_of_whales
                    self.data[t_] = Vis_state(num_boats = st.number_of_boats, b_xs = st.b_x, b_ys = st.b_y, \
                        num_whales = st.number_of_whales, w_xs = st.w_x, w_ys = st.w_y, \
                            w_as = st.assigned_whales, w_ups = st.whale_up2)
                else:
                    elms = line.split(';')

                    

                    print(elms[0], elms[12], elms[13])

                    t_ = int(int(elms[0])/number_of_observations_per_minute)
                    
                    self.num_boats = int(elms[1])
                
                    b_xs = list(map(float, elms[2].split(',')))
                    b_ys = list(map(float, elms[3].split(',')))
                    self.num_whales = int(elms[6])
                    w_xs = list(map(float, elms[7].split(',')))
                    w_ys = list(map(float, elms[8].split(',')))
                    w_ups = list(map(bool, elms[11].split(',')))
                    w_as_ = elms[12].split(',')
                    if len(w_as_) == 0 or w_as_ == ['']:
                        w_as = []
                    else:
                        w_as = list(map(bool, w_as_))

                    


                    self.data[t_] = Vis_state(num_boats = self.num_boats, b_xs = b_xs, b_ys = b_ys, \
                        num_whales = self.num_whales, w_xs = w_xs, w_ys = w_ys, w_as = w_as, w_ups = w_ups)

                self.boundary_x[0] = min([self.boundary_x[0], min(self.data[t_].b_xs), min(self.data[t_].w_xs)])
                self.boundary_y[0] = min([self.boundary_y[0], min(self.data[t_].b_ys), min(self.data[t_].b_ys)])

                self.boundary_x[1] = max([self.boundary_x[1], max(self.data[t_].b_xs), max(self.data[t_].w_xs)])
                self.boundary_y[1] = max([self.boundary_y[1], max(self.data[t_].b_ys), max(self.data[t_].w_ys)])

            
            if not self.paper_view:
                self.data[t_ + 1] = Vis_state(num_boats = self.num_boats, b_xs = b_xs, b_ys = b_ys, \
                    num_whales = self.num_whales, w_xs = w_xs, w_ys = w_ys, w_as = range(self.num_whales), w_ups = w_ups)
            
        
            

   
    def save_transtion_video(self):
        fig, ax = plt.subplots(figsize = (16,16))
        
        ax.set(xlim = [self.boundary_x[0] - 1500, self.boundary_x[1] + 1500], ylim = [self.boundary_y[0] - 1500, self.boundary_y[1] + 1500], \
            xlabel = 'Easting (in meters)', ylabel = 'Northing (in meters)', title = 'Top view of results of autonomous routing')
        ax.set_ylabel('Northing (in meters)', rotation=90)
        plt.setp(ax.get_yticklabels(), ha="right", rotation=55)
        plt.setp(ax.get_xticklabels(), ha="right", rotation=30)

       
        

        k_spline = 3
        orig_times = list(self.data.keys())
        num_per_interval = 1
        new_times = np.linspace(orig_times[0], orig_times[-1], num_per_interval * (orig_times[-1] - orig_times[0] + 1))

        
      
        new_b_xs = {}
        new_b_ys = {}
        observation_area = {}
        for bid in range(self.num_boats):
            
            orig_b_xs = [self.data[t_m].b_xs[bid] for t_m in orig_times]
            orig_b_ys = [self.data[t_m].b_ys[bid] for t_m in orig_times]

            x_tck = splrep(orig_times, orig_b_xs, k = k_spline)
            new_b_xs[bid] = splev(new_times, x_tck)
            y_tck = splrep(orig_times, orig_b_ys, k = k_spline)
            new_b_ys[bid] = splev(new_times, y_tck)

            
            
      
        new_w_xs = {}
        new_w_ys = {}
        new_w_as = {wid: [] for wid in range(self.num_whales)}
        new_w_ups = {wid: [] for wid in range(self.num_whales)}
        w_thetas =  {wid: [] for wid in range(self.num_whales)}
        for wid in range(self.num_whales):
            orig_w_xs = [self.data[t_m].w_xs[wid] for t_m in orig_times]
            orig_w_ys = [self.data[t_m].w_ys[wid] for t_m in orig_times]
            orig_w_as = [True if wid in self.data[t_m].w_as else False for t_m in orig_times]
            orig_w_ups = [self.data[t_m].w_ups[wid] for t_m in orig_times]

            x_tck = splrep(orig_times, orig_w_xs, k = k_spline)
            new_w_xs[wid] = splev(new_times, x_tck)
            y_tck = splrep(orig_times, orig_w_ys, k = k_spline)
            new_w_ys[wid] = splev(new_times, y_tck)

            for a in orig_w_as:
                new_w_as[wid].extend([True if a == True else False for _ in range(num_per_interval)])
            for u in orig_w_ups:
                new_w_ups[wid].extend([True if u == True else False for _ in range(num_per_interval)])

            w = num_per_interval+1
            thetas = [np.arctan2(new_w_ys[wid][t1+w] - new_w_ys[wid][t1], new_w_xs[wid][t1+w] - new_w_xs[wid][t1]) * 180 / np.pi \
                for t1,t2 in enumerate(new_times) if t1 + w < len(new_times) ]
            
            w_thetas[wid] = [theta for theta in thetas] + [thetas[-1] for _ in range(w)]
                
        
        b_plot = {bid: ax.plot(new_b_xs[bid][0], new_b_ys[bid][0], c= "orange", linestyle = '--',  linewidth=2)[0] for bid in range(self.num_boats)} 
        
        w_scat = {wid: ax.scatter(new_w_xs[wid][0], new_w_ys[wid][0], \
            c=('cyan' if new_w_ups[wid][0] else 'black'), marker = '^') \
            for wid in range(self.num_whales)}
        b_scat = {bid: ax.scatter(new_b_xs[bid][0], new_b_ys[bid][0], c= "orange", marker = '*', s = 500) for bid in range(self.num_boats)} 

            
        current_whale_assigned = [False for wid in range(self.num_whales)]

        def update(frame):
            for a in ax.artists:
                a.remove()
            for a in ax.patches:
                a.remove()

            


            for bid in range(self.num_boats):
                theta = 0
                viewing_radius_bid = pat.Circle((new_b_xs[bid][frame], new_b_ys[bid][frame]), \
                    edgecolor = 'orange', linewidth = 5, linestyle = '--', radius = self.tagging_radius, fill = False)
                observation_area[bid] = ax.add_patch(viewing_radius_bid)
                
                b_plot[bid].set_xdata(new_b_xs[bid][:frame + 1])
                b_plot[bid].set_ydata(new_b_ys[bid][:frame + 1])
                
                
                b_scat[bid].set_offsets([new_b_xs[bid][frame], new_b_ys[bid][frame]])
                

            for wid in range(self.num_whales):
                
                
                if not current_whale_assigned[wid]:
                                    
                    larr = np.array([new_w_xs[wid][:frame+1], new_w_ys[wid][:frame+1]]).transpose().reshape(frame + 1,2)
                    carr = np.array(['cyan' if up else 'black' for up in new_w_ups[wid][:frame+1]])
                    sarr = np.array([200 if up else 100 for up in new_w_ups[wid][:frame+1]])
                    sarr[-1] = 700
                    if new_w_as[wid][frame] == 1:
                        carr[-1] = 'red'
                    w_scat[wid].set_offsets(larr)
                    w_scat[wid].set_facecolors(carr)
                    w_scat[wid].set_sizes(sarr)
                    
                    
                    # except Exception as e:
                    #     full_traceback = traceback.extract_tb(e.__traceback__)
                    #     print(e)
                
                

                if new_w_as[wid][frame] == 1:
                    if not current_whale_assigned[wid]:
                        current_whale_assigned[wid] = True
                            
                   
                    
                        


                
            
                
            plt.savefig(self.output_folder + 'frame'+str(frame)+'.png')
            
        ani = animation.FuncAnimation(fig = fig, func = update, frames = len(new_times), interval = 120, repeat = False, blit = False)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        ani.save(filename=self.output_folder + "AVATAR_autonomy_example.gif", writer="pillow")
       
          
if __name__ == '__main__':
    
    output_foldername = 'visualize_output/'
    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)

    filename = sys.argv[1]

    v = Visualize_file(output_foldername, filename, 500, process_flag = True, number_of_observations_per_minute = 60, paper_view = True, legacy=True)

    v.save_transtion_video()
    

