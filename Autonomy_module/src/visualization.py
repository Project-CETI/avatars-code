import traceback, os
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
# Add more contrast to image_names
# Add dotted lines for traces
# Add more contrast to whale surface and underwater
class Visualize_file:

    def __init__(self, foldername, filename, tagging_radius, process_flag = True, number_of_observations_per_minute = 1, paper_view = False, legacy = False):
        self.output_folder = foldername +'figure_frames/' # 'output/dominica_experiment/frames/'
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
        with open(foldername + filename, 'r') as f:
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

                    if 1==2:
                        w_last_ss = list(map(int, elms[12].split(',')))
                        w_last_ss = [int(ss / number_of_observations_per_minute) for ss in w_last_ss]
                        se = elms[13].rstrip().split(',')
                        w_last_se = [0 for _ in range(self.num_whales)]
                        for wid in range(self.num_whales):
                            w_last_se[wid] = int(int(se[wid])/number_of_observations_per_minute) \
                                if se[wid] != 'None' and se[wid] != 'nan' else w_last_ss[wid] - 10

                        w_ups = [0 for _ in range(self.num_whales)]
                        for wid in range(self.num_whales):
                            if t_ >= w_last_ss[wid] and (w_last_ss[wid] > w_last_se[wid] or t_<=w_last_se[wid]): 
                                w_ups[wid] = 1


                    self.data[t_] = Vis_state(num_boats = self.num_boats, b_xs = b_xs, b_ys = b_ys, \
                        num_whales = self.num_whales, w_xs = w_xs, w_ys = w_ys, w_as = w_as, w_ups = w_ups)

                self.boundary_x[0] = min([self.boundary_x[0], min(self.data[t_].b_xs), min(self.data[t_].w_xs)])
                self.boundary_y[0] = min([self.boundary_y[0], min(self.data[t_].b_ys), min(self.data[t_].b_ys)])

                self.boundary_x[1] = max([self.boundary_x[1], max(self.data[t_].b_xs), max(self.data[t_].w_xs)])
                self.boundary_y[1] = max([self.boundary_y[1], max(self.data[t_].b_ys), max(self.data[t_].w_ys)])

            
            if not self.paper_view:
                self.data[t_ + 1] = Vis_state(num_boats = self.num_boats, b_xs = b_xs, b_ys = b_ys, \
                    num_whales = self.num_whales, w_xs = w_xs, w_ys = w_ys, w_as = range(self.num_whales), w_ups = w_ups)
            
        l = len(lines)

    def show(self):
        
        for t_ in self.data.keys():
            fig, ax = plt.subplots(figsize = (16,16))
            times = [t_m for t_m in self.data.keys() if t_m <= t_]
            b_xs = {}
            b_ys = {}
            for bid in range(self.num_boats):
                b_xs[bid] = {t_m: self.data[t_m].b_xs[bid] for t_m in times}
                b_ys[bid] = {t_m: self.data[t_m].b_ys[bid] for t_m in times}
            w_xs = {}
            w_ys = {}
            w_as = {}
            w_ups = {}
            for wid in range(self.num_whales):
                w_xs[wid] = {t_m: self.data[t_m].w_xs[wid] for t_m in times}
                w_ys[wid] = {t_m: self.data[t_m].w_ys[wid] for t_m in times}
                w_as[wid] = {t_m: True if wid in self.data[t_m].w_as else False for t_m in times}
                w_ups[wid] = {t_m: self.data[t_m].w_ups[wid] for t_m in times}


            for wid in range(self.num_whales):
                for all_t in times:
                    self.surfaced_whale = OffsetImage(image.imread("output/dominica_experiment/surfaced_whale.png"), zoom = 0.15)
                    self.surfaced_tagged_whale = OffsetImage(image.imread("output/dominica_experiment/surfaced_tagged_whale.png"), zoom = 0.07)
                    self.underwater_whale = OffsetImage(image.imread("output/dominica_experiment/underwater_whale.png"), zoom = 0.15)
                    self.underwater_tagged_whale = OffsetImage(image.imread("output/dominica_experiment/underwater_tagged_whale.png"), zoom = 0.07)

                    self.current_surfaced_whale = OffsetImage(image.imread("output/dominica_experiment/surfaced_whale.png"), zoom = 0.5)
                    self.current_surfaced_tagged_whale = OffsetImage(image.imread("output/dominica_experiment/surfaced_tagged_whale.png"), zoom = 0.14)
                    self.current_underwater_whale = OffsetImage(image.imread("output/dominica_experiment/underwater_whale.png"), zoom = 0.30)
                    self.current_underwater_tagged_whale = OffsetImage(image.imread("output/dominica_experiment/underwater_tagged_whale.png"), zoom = 0.14)

                    if w_ups[wid][all_t] == 1:
                        if all_t != t_:
                            if w_as[wid][all_t]:
                                ab = AnnotationBbox(self.surfaced_tagged_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                            else:
                                ab = AnnotationBbox(self.surfaced_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                        else:
                            if w_as[wid][all_t]:
                                ab = AnnotationBbox(self.current_surfaced_tagged_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                            else:
                                ab = AnnotationBbox(self.current_surfaced_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                    else:
                        if all_t != t_:
                            if w_as[wid][all_t]:
                                ab = AnnotationBbox(self.underwater_tagged_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                            else:
                                ab = AnnotationBbox(self.underwater_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                        else:
                            if w_as[wid][all_t]:
                                ab = AnnotationBbox(self.current_underwater_tagged_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)
                            else:
                                ab = AnnotationBbox(self.current_underwater_whale, (w_xs[wid][all_t], w_ys[wid][all_t]), frameon = False)
                                ax.add_artist(ab)


            for bid in range(self.num_boats):
                # plt.scatter(b_xs[bid], b_ys[bid])
                for all_t in times:
                    
                    self.boat_img = OffsetImage(image.imread("output/dominica_experiment/boat.png"), zoom = 0.05)
                    self.current_boat_img = OffsetImage(image.imread("output/dominica_experiment/boat.png"), zoom = 0.3)
                    if all_t != t_:
                        None
                        # ab = AnnotationBbox(self.boat_img, (b_xs[bid][all_t], b_ys[bid][all_t]), frameon = False)
                        # ax.add_artist(ab)
                    else:
                        ab = AnnotationBbox(self.current_boat_img, (b_xs[bid][all_t], b_ys[bid][all_t]), frameon = False)
                        ax.add_artist(ab)
                        viewing_radius_bid = pat.Circle((b_xs[bid][all_t], b_ys[bid][all_t]), color = 'b', alpha=0.1, radius = self.tagging_radius)
                        plt.gca().add_patch(viewing_radius_bid)


                k_ = 2
                if len(times) <= 1:
                    continue
                if len(times) <= k_ + 1:
                    k_ = len(times) - 1
                x_tck = splrep(times, list(b_xs[bid].values()), k = k_)
                x_coords = splev(times, x_tck)
                y_tck = splrep(times, list(b_ys[bid].values()), k = k_)
                y_coords = splev(times, y_tck)
            
                plt.plot(x_coords, y_coords, linestyle = '--', c = 'orange')
            
                    
            plt.xlim(self.boundary_x[0] - 500, self.boundary_x[1] + 500)
            plt.ylim(self.boundary_y[0] - 500, self.boundary_y[1] + 500)
            plt.savefig(self.output_folder + 'frame'+str(t_)+'.png')
            plt.close()
            

    def savegif(self):
        
        images = []
        # folder = 'output/dominica_experiment/frames/'
        filenames = {int(f.split('frame')[1].split('.')[0]): self.output_folder + f for f in listdir(self.output_folder) if isfile(join(self.output_folder, f)) and 'frame' in f and 'png' in f}
        filenames = collections.OrderedDict(sorted(filenames.items()))
        for id, filename in filenames.items():
            images.append(imageio.imread(filename))
        imageio.mimsave(self.output_folder + 'movie.gif', images,  fps=3)

    def save_transtion_video(self):
        show_header = False
        show_after_rendezvous = False
        fig, ax = plt.subplots(figsize = (16,16))
        # ax.set_facecolor(color = [100/255,170/255,180/255])
        
        
        # ax.imshow(img)
        # print(plt.rcParams.keys())
        
        # plt.rcParams.update({'font.size': 44})#, 'xtick.labelsize': 'large', 'ytick.labelsize': 'large', 'axes.labelsize': 'large'})

        ax.set(xlim = [self.boundary_x[0] - 1500, self.boundary_x[1] + 1500], ylim = [self.boundary_y[0] - 1500, self.boundary_y[1] + 1500], \
            xlabel = 'Easting (in meters)', ylabel = 'Northing (in meters)', title = 'Top view of results of autonomous routing')
        ax.set_ylabel('Northing (in meters)', rotation=90)
        plt.setp(ax.get_yticklabels(), ha="right", rotation=90)

        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        if not self.paper_view:
            img = plt.imread("output/dominica_experiment/ocean_light.png")
            ax.imshow(img, extent=[x0, x1, y0, y1], aspect='auto', alpha = 0.9)

        ylim = [self.boundary_y[0] - 500, self.boundary_y[1] + 1500]
        xgap = 1000
        ygap = 1000
        if show_header:
            ax.annotate('Observation\nradius', xy = (self.boundary_x[0] - 0.25* xgap, ylim[1] - 1.25* ygap))
            ax.annotate('Agent', xy = (self.boundary_x[0] + xgap- 0.25* xgap, ylim[1] - 1.25* ygap))
            ax.annotate('Surfaced\nwhale', xy = (self.boundary_x[0] + 2 * xgap- 0.25* xgap, ylim[1] - 1.25* ygap))
            ax.annotate('Underwater\nwhale', xy = (self.boundary_x[0] + 3 * xgap- 0.25* xgap, ylim[1] - 1.25* ygap))
            ax.annotate('Whale\ntag', xy = (self.boundary_x[0] + 4 * xgap- 0.25* xgap, ylim[1] - 1.25* ygap))
            ax.plot([self.boundary_x[0] - 500, self.boundary_x[1] + 500], [ylim[1] - 1.5* ygap, ylim[1] - 1.5*ygap])

            viewing_radius_bid = pat.Circle((self.boundary_x[0], ylim[1] - 0.5 * ygap), color = 'w', alpha=0.3, radius = self.tagging_radius, label = 'Agent observation radius')
            ax.add_patch(viewing_radius_bid)
            current_boat_img = OffsetImage(Image.open("output/dominica_experiment/boat_new.png"), zoom = 0.08)
            ab = AnnotationBbox(current_boat_img, (self.boundary_x[0] + 2 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
            ax.add_artist(ab)
            current_surfaced_whale = OffsetImage(Image.open("output/dominica_experiment/surfaced3.png").rotate(90), zoom = 0.25)
            tag_image = OffsetImage(Image.open("output/dominica_experiment/tag.png"), zoom = 0.25)
            current_underwater_whale = OffsetImage(Image.open("output/dominica_experiment/underwater3.png").rotate(90), zoom = 0.25)    
            current_whale_ = AnnotationBbox(current_surfaced_whale, (self.boundary_x[0] + 3 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
            ax.add_artist(current_whale_)
            current_whale_ = AnnotationBbox(current_underwater_whale, (self.boundary_x[0] + 4 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
            ax.add_artist(current_whale_)
            tag_image_ = AnnotationBbox(tag_image, (self.boundary_x[0] + 5 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
            ax.add_artist(tag_image_)
        

        # plt.savefig(self.output_folder + 'frame'+str(t_)+'.png')
        # plt.close()
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
            
            # thetas = np.convolve(w_thetas[wid], np.ones(w), 'valid') / w
            w_thetas[wid] = [theta for theta in thetas] + [thetas[-1] for _ in range(w)]
            #[np.arctan2(new_w_ys[wid][t1+1] - new_w_ys[wid][t1], new_w_xs[wid][t1+1] - new_w_xs[wid][t1]) * 180 / np.pi \
                # if t2 < new_times[-1] else 0 for t1,t2 in enumerate(new_times) ]
                
        
        b_plot = {bid: ax.plot(new_b_xs[bid][0], new_b_ys[bid][0], c= "orange", linestyle = '--',  linewidth=2)[0] for bid in range(self.num_boats)} 
        
        w_scat = {wid: ax.scatter(new_w_xs[wid][0], new_w_ys[wid][0], \
            c=('cyan' if new_w_ups[wid][0] else 'black'), marker = '^') \
            for wid in range(self.num_whales)}
        b_scat = {bid: ax.scatter(new_b_xs[bid][0], new_b_ys[bid][0], c= "orange", marker = '*', s = 500) for bid in range(self.num_boats)} 

        # w_plot = {wid: ax.plot(new_w_xs[wid][0], new_w_ys[wid][0], c='blue', linestyle = '--',  linewidth=4)[0] for wid in range(self.num_whales)}
        # b_plot = {bid: ax.plot(new_b_xs[bid][0], new_b_ys[bid][0], c= "orange", linestyle = '--',  linewidth=4)[0] for bid in range(self.num_boats)} 


        # current_whale_ab = {}
        # for bid in range(self.num_boats):
        #     theta = 0
        #     viewing_radius_bid = pat.Circle((new_b_xs[bid][0], new_b_ys[bid][0]), color = 'w', alpha=0.5, radius = self.tagging_radius)
        #     observation_area[bid] = ax.add_patch(viewing_radius_bid)
        #     current_boat_img = OffsetImage(Image.open("output/dominica_experiment/boat_new.png").rotate(theta), zoom = 0.08)
        #     ab = AnnotationBbox(current_boat_img, (new_b_xs[bid][0], new_b_ys[bid][0]), frameon = False)
        #     ax.add_artist(ab)
        
        # for wid in range(self.num_whales):
        #     current_surfaced_whale = OffsetImage(Image.open("output/dominica_experiment/surfaced3.png").rotate(w_thetas[wid][0] - 90), zoom = 0.25)
        #     tag_image = OffsetImage(Image.open("output/dominica_experiment/tag.png").rotate(w_thetas[wid][0]), zoom = 0.25)
        #     current_underwater_whale = OffsetImage(Image.open("output/dominica_experiment/underwater3.png").rotate(w_thetas[wid][0] - 90), zoom = 0.25)

            
            
        #     if new_w_ups[wid]:
        #         current_whale_ab[wid] = AnnotationBbox(current_surfaced_whale, (new_w_xs[wid][0], new_w_ys[wid][0]), frameon = False)
        #         ax.add_artist(current_whale_ab[wid])
        #         # ax.draw_artist(current_whale_ab[wid])
        #     else:
        #         current_whale_ab[wid] = AnnotationBbox(current_underwater_whale, (new_w_xs[wid][0], new_w_ys[wid][0]), frameon = False)
        #         # ax.draw_artist(current_whale_ab[wid])
        #         ax.add_artist(current_whale_ab[wid])
        #     if new_w_as[wid] == 1:
        #         tag_image_ab = AnnotationBbox(tag_image, (new_w_xs[wid][0], new_w_ys[wid][0]), frameon = False)
        #         ax.add_artist(tag_image_ab)
            
        current_whale_assigned = [False for wid in range(self.num_whales)]

        def update(frame):
            for a in ax.artists:
                a.remove()
            for a in ax.patches:
                a.remove()

            if show_header:
                viewing_radius_bid = pat.Circle((self.boundary_x[0] , ylim[1] - 0.5 * ygap), color = 'w', alpha=0.5, radius = self.tagging_radius, label = 'Agent observation radius')
                ax.add_patch(viewing_radius_bid)
                current_boat_img = OffsetImage(Image.open("output/dominica_experiment/boat_new.png"), zoom = 0.08)
                ab = AnnotationBbox(current_boat_img, (self.boundary_x[0] + xgap, ylim[1] - 0.5 * ygap), frameon = False)
                ax.add_artist(ab)
                current_surfaced_whale = OffsetImage(Image.open("output/dominica_experiment/surfaced3.png").rotate(90), zoom = 0.25)
                tag_image = OffsetImage(Image.open("output/dominica_experiment/tag.png"), zoom = 0.25)
                current_underwater_whale = OffsetImage(Image.open("output/dominica_experiment/underwater3.png").rotate(90), zoom = 0.25)    
                current_whale_ = AnnotationBbox(current_surfaced_whale, (self.boundary_x[0] + 2 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
                ax.add_artist(current_whale_)
                current_whale_ = AnnotationBbox(current_underwater_whale, (self.boundary_x[0] + 3 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
                ax.add_artist(current_whale_)
                tag_image_ = AnnotationBbox(tag_image, (self.boundary_x[0] + 4 * xgap, ylim[1] - 0.5 * ygap), frameon = False)
                ax.add_artist(tag_image_)
            


            for bid in range(self.num_boats):
                theta = 0
                # plt.plot(new_b_xs[:frame], new_b_xs[:frame], linestyle = '--', c = 'orange')
                # data = np.stack([new_b_xs[bid][:frame], new_b_ys[bid][:frame]]).T
                # b_scat[bid].set_offsets(data)
                viewing_radius_bid = pat.Circle((new_b_xs[bid][frame], new_b_ys[bid][frame]), \
                    edgecolor = 'orange', linewidth = 5, linestyle = '--', radius = self.tagging_radius, fill = False)
                observation_area[bid] = ax.add_patch(viewing_radius_bid)
                # if show_after_rendezvous:
                b_plot[bid].set_xdata(new_b_xs[bid][:frame + 1])
                b_plot[bid].set_ydata(new_b_ys[bid][:frame + 1])
                
                if show_after_rendezvous:
                    current_boat_img = OffsetImage(Image.open("output/dominica_experiment/boat_new.png").rotate(theta), zoom = 0.05)
                    ab = AnnotationBbox(current_boat_img, (new_b_xs[bid][frame], new_b_ys[bid][frame]), frameon = False)
                    ax.add_artist(ab)
                else:
                    b_scat[bid].set_offsets([new_b_xs[bid][frame], new_b_ys[bid][frame]])
                

            for wid in range(self.num_whales):
                theta = 0
                # data = np.stack([new_w_xs[wid][:frame], new_w_ys[wid][:frame]]).T
                # w_scat[wid].set_offsets(data)
               
                if show_after_rendezvous or not current_whale_assigned[wid]:
                    # w_scat[wid].set_xdata(new_w_xs[wid][:frame+1])
                    # w_scat[wid].set_ydata(new_w_ys[wid][:frame+1])
                                    
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
                current_surfaced_whale = OffsetImage(Image.open("output/dominica_experiment/surfaced3.png").rotate(w_thetas[wid][frame] - 90), zoom = 0.2)
                tag_image = OffsetImage(Image.open("output/dominica_experiment/tag.png").rotate(w_thetas[wid][frame]), zoom = 0.250)
                current_underwater_whale = OffsetImage(Image.open("output/dominica_experiment/underwater3.png").rotate(w_thetas[wid][frame] - 90), zoom = 0.20)
                
                if show_after_rendezvous:
                    if new_w_ups[wid][frame]:
                        ab = AnnotationBbox(current_surfaced_whale, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                        ax.add_artist(ab)
                    else:
                        ab = AnnotationBbox(current_underwater_whale, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                        ax.add_artist(ab)
                    if new_w_as[wid][frame] == 1:
                        ab = AnnotationBbox(tag_image, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                        ax.add_artist(ab)
                else:
                    if new_w_as[wid][frame] == 1:
                        if not current_whale_assigned[wid]:
                            # ab = AnnotationBbox(current_surfaced_whale, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                            # ax.add_artist(ab)
                            # ab = AnnotationBbox(tag_image, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                            # ax.add_artist(ab)
                            current_whale_assigned[wid] = True
                            
                    # else:
                    #     if new_w_ups[wid][frame]:
                    #         ab = AnnotationBbox(current_surfaced_whale, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                    #         ax.add_artist(ab)
                    #     else:
                    #         ab = AnnotationBbox(current_underwater_whale, (new_w_xs[wid][frame], new_w_ys[wid][frame]), frameon = False)
                    #         ax.add_artist(ab)
                    
                        


                
            
                
            # print('Here ', frame)
            plt.savefig(self.output_folder + 'frame'+str(frame)+'.png')
            # return w_scat, b_plot, current_whale_ab
        
        ani = animation.FuncAnimation(fig = fig, func = update, frames = len(new_times), interval = 120, repeat = False, blit = False)
        # ani2 = animation.ArtistAnimation(fig = fig, func = update2, frames = len(new_times), interval = 30, repeat = False, blit = False)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        ani.save(filename=self.output_folder + "AVATAR_autonomy_example.gif", writer="pillow")
        # plt.show()
        print('')
       
            # plt.savefig(self.output_folder + 'frame'+str(t_)+'.png')
            # plt.close()
if __name__ == '__main__':
    # foldername = 'rebuttal_output_Nov23/Combined_Dominica_Data_r500_w5_a3/Run_-1/MA_rollout/'
    # foldername = 'rebuttal_output_Nov23/Combined_Dominica_Data_r500_w5_a3/Run_-1/MA_rollout/'
    foldername = 'visualize_output/'
    # v = Visualize_file('output/dominica_experiment/state_log.csv', 500, process_flag = True)

    v = Visualize_file(foldername, 'state_log.csv', 500, process_flag = True, number_of_observations_per_minute = 60, paper_view = True, legacy=True)

    v.save_transtion_video()
    # time;number_of_boats;b_x;b_y;b_theta;b_v;number_of_whales;w_x;w_y;w_theta;w_v;assigned_whales
    # 0; t
    # 2; n_b
    # 27257.764625898308,29458.26020613228; b_x
    # 22144.458345054096,13476.92372739528; b_y
    # -1.360811357793896,-2.431009429274897;
    # 900.0,900.0;
    # 4; n_w
    # 29712.56780244724,28334.717232897994,27177.7091759504,29877.11231959955; w_x
    # 19264.492817017417,20010.328614313654,12584.401446147185,13598.560015792444; w_y
    # 2.987536481116627,-0.9842328388365635,-1.344336572781773,1.8387103603840733;
    # 2.8734999835709605,0.2471492619335916,0.24890004288560186,0.24877039829180295;
    # ; w_a
    # -8,-18,-28,-16; last_up
    # 0,-9,-19,-7 last down

from mpl_toolkits.mplot3d.proj3d import proj_transform


if __name__ == '__main__3d':
    fig = plt.figure()

    boundary_x = [0, 300]
    boundary_y = [0, 300]
    xx, yy = np.meshgrid(range(boundary_x[0], boundary_x[1]), \
        range(boundary_y[0], boundary_y[1]))
    # ax.set_xlim(boundary_x[0], boundary_x[1])
    # ax.set_ylim(boundary_y[0], boundary_y[1])

    
    ax = fig.add_subplot(projection = '3d')

    # ax.plot_surface(xx, yy, z_surface, alpha=0.2)
    # ax.plot_surface(xx, yy, z_bottom, alpha=0.2)

    

    delta = theta = 20
    thetas = np.arange(0, 360 - delta, delta)
    bxs = np.random.uniform(boundary_x[0], boundary_x[1], len(thetas))
    bys = np.random.uniform(boundary_y[0], boundary_y[1], len(thetas))

    wxs = np.random.uniform(boundary_x[0], boundary_x[1], len(thetas))
    wys = np.random.uniform(boundary_y[0], boundary_y[1], len(thetas))

    # xs, ys, zs = proj_transform(bxs, bys, np.zeros(len(bxs)), ax.get_proj())

    # x_min1 = int(np.floor(min(xs)))
    # x_max1 = int(np.ceil(max(xs)))
    # y_min1 = int(np.floor(min(ys)))
    # y_max1 = int(np.ceil(max(ys)))
    # xx, yy = np.meshgrid(range(x_min1, x_max1), \
    #     range(y_min1, y_max1))
    z_bottom = -1 * np.ones(shape = xx.shape)
    z_surface = np.zeros(shape = xx.shape)

    ax.plot_surface(xx, yy, z_surface, alpha=0.2)
    ax.plot_surface(xx, yy, z_bottom, alpha=0.2)

    # for tid, theta in enumerate(thetas):
    #     boat_image = OffsetImage(Image.open("output/dominica_experiment/boat2.png").rotate(theta), zoom=1)
        
    #     ab = AnnotationBbox(boat_image, (xs[tid], ys[tid]), frameon=False)
    #     ax.add_artist(ab)


    from matplotlib.cbook import get_sample_data
    import cv2
    fn = get_sample_data("output/dominica_experiment/boat2.png", asfileobj=False)
    img = cv2.imread(fn)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # boat_image = OffsetImage(Image.open("output/dominica_experiment/boat2.png").rotate(thetas[0]), zoom=1)
    # ax.plot_surface(bxs[0], bys[0], np.atleast_2d(0), facecolors=img)

    # for tid, theta in enumerate(thetas):
    #     whale_image = OffsetImage(Image.open("output/dominica_experiment/whale.png").rotate(theta), zoom=0.1)
    #     ab = AnnotationBbox(whale_image, ([wxs[tid], wys[tid]], 0), frameon=False)
    #     ax.add_artist(ab)
    
    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    img = img / 255
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, np.atleast_2d(0), rstride=10, cstride=10, facecolors=img)


    # print(xs, ys)
    plt.savefig('output/dominica_experiment/frame_3d.png')
    # plt.show()
    print('')