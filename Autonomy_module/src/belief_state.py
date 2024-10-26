import numpy as np
import typing as t
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import math, do_mpc
from casadi import *

from src.configs.constants import *
from src.global_knowledge import Global_knowledge




class Belief_State:
    number_of_boats : int
    number_of_whales :int
    b_x: np.ndarray
    b_y: np.ndarray
    b_theta: np.ndarray
    b_v: np.ndarray
    w_x: np.ndarray
    w_y: np.ndarray
    w_theta : np.ndarray
    w_v : np.ndarray
    assigned_whales : np.ndarray
    whale_up2 : np.ndarray
    w_Pcov: np.ndarray

    w_last_surface_start_time: np.ndarray 
    w_last_surface_end_time: np.ndarray 
    
    def __init__(self, knowledge: Global_knowledge, state_str:str = None, time: int = 0, number_of_boats: int = 1, b_x: np.ndarray = None, b_y: np.ndarray = None, b_theta: np.ndarray = None, b_v: np.ndarray = None, \
        number_of_whales: int = 2, w_x: np.ndarray = None, w_y: np.ndarray = None, w_theta: np.ndarray = None, w_v: np.ndarray = None, assigned_whales: t.List = [], \
            w_last_surface_start_time: np.ndarray = None, w_last_surface_end_time: np.ndarray = None, Pcov: np.ndarray = None, debug = False) -> None: 
        self.knowledge = knowledge 
        if state_str == None:
            self.time = time
            self.number_of_boats = number_of_boats
            self.b_x = np.zeros(number_of_boats) if b_x is None else b_x
            self.b_y = np.zeros(number_of_boats) if b_y is None else b_y
            self.b_theta = np.zeros(number_of_boats) if b_theta is None else b_theta
            self.b_v = self.knowledge.boat_max_speed_mtpm * np.ones(number_of_boats) if b_v is None else b_v

            self.number_of_whales = number_of_whales
            self.w_x = np.random.uniform(self.knowledge.boundary['data_min_x'], self.knowledge.boundary['data_max_x'], size = number_of_whales) if w_x is None else w_x
            self.w_y = np.random.uniform(self.knowledge.boundary['data_min_y'], self.knowledge.boundary['data_max_y'], size = number_of_whales) if w_y is None else w_y
            self.w_theta = np.random.uniform(-np.pi, np.pi, size = number_of_whales) if w_theta is None else w_theta
            self.w_v = Whale_speed_mtpm * np.ones(number_of_whales) if w_v is None else w_v
            self.w_Pcov = Pcov
            
            
            self.assigned_whales = assigned_whales

            self.w_last_surface_start_time = w_last_surface_start_time
            self.w_last_surface_end_time = w_last_surface_end_time
            self.whale_up2 = np.array([True if self.w_last_surface_start_time[wid] <= self.time \
                and (self.w_last_surface_end_time[wid] is None or math.isnan(self.w_last_surface_end_time[wid]) or \
                    self.w_last_surface_end_time[wid] < self.w_last_surface_start_time[wid]) \
                        else False for wid in range(self.number_of_whales)])

            
        else:
            self.add_to_state_from_str(state_str, debug)
            
        
        

    def state_copy(self, bid: int, wid: int):
        if not isinstance(wid, list):
            return Belief_State(knowledge = self.knowledge, state_str = None, time = self.time, number_of_boats = 1, \
                b_x = np.array([self.b_x[bid]]), b_y = np.array([self.b_y[bid]]), \
                    b_theta = np.array([self.b_theta[bid]]), b_v = np.array([self.b_v[bid]]), \
                        number_of_whales = 1, w_x = np.array([self.w_x[wid]]), w_y = np.array([self.w_y[wid]]), \
                            w_theta = np.array([self.w_theta[wid]]), w_v = np.array([self.w_v[wid]]), \
                                    assigned_whales = [], \
                                        w_last_surface_start_time = np.array([self.w_last_surface_start_time[wid]]), \
                                            w_last_surface_end_time = np.array([self.w_last_surface_end_time[wid]]))
        else:
            return Belief_State(knowledge = self.knowledge, state_str = None, time = self.time, number_of_boats = 1, \
                b_x = np.array([self.b_x[bid]]), b_y = np.array([self.b_y[bid]]), \
                    b_theta = np.array([self.b_theta[bid]]), b_v = np.array([self.b_v[bid]]), \
                        number_of_whales = len(wid), w_x = np.array([self.w_x[w] for w in wid]), w_y = np.array([self.w_y[w] for w in wid]), \
                            w_theta = np.array([self.w_theta[w]for w in wid]), w_v = np.array([self.w_v[w] for w in wid]), \
                                    assigned_whales = [], \
                                        w_last_surface_start_time = np.array([self.w_last_surface_start_time[w] for w in wid]), \
                                            w_last_surface_end_time = np.array([self.w_last_surface_end_time[w] for w in wid]))

    def add_to_state_from_str(self, state_str, debug):
        state_str_elems = state_str.split(';')
        if debug:
            print(state_str_elems)
        self.time = int(state_str_elems[0])
        self.number_of_boats = int(state_str_elems[1])
        self.b_x = np.array([float(ai) for ai in state_str_elems[2].split(',')])
        self.b_y = np.array([float(ai) for ai in state_str_elems[3].split(',')])
        self.b_theta = np.array([float(ai) for ai in state_str_elems[4].split(',')])
        self.b_v = np.array([float(ai) for ai in state_str_elems[5].split(',')])

        self.number_of_whales = int(state_str_elems[6])
        self.w_x = np.array([float(ai) for ai in state_str_elems[7].split(',')])
        self.w_y = np.array([float(ai) for ai in state_str_elems[8].split(',')])
        self.w_theta = np.array([float(ai) for ai in state_str_elems[9].split(',')])
        self.w_v = np.array([float(ai) for ai in state_str_elems[10].split(',')])
        self.whale_up2 = np.array([True if ai == 'True' else False for ai in state_str_elems[11].split(',')])
        
        if state_str_elems[12] != '\n' and state_str_elems[12] != '':
            self.assigned_whales = [int(ai) for ai in state_str_elems[12].rstrip('\n').split(',')]
        else:
            self.assigned_whales = []

        if len(state_str_elems) > 13 and state_str_elems[13] != '\n' and state_str_elems[13] != '':
            self.w_last_surface_start_time = [int(ai) for ai in state_str_elems[13].rstrip('\n').split(',')]

        if len(state_str_elems) > 14 and state_str_elems[14] != '\n' and state_str_elems[14] != '':
            self.w_last_surface_end_time = [int(ai) if ai != 'None' and ai != 'nan' else None for ai in state_str_elems[14].rstrip('\n').split(',')]



    def next_state(self, control: Boat_Control, observations_x_y_v_theta_up = None, ground_truth_for_evaluation = None, Pcov = None, b_xys = None, w_assigned = None):
        time_delta = 1
        self.Pcov = Pcov
        
        self.time += time_delta
        
        if b_xys is None:
            self.b_x += time_delta * control.b_v * np.cos(control.b_theta).reshape(self.b_x.shape)
            self.b_y += time_delta * control.b_v * np.sin(control.b_theta).reshape(self.b_y.shape)
        else:
            self.b_x = b_xys[0]
            self.b_y = b_xys[1]
        
        
        self.b_theta = control.b_theta.reshape(self.b_theta.shape)
        self.b_v = time_delta * control.b_v.reshape(self.b_v.shape)

        if observations_x_y_v_theta_up is None:

            self.w_x += time_delta * self.w_v * np.cos(self.w_theta).reshape(self.w_x.shape)
            self.w_y += time_delta * self.w_v * np.sin(self.w_theta).reshape(self.w_y.shape)

        
        else:
            self.w_x = observations_x_y_v_theta_up[:, 0]
            self.w_y = observations_x_y_v_theta_up[:, 1]
            self.w_v = observations_x_y_v_theta_up[:, 2]
            self.w_theta = observations_x_y_v_theta_up[:, 3]

        
        for wid in range(self.number_of_whales):
            if self.whale_up2[wid] == False and observations_x_y_v_theta_up[wid, 4] == True:
                self.whale_up2[wid] = True
                self.w_last_surface_start_time[wid] = self.time
            elif self.whale_up2[wid] == True and observations_x_y_v_theta_up[wid, 4] == False:
                self.whale_up2[wid] = False
                self.w_last_surface_end_time[wid] = self.time - 1
                
        self.agent_loc_aoa = {}
        for wid in range(self.number_of_whales):
            if wid in self.assigned_whales:
                continue
            if self.knowledge.overlay_GPS and (ground_truth_for_evaluation is None or ground_truth_for_evaluation[wid] is None):
                continue
            if w_assigned is not None and w_assigned[wid] is not None and w_assigned[wid] and wid not in self.assigned_whales:                 
                self.assigned_whales.append(wid)
                continue
            if ground_truth_for_evaluation is None:

                if any([np.sqrt( (self.w_x[wid] - self.b_x[bid])**2 + (self.w_y[wid] - self.b_y[bid])**2 ) \
                    <= self.knowledge.tagging_distance \
                        and self.whale_up2[wid] for bid in range(self.number_of_boats)]) :
                    self.assigned_whales.append(wid)
            else:
                continue
                
                                
        return self.stage_cost()

    
    

    def terminal(self):
        if len(self.assigned_whales) == self.number_of_whales:
            return True
        return False

    def stage_cost(self):
        return 0
        
    def terminal_cost(self):
        return self.stage_cost()

    def plot_state(self, path = None, plot = True):
        plt.ioff()
        plt.cla()
        

        
        
        if not hasattr(self, 'history'):
            self.history = {'boats': {bid: {'x': [], 'y': []} for bid in range(self.number_of_boats)}, \
                'whales': {wid: {'x': [], 'y': [], 'up': []}  for wid in range(self.number_of_whales)}}
        
        min_x = np.iinfo('i').max
        min_y = np.iinfo('i').max
        max_x = np.iinfo('i').min
        max_y = np.iinfo('i').min
        for bid in range(self.number_of_boats):
            p = self.b_x[bid]
            q = self.history['boats'][bid]['x']
            if isinstance(p, list):
                self.history['boats'][bid]['x'].extend(self.b_x[bid]) 
                self.history['boats'][bid]['y'].extend(self.b_y[bid]) 
            else:
                self.history['boats'][bid]['x'].append(self.b_x[bid]) 
                self.history['boats'][bid]['y'].append(self.b_y[bid]) 
            min_x = min(min_x, min(self.history['boats'][bid]['x']))
            min_y = min(min_y, min(self.history['boats'][bid]['y']))
            max_x = max(max_x, max(self.history['boats'][bid]['x']))
            max_y = max(max_y, max(self.history['boats'][bid]['y']))

        for wid in range(self.number_of_whales):
            if isinstance(self.w_x[wid], list):
                self.history['whales'][wid]['x'].extend(self.w_x[wid]) 
                self.history['whales'][wid]['y'].extend(self.w_y[wid]) 
                if wid in self.assigned_whales:
                    self.history['whales'][wid]['up'].append(2)
                else:
                    self.history['whales'][wid]['up'].extend(self.whale_up2[wid])
            else:
                self.history['whales'][wid]['x'].append(self.w_x[wid]) 
                self.history['whales'][wid]['y'].append(self.w_y[wid]) 
                if wid in self.assigned_whales:
                    self.history['whales'][wid]['up'].append(2)
                else:
                    self.history['whales'][wid]['up'].append(self.whale_up2[wid])

            min_x = min(min_x, min(self.history['whales'][wid]['x']))
            min_y = min(min_y, min(self.history['whales'][wid]['y']))
            max_x = max(max_x, max(self.history['whales'][wid]['x']))
            max_y = max(max_y, max(self.history['whales'][wid]['y']))
        
        cols = {}
        w_sizes = {}
        for bid in range(self.number_of_boats):
            plt.plot(self.history['boats'][bid]['x'], self.history['boats'][bid]['y'])
        plt.scatter(self.b_x, self.b_y, c = 'brown', label = 'Boat', alpha = 0.5, s = mpl.rcParams['lines.markersize']*2)
        for bid in range(self.number_of_boats):
            plt.text(self.b_x[bid], self.b_y[bid], 'b'+str(bid), fontsize=12)
        for wid in range(self.number_of_whales):

            royalblue = [65/255, 105/255, 225/255]
            cols[wid] = np.array([royalblue if up ==1 else [0, 0, 0] if up ==0 else [1,0,0] for up in self.history['whales'][wid]['up']])
            
            w_sizes[wid] = np.array([8 if up == 1 else 2 if up ==0 else 5 for up in self.history['whales'][wid]['up']])

            plt.scatter(self.history['whales'][wid]['x'], self.history['whales'][wid]['y'], c = cols[wid], s = w_sizes[wid])
        
        for wid in range(self.number_of_whales):
            plt.scatter(self.w_x[wid], self.w_y[wid], label = 'Whale', c = np.array([cols[wid][-1]]))
            plt.text(self.w_x[wid], self.w_y[wid], 'w'+str(wid), fontsize=12)
        
        
        plt.xlim(min_x - 500, max_x + 500)
        plt.ylim(min_y - 500, max_y + 500)
        
        if plot == False:
            return 
        if path is None:
            plt.savefig(self.knowledge.output_foldername_deprecated + 'state_'+str(self.time)+ '.png')
        else:
            plt.savefig(path  + 'new_state_'+str(self.time) + '.png')

    def state_string(self, first_line = False):
        state_str = ""
        if first_line:
            state_str += "time;number_of_boats;b_x;b_y;b_theta;b_v;number_of_whales;w_x;w_y;w_theta;w_v;assigned_whales\n"
            
        state_str += str(self.time) + ";" + \
            str(self.number_of_boats) + ";" + ','.join([str(ai) for ai in self.b_x]) + ";" + ','.join([str(ai) for ai in self.b_y]) + ";" + ','.join([str(ai) for ai in self.b_theta]) + ";" + ','.join([str(ai) for ai in self.b_v]) \
                + ";" + str(self.number_of_whales) + ";" + ','.join([str(ai) for ai in self.w_x]) + ";" + ','.join([str(ai) for ai in self.w_y]) + ";" + ','.join([str(ai) for ai in self.w_theta]) + ";" + ','.join([str(ai) for ai in self.w_v]) \
                    + ";" + ','.join([str(ai) for ai in self.whale_up2]) + ";" + ','.join([str(ai) for ai in self.assigned_whales]) + ";" \
                        + ','.join([str(ai) for ai in self.w_last_surface_start_time])+ ";" + ','.join([str(ai) if ai is not None else 'None' for ai in self.w_last_surface_end_time]) \
                            + ';' + ','.join([str(self.w_Pcov[wid][0,0]) + '|' + str(self.w_Pcov[wid][1,1]) for wid in range(self.number_of_whales)]) + "\n"
        return state_str


class Policy_base:
    def __init__(self) -> None:
        self.policy_name = "No policy"

    def get_control(self, belief: Belief_State):
        return Boat_Control(b_theta = np.zeros(belief.number_of_boats), b_v = np.zeros(belief.number_of_boats))

    def setup_MPC_for_last_mile(self, knowledge: Global_knowledge, dt_sec: int):
    
        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)
 
        t = model.set_variable(var_type='_x', var_name='t', shape=(1, 1))
        b_x = model.set_variable(var_type='_x', var_name='b_x', shape=(1,1))
        b_y = model.set_variable(var_type='_x', var_name='b_y', shape=(1,1))
        b_theta = model.set_variable(var_type='_x', var_name='b_theta', shape=(1,1))
        w_x = model.set_variable(var_type='_x', var_name='w_x', shape=(1, 1))
        w_y = model.set_variable(var_type='_x', var_name='w_y', shape=(1, 1))
        u_b_theta = model.set_variable(var_type='_u', var_name='u_b_theta')
        u_b_v = model.set_variable(var_type='_u', var_name='u_b_v')
      
        
        model.set_rhs('t', vertcat(dt_sec))
        model.set_rhs('b_x', b_x + dt_sec * (u_b_v) * np.cos(u_b_theta))
        model.set_rhs('b_y', b_y + dt_sec * (u_b_v) * np.sin(u_b_theta))
        model.set_rhs('b_theta', u_b_theta)
        model.set_rhs('w_x', w_x)
        model.set_rhs('w_y', w_y)
        
    
        model.setup()

        self.mpc = do_mpc.controller.MPC(model)
        n_horizon = 10
        setup_mpc = {'n_horizon': n_horizon, 't_step': 1, 'n_robust': 1, 'store_full_solution': True, \
            'nlpsol_opts':{'ipopt.print_level':0, 'print_time':0}} 
        
        self.mpc.set_param(**setup_mpc)


        lterm = (w_x - b_x)**2 + (w_y - b_y)**2
        mterm = (w_x - b_x)**2 + (w_y - b_y)**2
    
        self.mpc.set_objective(mterm = mterm, lterm = lterm)
        theta_penalty = 1e-3
        v_penalty = 0
        self.mpc.set_rterm(u_b_theta = theta_penalty, u_b_v = v_penalty)


        
        self.mpc.bounds['lower','_u', 'u_b_v'] = 0
        self.mpc.bounds['upper','_u', 'u_b_v'] = knowledge.boat_max_speed_mtpm * dt_sec / 60

        self.mpc.setup()

        self.simulator = do_mpc.simulator.Simulator(model)
        self.simulator.set_param(t_step = 1)
        self.simulator.setup()
        self.us = {}

    def get_MPC_control(self, state: Belief_State, bid, wid):
        
        if state.time % state.knowledge.observations_per_minute == 0:
     
            self.us[bid] = []

            x0 = np.array([state.time, state.b_x[bid], state.b_y[bid], state.b_theta[bid], state.w_x[wid], state.w_y[wid]])
            x0 = x0.reshape(-1,1)

            self.mpc.reset_history()
            self.simulator.reset_history()
            self.mpc.x0 = x0

            self.mpc.set_initial_guess()

            u0 = np.array([state.b_theta[0], state.b_v[0]]).reshape(2,1)
        
            self.simulator.x0 = x0
            for t_ in range(state.knowledge.observations_per_minute):
                u0 = self.mpc.make_step(x0)
                x0 = self.simulator.make_step(u0)
                self.us[bid].append(u0)
            u0 = self.us[bid][0]
        else:
            u0 = self.us[bid][state.time % state.knowledge.observations_per_minute]
        
        return u0
    
