from src.configs.constants import Boat_Control
from src.belief_state import Belief_State, Policy_base
from src.global_knowledge import Global_knowledge
import numpy as np
from ortools.linear_solver import pywraplp    
from scipy.optimize import linear_sum_assignment

import sys

class IA_with_commitment_science(Policy_base):
    def __init__(self, knowledge: Global_knowledge, state: Belief_State = None, CE = False) -> None:
        self.num_scenarios = 50
        self.knowledge = knowledge
        self.div = self.knowledge.observations_per_minute
        self.setup_MPC_for_last_mile(knowledge, 60 / self.knowledge.observations_per_minute)
        self.policy_name = 'IA_replan'
        self.CE = CE
        
        if self.CE:
            self.future_scenario_ia = {wid:[] for wid in range(state.number_of_whales)}
            for wid in range(state.number_of_whales):
                if state.whale_up2[wid] == True:
                    interval_end_time = int(state.w_last_surface_start_time[wid]/self.div + state.knowledge.surface_time_mean/self.div)
                    self.future_scenario_ia[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                else:
                    interval_end_time = int(state.w_last_surface_end_time[wid]/self.div)
                    self.future_scenario_ia[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                while interval_end_time < int(state.knowledge.n_horizon_for_evaluation/self.div):
                    interval_start_time = int(interval_end_time + state.knowledge.down_time_mean/self.div)
                    interval_end_time = int(interval_start_time + state.knowledge.surface_time_mean/self.div)
                    self.future_scenario_ia[wid].append((interval_start_time, interval_end_time))
            

    def get_cost_of_single_assignment_for_a_scene(self, state: Belief_State):
    
        
        wxdot_np = np.array([0] + [state.w_v[0] * np.cos(state.w_theta[0])] * int(self.knowledge.n_horizon_for_evaluation/self.div))
        wydot_np = np.array([0] + [state.w_v[0] * np.sin(state.w_theta[0])] * int(self.knowledge.n_horizon_for_evaluation/self.div))
        wx_np = np.cumsum(wxdot_np + state.w_x[0] * np.eye(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1)[0])
        wy_np = np.cumsum(wydot_np + state.w_y[0] * np.eye(int(self.knowledge.n_horizon_for_evaluation /self.div)+ 1)[0])

        bx_np = state.b_x[0] + np.eye(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1)[0]
        by_np = state.b_y[0] + np.eye(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1)[0]

        whale_ups = np.zeros(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1)
        for interval in state.surface_interval_scenario[0]:
            if interval[1] >= int(state.time/self.div) and interval[0] <= int(state.time/self.div + self.knowledge.n_horizon_for_evaluation/self.div):
                whale_ups[ max(0, interval[0] - int(state.time/self.div) ) : min(interval[1] - int(state.time/self.div) + 1, \
                    int(self.knowledge.n_horizon_for_evaluation/self.div)) ] = 1

        distances = np.zeros(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1)
        assigned_index = -1
        for n in range(1, int(self.knowledge.n_horizon_for_evaluation/self.div) + 1):
           
            btheta = np.arctan2(wy_np[n-1] - by_np[n-1], wx_np[n-1] - bx_np[n-1])
            distances[n-1] = np.sqrt((wy_np[n-1] - by_np[n-1])**2 + (wx_np[n-1] - bx_np[n-1])**2)
            bv = min(self.knowledge.boat_max_speed_mtpm, distances[n-1])
            bx_np[n] = bx_np[n-1] + bv * np.cos(btheta)
            by_np[n] = by_np[n-1] + bv * np.sin(btheta) 
        
            if n == 1:
                btheta0 = btheta
                bv0 = bv
            if assigned_index == - 1 and whale_ups[n - 1] and distances[n-1] < int(state.knowledge.tagging_distance/self.div):
                assigned_index = n - 1
                break
            
        

        u0_final2 = np.array([btheta0, bv0])


        assigned = np.zeros(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1).astype(int)
        assigned[assigned_index:] = 1
        

        avg_cost = int(self.knowledge.n_horizon_for_evaluation/self.div) + 1 - np.sum(assigned) 

        
        avg_cost_distance_based = np.sum(distances[:avg_cost])/state.knowledge.boat_max_speed_mtpm

        
        return avg_cost, u0_final2, 'states_greedy', avg_cost_distance_based

    
    def get_cost_of_single_assignment(self, state: Belief_State):
        
        avg_cost = 0

        avg_cost_distance_based = 0
        
        states_greedy = [ [state.b_x[0], state.b_y[0], state.w_x[0], state.w_v[0] * np.cos(state.w_theta[0]), \
            state.w_y[0], state.w_v[0] * np.sin(state.w_theta[0])]  ]
    
        for n in range(1, int(self.knowledge.n_horizon_for_evaluation /self.div) + 1):
            bx = states_greedy[n-1][0]
            by = states_greedy[n-1][1]
            
            wx = states_greedy[n-1][2]
            wxdot = states_greedy[n-1][3]
            wy = states_greedy[n-1][4]
            wydot = states_greedy[n-1][5]

            
            btheta = np.arctan2(wy - by, wx - bx)
            bv = min(self.knowledge.boat_max_speed_mtpm, np.sqrt((wy - by)**2 + (wx - bx)**2))
            states_greedy.append([bx + bv * np.cos(btheta), by + bv * np.sin(btheta), wx + wxdot, wxdot, wy + wydot, wydot ])

            if n==1:
                u0_final2 = np.array([btheta, bv])

            
    
        
        distances = [np.sqrt((st[0]-st[2])**2 + (st[1]-st[4])**2) for st in states_greedy]

        for n in range(self.num_scenarios):
            up_arrays = np.maximum(self.knowledge.surface_time_var/self.div, np.random.normal(self.knowledge.surface_time_mean/self.div, self.knowledge.surface_time_var/self.div, \
                size = int(self.knowledge.n_horizon_for_evaluation/self.div))).astype(int)
            down_arrays = np.maximum(self.knowledge.down_time_var/self.div, np.random.normal(self.knowledge.down_time_mean/self.div, self.knowledge.down_time_var/self.div, \
                size = int(self.knowledge.n_horizon_for_evaluation/self.div))).astype(int)
            if state.whale_up2[0] == True:
                up_arrays[0] = max(up_arrays[0], int(state.time/self.div) + 1 - int(state.w_last_surface_start_time[0]/self.div))
                whale_ups = [1]*(up_arrays[0] - int(state.time/self.div) + int(state.w_last_surface_start_time[0]/self.div))  \
                    + [0] * down_arrays[0]
                for n_ in range(1, int(self.knowledge.n_horizon_for_evaluation/self.div)):
                    whale_ups.extend([1] * up_arrays[n_])
                    whale_ups.extend([0] * down_arrays[n_])
            else:
                down_arrays[0] = max(down_arrays[0], int(state.time/self.div) + 1 - int(state.w_last_surface_end_time[0]/self.div))
                whale_ups = [0]*(down_arrays[0] - int(state.time/self.div) + int(state.w_last_surface_end_time[0]/self.div)) \
                    + [1] * up_arrays[0]
                for n_ in range(1, int(self.knowledge.n_horizon_for_evaluation /self.div)):
                    whale_ups.extend([0]*down_arrays[n_])
                    whale_ups.extend([1]*up_arrays[n_])

            assigned = np.array([0]*(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1))
            for n_ in range(1, int(self.knowledge.n_horizon_for_evaluation/self.div) + 1):
                
                assigned[n_] = min(1, assigned[n_ -1] + (1 - assigned[n_ -1]) * whale_ups[n_] * (distances[n_] <= self.knowledge.tagging_distance))
                
    
            avg_cost += np.sum( [ 1 for st_id in range(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1) if not assigned[st_id]  ] )

            sum_d = sum(distances)
            
            avg_cost_distance_based += np.sum( [ distances[st_id] for st_id in range(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1) if not assigned[st_id]  ] )
            
        avg_cost /= self.num_scenarios
        avg_cost_distance_based /= self.num_scenarios


        
        return avg_cost, u0_final2, states_greedy, avg_cost_distance_based

    

    def update_future_ia(self, state_orig : Belief_State):
        surface_start_times = {wid: [t[0] for t in self.future_scenario_ia[wid]] \
            for wid in range(state_orig.number_of_whales)}
        surface_end_times = {wid: [t[1] for t in self.future_scenario_ia[wid]] \
            for wid in range(state_orig.number_of_whales)}
        wids_to_update = [wid for wid in range(state_orig.number_of_whales) \
            if (state_orig.time == state_orig.w_last_surface_start_time[wid] \
                and int(state_orig.time/self.div) not in surface_start_times[wid]) \
                    or (state_orig.time == state_orig.w_last_surface_end_time[wid] \
                        and int(state_orig.time/self.div) not in surface_end_times[wid]) ]
        if len(wids_to_update) >= 1:
            for wid in wids_to_update:
                number_of_intervals = int(3 * np.ceil(state_orig.knowledge.n_horizon_for_evaluation \
                    / (state_orig.knowledge.down_time_mean + state_orig.knowledge.surface_time_mean)))
            
                interval_up_times = [int(state_orig.knowledge.surface_time_mean/self.div)]* number_of_intervals
                interval_down_times = [int(state_orig.knowledge.down_time_mean/self.div)]* number_of_intervals
            
                if state_orig.whale_up2[wid] == True:
                    interval_end_time = int(state_orig.w_last_surface_start_time[wid]/self.div) + interval_up_times[0]
                    self.future_scenario_ia[wid] = [(int(state_orig.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                else:
                    interval_end_time = int(state_orig.w_last_surface_end_time[wid]/self.div)
                    self.future_scenario_ia[wid] = [(int(state_orig.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                int_id = 1
                while interval_end_time < int(state_orig.knowledge.n_horizon_for_evaluation/self.div):
                    interval_start_time = interval_end_time + interval_down_times[int_id]
                    interval_end_time = interval_start_time + interval_up_times[int_id]
                    self.future_scenario_ia[wid].append((interval_start_time, interval_end_time))
                    int_id += 1


    def get_control_assignment(self, state: Belief_State, scene = False, bids_to_exclude = []):
        nbid_to_bid = [bid for bid in range(state.number_of_boats) if bid not in bids_to_exclude]
        nwid_to_wid = [wid for wid in range(state.number_of_whales) if wid not in state.assigned_whales]
        len_nwid_to_wid = len(nwid_to_wid)
        len_nbid_to_bid = len(nbid_to_bid)
        cost = np.zeros((len_nbid_to_bid, len_nwid_to_wid))
        u0 = np.zeros((len_nbid_to_bid, len_nwid_to_wid, 2))
        
        if self.CE:
            self.update_future_ia(state)

        for nbid in range(len_nbid_to_bid):
            for nwid in range(len_nwid_to_wid):
                wid = nwid_to_wid[nwid]
                bid = nbid_to_bid[nbid]
                st = state.state_copy(bid, wid)
                
                if self.CE == True:
                    st.surface_interval_scenario = [self.future_scenario_ia[wid]]
                    output = self.get_cost_of_single_assignment_for_a_scene(st)
                elif scene == True:
                    st.surface_interval_scenario = [state.surface_interval_scenario[wid]]
                    output = self.get_cost_of_single_assignment_for_a_scene(st)
                else:
                    output = self.get_cost_of_single_assignment(st)
                cost[nbid, nwid] = output[0]
                u0[nbid, nwid, :] = output[1].reshape(2)
                
        assinged_nbids, assigned_nwids = linear_sum_assignment(cost)
        assignment_bid_wid = {}
        cost_bid_wid = {(bid, wid): sys.maxsize for bid in range(state.number_of_boats) for wid in range(state.number_of_whales)}
        for aid, nbid in enumerate(assinged_nbids):
            nwid = assigned_nwids[aid]
            assignment_bid_wid[nbid_to_bid[nbid]] = nwid_to_wid[nwid]
            cost_bid_wid[(nbid_to_bid[nbid], nwid_to_wid[nwid])] = cost[nbid, nwid]
        
        return assignment_bid_wid, cost_bid_wid, cost


    def get_control(self, state: Belief_State):
        


        final_assignment, _, _ = self.get_control_assignment(state)
        bthetas = []
        bvs = []
        for bid in range(state.number_of_boats):
            if bid in final_assignment.keys():
                wid = final_assignment[bid]
                u = self.get_MPC_control(state, bid, wid)
                bthetas.append(float(u[0]))
                bvs.append(float(u[1]))
            else:
                bthetas.append(state.b_theta[bid])
                bvs.append(0)
        
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
