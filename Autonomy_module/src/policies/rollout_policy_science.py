from src.configs.constants import Boat_Control
from src.belief_state import Belief_State, Policy_base
from src.global_knowledge import Global_knowledge
import numpy as np
from ortools.linear_solver import pywraplp    
import sys, copy
import typing as t
from src.policies.ia_with_commitment_science import IA_with_commitment_science
from scipy.optimize import linear_sum_assignment
from scipy.stats import truncnorm 


class Rollout_Bel_class():
    def __init__(self, time: int = 0, number_of_boats: int = 1, b_x: np.ndarray = None, b_y: np.ndarray = None, \
        number_of_whales: int = 2, w_x: np.ndarray = None, w_y: np.ndarray = None, w_theta: np.ndarray = None, w_v: np.ndarray = None, \
            assigned_whales = [], up_scenario = {}, boat_max_speed: float = 900, tagging_distance: float = 500) -> None:
        self.boat_max_speed = boat_max_speed
        self.tagging_distance = tagging_distance
        self.time = time
        self.number_of_boats = number_of_boats
        self.b_x = b_x
        self.b_y = b_y
        self.number_of_whales = number_of_whales
        self.w_x = w_x
        self.w_y = w_y
        self.w_theta = w_theta
        self.w_v = w_v
        self.assigned_whales = assigned_whales 
        self.up_scenario = up_scenario
        self.up = {}
        for wid in range(self.number_of_whales):
            self.up[wid] = any([scene[0] <= self.time and self.time >= scene[1] for scene in up_scenario[wid]])

    def get_next_state(self, bid_wid_assignement):
        initial_num_assigned = len(self.assigned_whales)
        init_time = self.time
        while len(self.assigned_whales) == initial_num_assigned:
            self.time += 1
            if self.time - init_time > 180: #TODO: not correct
                break
            for abid in bid_wid_assignement:
                awid = bid_wid_assignement[abid]
                da = [ self.w_x[awid] - self.b_x[abid], self.w_y[awid] - self.b_y[abid] ]
                d0 = np.linalg.norm(da)
                bv = min(self.boat_max_speed, d0)
                bt = np.arctan2(self.w_y[awid] - self.b_y[abid], self.w_x[awid] - self.b_x[abid])
                self.b_x[abid] += bv * np.cos(bt)
                self.b_y[abid] += bv * np.sin(bt)

            self.w_x += self.w_v * np.cos(self.w_theta)
            self.w_y += self.w_v * np.sin(self.w_theta)
            for wid in range(self.number_of_whales):
                self.up[wid] = any([scene[0] <= self.time and self.time <= scene[1] for scene in self.up_scenario[wid]])
                if wid in self.assigned_whales:
                    continue
                dist = min([np.linalg.norm([ self.w_x[wid] - self.b_x[bid], self.w_y[wid] - self.b_y[bid] ]) \
                    for bid in range(self.number_of_boats)])
                if self.up[wid] and dist <= self.tagging_distance:
                    self.assigned_whales.append(wid)
        return self.time - init_time

    def time_to_reach(self, awid, abid):
        return np.linalg.norm([ self.w_x[awid] - self.b_x[abid], self.w_y[awid] - self.b_y[abid] ]) / self.boat_max_speed

    def time_to_reach_old(self, awid, abid):
        init_time = self.time
        time = init_time
        if awid in self.assigned_whales:
            return 0
        assigned_whales = []
        w_x = self.w_x[awid]
        w_y = self.w_y[awid]
        b_x = self.b_x[abid]
        b_y = self.b_y[abid]
        up = copy.deepcopy(self.up[awid])
        while awid not in assigned_whales:
            time += 1
            
            bv = min(self.boat_max_speed, \
                np.linalg.norm([ w_x - b_x, w_y - b_y ]))
            bt = np.arctan2(w_y - b_y, w_x - b_x)
            b_x += bv * np.cos(bt)
            b_y += bv * np.sin(bt)

            w_x += self.w_v[awid] * np.cos(self.w_theta[awid])
            w_y += self.w_v[awid] * np.sin(self.w_theta[awid])
            
            up = any([scene[0] <= time and time <= scene[1] for scene in self.up_scenario[awid]])
        
            dist = np.linalg.norm([ w_x - b_x, w_y - b_y ])
            if up and dist <= self.tagging_distance:
                break
        return time - init_time


class MA_Rollout_science(Policy_base):
    
    def __init__(self, knowledge: Global_knowledge, state: Belief_State, \
        CE = False, commitment = False, direction = False, rollout_time_dist = False) -> None:
        self.knowledge = knowledge
        self.commitment = commitment
        self.rollout_time_dist = rollout_time_dist
        self.div = self.knowledge.observations_per_minute
        self.base_policy = IA_with_commitment_science(knowledge)
        if CE:
            self.number_of_scenarios = 1
        else:
            self.number_of_scenarios = 50
        self.future_scenarios, p = self.generate_future_scenarios(state, [wid for wid in range(state.number_of_whales)])
        if self.commitment == False:
            self.policy_name = 'MA_rollout' + ('_CE' if CE else '')
        else:
            self.policy_name = 'MA_rollout' + ('_CE' if CE else '') + ('_comm' if self.commitment else '')

        self.direction = direction
        if direction:
            self.policy_name = 'MA_rollout_direction'
        if self.rollout_time_dist:
            self.policy_name = 'MA_rollout_time_dist'
        self.setup_MPC_for_last_mile(knowledge, 60 / self.knowledge.observations_per_minute)

    def generate_future_scenarios(self, state: Belief_State, current_wids: t.List[int]):
        future_scenarios = []
        p = [] #TODO use prob for these
        if self.number_of_scenarios == 1:
            future_scenario = {wid:[] for wid in range(state.number_of_whales)}
            for wid in range(state.number_of_whales):
                if state.whale_up2[wid] == True:
                    interval_end_time = int(state.w_last_surface_start_time[wid]/self.div + state.knowledge.surface_time_mean/self.div)
                    future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                else:
                    interval_end_time = int(state.w_last_surface_end_time[wid]/self.div)
                    future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                while interval_end_time < int(state.knowledge.n_horizon_for_evaluation/self.div):
                    interval_start_time = int(interval_end_time + state.knowledge.down_time_mean/self.div)
                    interval_end_time = int(interval_start_time + state.knowledge.surface_time_mean/self.div)
                    future_scenario[wid].append((interval_start_time, interval_end_time))
            
            future_scenarios.append(future_scenario)
        else:
            number_of_intervals = int(10 * np.ceil(state.knowledge.n_horizon_for_evaluation / (state.knowledge.down_time_mean + state.knowledge.surface_time_mean)))
            # print(number_of_intervals)
            interval_up_times = np.maximum(state.knowledge.surface_time_var/self.div, \
                np.random.normal(state.knowledge.surface_time_mean/self.div, state.knowledge.surface_time_var/self.div, \
                    size = state.number_of_whales * self.number_of_scenarios * number_of_intervals ))\
                        .reshape(self.number_of_scenarios, state.number_of_whales, number_of_intervals).astype(int)
            interval_down_times = np.maximum(state.knowledge.down_time_var/self.div, \
                np.random.normal(state.knowledge.down_time_mean/self.div, state.knowledge.down_time_var/self.div, \
                    size = state.number_of_whales * self.number_of_scenarios * number_of_intervals ))\
                        .reshape(self.number_of_scenarios, state.number_of_whales, number_of_intervals).astype(int)
            for scene_id in range(self.number_of_scenarios):
                future_scenario = {wid:[] for wid in range(state.number_of_whales)}
                for wid in range(state.number_of_whales):
                    if state.whale_up2[wid] == True:
                        # # if we do not want full stochasticity for the future whales
                        # if wid in current_wids: 
                        #     interval_end_time = state.w_last_surface_start_time[wid] + interval_up_times[scene_id][wid][0]
                        # else:
                        #     interval_end_time = state.w_last_surface_start_time[wid] + state.knowledge.surface_time_mean
                        interval_end_time = int(state.w_last_surface_start_time[wid]/self.div) + interval_up_times[scene_id][wid][0]
                        future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                    else:
                        interval_end_time = int(state.w_last_surface_end_time[wid]/self.div)
                        future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
                    # int_id = 1
                    # while interval_end_time < state.knowledge.n_horizon_for_evaluation:
                    for int_id in range(1, number_of_intervals):
                        # # if we do not want full stochasticity for the future whales
                        # if wid in current_wids: 
                        #     interval_start_time = interval_end_time + interval_down_times[scene_id][wid][int_id]
                        #     interval_end_time = interval_start_time + interval_up_times[scene_id][wid][int_id]
                        # else:
                        #     interval_start_time = interval_end_time + state.knowledge.down_time_mean
                        #     interval_end_time = interval_start_time + state.knowledge.surface_time_mean
                        interval_start_time = interval_end_time + interval_down_times[scene_id][wid][int_id]
                        interval_end_time = interval_start_time + interval_up_times[scene_id][wid][int_id]
                        future_scenario[wid].append((interval_start_time, interval_end_time))
                        # int_id += 1
                future_scenarios.append(future_scenario)
        return future_scenarios, p

    def update_future_old(self, state : Belief_State, wids_to_update: t.List[int], scene_id: int):
        future_scenario = {wid:[] if wid in wids_to_update else self.future_scenarios[scene_id][wid] for wid in range(state.number_of_whales)} 
        p = []
        for wid in wids_to_update:
            number_of_intervals = int(10 * np.ceil(state.knowledge.n_horizon_for_evaluation / (state.knowledge.down_time_mean + state.knowledge.surface_time_mean)))
            
            if self.number_of_scenarios == 1:
                interval_up_times = [int(state.knowledge.surface_time_mean/self.div)]* number_of_intervals
                interval_down_times = [int(state.knowledge.down_time_mean/self.div)]* number_of_intervals
            else:
                interval_up_times = np.maximum(state.knowledge.surface_time_var/self.div, \
                    np.random.normal(state.knowledge.surface_time_mean/self.div, state.knowledge.surface_time_var/self.div, \
                        size = number_of_intervals )).astype(int)
            
                interval_down_times = np.maximum(state.knowledge.down_time_var/self.div, \
                    np.random.normal(state.knowledge.down_time_mean/self.div, state.knowledge.down_time_var/self.div, \
                        size = number_of_intervals )).astype(int)

            if state.whale_up2[wid] == True:
                interval_end_time = max(0, int(state.w_last_surface_start_time[wid]/self.div) + interval_up_times[0])
                future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
            else:
                interval_end_time = int(state.w_last_surface_end_time[wid]/self.div)
                future_scenario[wid] = [(int(state.w_last_surface_start_time[wid]/self.div), interval_end_time)]
            int_id = 1
            # while interval_end_time < state.knowledge.n_horizon_for_evaluation:
            for int_id in range(1, number_of_intervals):
                interval_start_time = interval_end_time + interval_down_times[int_id]
                interval_end_time = interval_start_time + interval_up_times[int_id]
                future_scenario[wid].append((interval_start_time, interval_end_time))
                # int_id += 1

        return future_scenario, p

    def update_future(self, state : Belief_State, wids_to_update: t.List[int], scene_id: int):
        future_scenario = {wid:[] if wid in wids_to_update else self.future_scenarios[scene_id][wid] for wid in range(state.number_of_whales)} 
        p = []
        for wid in wids_to_update:
            number_of_intervals = int(10 * np.ceil(state.knowledge.n_horizon_for_evaluation / (state.knowledge.down_time_mean + state.knowledge.surface_time_mean)))
            
            if self.number_of_scenarios == 1:
                interval_up_times = [int(state.knowledge.surface_time_mean/self.div)]* number_of_intervals
                interval_down_times = [int(state.knowledge.down_time_mean/self.div)]* number_of_intervals
            else:
                interval_up_times = np.maximum(state.knowledge.surface_time_var/self.div, \
                    np.random.normal(state.knowledge.surface_time_mean/self.div, state.knowledge.surface_time_var/self.div, \
                        size = number_of_intervals )).astype(int)
            
                interval_down_times = np.maximum(state.knowledge.down_time_var/self.div, \
                    np.random.normal(state.knowledge.down_time_mean/self.div, state.knowledge.down_time_var/self.div, \
                        size = number_of_intervals )).astype(int)

            if state.whale_up2[wid] == True:
                interval_start_time = state.w_last_surface_start_time[wid]/self.div
                mu = state.knowledge.surface_time_mean/self.div
                sigma = state.knowledge.surface_time_var/self.div
                lower = state.time/self.div - interval_start_time
                upper = lower + state.knowledge.surface_time_mean/self.div
                interval_duration = truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1)[0]
                interval_end_time = int(interval_start_time + interval_duration)
                interval_start_time = int(interval_start_time)
                future_scenario[wid] = [(interval_start_time, interval_end_time)]
                
            else: # underwater
                mu = state.knowledge.down_time_mean/self.div
                sigma = state.knowledge.down_time_var/self.div
                lower = state.time/self.div - state.w_last_surface_end_time[wid]/self.div
                upper = lower + state.knowledge.down_time_mean/self.div # 2 means
                interval_duration = truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=1)[0]
                interval_start_time = state.w_last_surface_end_time[wid]/self.div + interval_duration
                interval_end_time = interval_start_time + interval_down_times[0]
                interval_start_time = int(interval_start_time)
                interval_end_time = int(interval_end_time)
                future_scenario[wid] = [(interval_start_time, interval_end_time)]
            int_id = 1
            # while interval_end_time < state.knowledge.n_horizon_for_evaluation:
            for int_id in range(1, number_of_intervals):
                interval_start_time = interval_end_time + interval_down_times[int_id]
                interval_end_time = interval_start_time + interval_up_times[int_id]
                future_scenario[wid].append((interval_start_time, interval_end_time))
                # int_id += 1

        return future_scenario, p

    
    def MC_Cost_with_future_scenarios(self, state_orig: Belief_State, current_assignment_):
        
        total_cost = 0
        distace_traveled = 0
        for scene_id in range(self.number_of_scenarios):
            time_taken_scene = 0
            current_assignment = copy.deepcopy(current_assignment_)
            # state_scene = copy.deepcopy(state_orig)
            # state_scene.surface_interval_scenario = self.future_scenarios[scene_id]
            
            surface_start_times = {wid: [t[0] for t in self.future_scenarios[scene_id][wid]] for wid in range(state_orig.number_of_whales)}
            surface_end_times = {wid: [t[1] for t in self.future_scenarios[scene_id][wid]] for wid in range(state_orig.number_of_whales)}
            
            wids_to_update = []
            for wid in range(state_orig.number_of_whales):
                # if (state_orig.time == state_orig.w_last_surface_start_time[wid] and int(state_orig.time/self.div) not in surface_start_times[wid]) \
                #     or (state_orig.time == state_orig.w_last_surface_end_time[wid] and int(state_orig.time/self.div) not in surface_end_times[wid]) :
                #     wids_to_update.append(wid)
                if (state_orig.whale_up2[wid] and \
                    not any([ss_se[0]<=int(state_orig.time/self.div) and ss_se[1]>=int(state_orig.time/self.div) for ss_se in self.future_scenarios[scene_id][wid]])) \
                        or (state_orig.whale_up2[wid] == 0 and \
                            any([ss_se[0]<=(state_orig.time/self.div) and ss_se[1]>=(state_orig.time/self.div) for ss_se in self.future_scenarios[scene_id][wid]])):
                    wids_to_update.append(wid)
            
            if len(wids_to_update) >= 1:
                self.future_scenarios[scene_id], p = self.update_future(state_orig, wids_to_update, scene_id)
                # state_scene.surface_interval_scenario = copy.deepcopy(self.future_scenarios[scene_id])
                # print('updated scenario: ', scene_id, state_orig.time, wids_to_update)
            
            state_scene : Rollout_Bel_class = Rollout_Bel_class(int(state_orig.time/self.div), state_orig.number_of_boats, np.copy(state_orig.b_x), np.copy(state_orig.b_y), \
                state_orig.number_of_whales, np.copy(state_orig.w_x), np.copy(state_orig.w_y), \
                    np.copy(state_orig.w_theta), np.copy(state_orig.w_v) * self.div, copy.copy(state_orig.assigned_whales), \
                        up_scenario = self.future_scenarios[scene_id], \
                            boat_max_speed = self.knowledge.boat_max_speed_mtpm, tagging_distance = self.knowledge.tagging_distance)
            prev_loc = ([state_scene.b_x[bid] for bid in range(state_scene.number_of_boats)], \
                [state_scene.b_y[bid] for bid in range(state_scene.number_of_boats)])
            while len(state_scene.assigned_whales) < state_scene.number_of_whales:
                num_unassigned = state_scene.number_of_whales - len(state_scene.assigned_whales)
                stage_cost = state_scene.get_next_state(current_assignment)
                distace_traveled += sum([np.linalg.norm([prev_loc[0][bid] - state_scene.b_x[bid], prev_loc[1][bid] - state_scene.b_y[bid]])/self.knowledge.boat_max_speed_mtpm for bid in range(state_scene.number_of_boats)])
                prev_loc = ([state_scene.b_x[bid] for bid in range(state_scene.number_of_boats)], \
                    [state_scene.b_y[bid] for bid in range(state_scene.number_of_boats)])
                time_taken_scene += stage_cost
                if self.rollout_time_dist:
                    total_cost += stage_cost * num_unassigned + distace_traveled
                else:
                    total_cost += stage_cost * num_unassigned
                current_assignment = self.get_control_assignment(state_scene) # TODO: Do I need to keep the assignment for the first step where bid not in bid_to_finish_first
                if time_taken_scene > int(self.knowledge.n_horizon_for_evaluation/self.div):
                    break

        return total_cost/ self.number_of_scenarios

    def get_control_assignment(self, state: Rollout_Bel_class, bids_to_exclude = []):
        nbid_to_bid = [bid for bid in range(state.number_of_boats) if bid not in bids_to_exclude]
        nwid_to_wid = [wid for wid in range(state.number_of_whales) if wid not in state.assigned_whales]
        len_nwid_to_wid = len(nwid_to_wid)
        len_nbid_to_bid = len(nbid_to_bid)
        cost = np.zeros((len_nbid_to_bid, len_nwid_to_wid))

        for nbid in range(len_nbid_to_bid):
            for nwid in range(len_nwid_to_wid):
                wid = nwid_to_wid[nwid]
                bid = nbid_to_bid[nbid]
                # TODO: Why does not it consider the surface scenario?
                time = state.time_to_reach(wid, bid)
                
                cost[nbid, nwid] = time
                
        # assignments = self.base_policy.mip_solver_2(cost)
        assinged_nbids, assigned_nwids = linear_sum_assignment(cost)
        
        assignment_bid_wid = {}
        # for (nbid, nwid) in assignments:
        for aid, nbid in enumerate(assinged_nbids):
            nwid = assigned_nwids[aid]
            assignment_bid_wid[nbid_to_bid[nbid]] = nwid_to_wid[nwid]

        return assignment_bid_wid

    def get_auction(self, state: Belief_State, scene = False, bids_to_exclude = []):
        nbid_to_bid = [bid for bid in range(state.number_of_boats) if bid not in bids_to_exclude]
        nwid_to_wid = [wid for wid in range(state.number_of_whales) if wid not in state.assigned_whales]
        len_nwid_to_wid = len(nwid_to_wid)
        len_nbid_to_bid = len(nbid_to_bid)
        cost = np.zeros((len_nbid_to_bid, len_nwid_to_wid))
        
        for nbid in range(len_nbid_to_bid):
            for nwid in range(len_nwid_to_wid):
                wid = nwid_to_wid[nwid]
                bid = nbid_to_bid[nbid]
                st = state.state_copy(bid, wid)
                # self.future_scenarios[scene_id][wid]
                for scene_id in range(self.number_of_scenarios):
                    st.surface_interval_scenario = [self.future_scenarios[scene_id][wid]]
                    ia_output = self.base_policy.get_cost_of_single_assignment_for_a_scene(st)
                    if self.rollout_time_dist:
                        cost[nbid, nwid] += ia_output[0] + ia_output[3]
                    else:
                        cost[nbid, nwid] += ia_output[0]
                cost[nbid, nwid] /=self.number_of_scenarios
                 
        assinged_nbids, assigned_nwids = linear_sum_assignment(cost)
        assignment_bid_wid = {}
        cost_bid_wid = {(bid, wid): sys.maxsize for bid in range(state.number_of_boats) for wid in range(state.number_of_whales)}
        for aid, nbid in enumerate(assinged_nbids):
            nwid = assigned_nwids[aid]
            assignment_bid_wid[nbid_to_bid[nbid]] = nwid_to_wid[nwid]
            cost_bid_wid[(nbid_to_bid[nbid], nwid_to_wid[nwid])] = cost[nbid, nwid]
        
        return assignment_bid_wid, cost_bid_wid, cost
    
    def get_control(self, state: Belief_State):
        if state.number_of_whales > 10:
            # todo: if number of agents greater than 10 spawn parallel threads
            raise NotImplemented
        final_btheta_bv = {}
        bids_to_exclude = []
        if state.time % self.knowledge.observations_per_minute == 0:
            # TODO: USE Latest AOA if we do not see a whale at place? visual AOA from the agent itself ?

            self.bid_whale_assignment = {}
            if self.direction:
                if hasattr(state, 'agent_loc_aoa'):
                    for bid in state.agent_loc_aoa:
                        bids_to_exclude.append(bid)
                        final_btheta_bv[bid] = (state.agent_loc_aoa[bid][0], state.knowledge.tagging_distance)

            # if self.knowledge.overlay_GPS:
            final_assignment, _, costs_all_bp = self.get_auction(state, bids_to_exclude = bids_to_exclude)
            # else:
            #     final_assignment, costs_bp, costs_all_bp = self.base_policy.get_control_assignment(state, bids_to_exclude = bids_to_exclude)

            for bid in bids_to_exclude:
                final_assignment[bid] = state.agent_loc_aoa[bid][1]
            # print('time: ', state.time, ', base_assignment: ', final_assignment, ', costs_all: ', costs_all_bp)
            if state.number_of_boats < (state.number_of_whales - len(state.assigned_whales)):

                for bid in range(state.number_of_boats):
                    if bid in bids_to_exclude:
                        continue
                    C = {}
                    for wid in range(state.number_of_whales):
                        if wid in state.assigned_whales or wid in [final_assignment[bid_] for bid_ in range(bid)]:
                            continue
                        final_assignment[bid] = wid
                         
                        new_state = copy.deepcopy(state)
                        C[wid] = self.MC_Cost_with_future_scenarios(new_state, final_assignment)
                
                    wid_to_assign = min(C, key=C.get)
                    # print('time: ', state.time, ', bid: ', bid, ', C:', C, ', wid_to_assign:', wid_to_assign)
                    # wid_to_assign = np.argmin([C[wid] for wid in C.keys()]) 
                    final_assignment[bid] = wid_to_assign
                    self.bid_whale_assignment[bid] = wid_to_assign
            else:
                for bid in range(state.number_of_boats):
                    if bid in final_assignment.keys():
                        self.bid_whale_assignment[bid] = final_assignment[bid]
          

        bthetas = [state.b_theta[bid] for bid in range(state.number_of_boats)]
        bvs = [0 for _ in range(state.number_of_boats)]
        for bid in range(state.number_of_boats):
            if bid in final_btheta_bv.keys():
                bthetas[bid] = final_btheta_bv[bid][0]
                bvs[bid] = final_btheta_bv[bid][1]
            elif bid in self.bid_whale_assignment.keys():
                wid = self.bid_whale_assignment[bid]
                u = self.get_MPC_control(state, bid, wid)
                # arctan_u = np.arctan2(state.w_y[wid] - state.b_y[bid], state.w_x[wid] - state.b_x[bid])
                u0 = np.mod(u[0], 2 * np.pi)
                bthetas[bid] = float(u0)
                bvs[bid] = float(u[1])
            
        # print(state.time, bthetas, bvs)
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
        
        