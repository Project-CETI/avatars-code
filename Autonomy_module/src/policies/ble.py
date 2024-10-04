import numpy as np
import copy
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from geopy import Point, distance                                                                                                                                                                    
from geopy.distance import geodesic 
from geographiclib.geodesic import Geodesic
from src.global_knowledge import Global_knowledge
from src.configs.constants import Boat_Control
from src.belief_state import Belief_State, Policy_base

class BLE(Policy_base): #distance based greedy assignment
    def __init__(self, knowledge: Global_knowledge, state: Belief_State) -> None:
        self.knowledge = knowledge
        self.setup_MPC_for_last_mile(knowledge, 60 / self.knowledge.observations_per_minute)
        
        
        self.policy_name = 'BLE'
        # self.number_of_scenarios = 1
        
        self.fixed_assignment = self.get_control_assignment(state, np.arange(state.number_of_boats), np.arange(state.number_of_whales))
    
    def get_cost_of_single_assignment_for_a_scene(self, state: Belief_State, surface_interval_scenario):
    
        
        wxdot_np = np.array([0] + [state.w_v[0] * np.cos(state.w_theta[0])] * (self.knowledge.n_horizon_for_evaluation))
        wydot_np = np.array([0] + [state.w_v[0] * np.sin(state.w_theta[0])] * (self.knowledge.n_horizon_for_evaluation))
        wx_np = np.cumsum(wxdot_np + state.w_x[0] * np.eye(self.knowledge.n_horizon_for_evaluation + 1)[0])
        wy_np = np.cumsum(wydot_np + state.w_y[0] * np.eye(self.knowledge.n_horizon_for_evaluation + 1)[0])

        bx_np = state.b_x[0] + np.eye(self.knowledge.n_horizon_for_evaluation + 1)[0]
        by_np = state.b_y[0] + np.eye(self.knowledge.n_horizon_for_evaluation + 1)[0]

        whale_ups = np.zeros(self.knowledge.n_horizon_for_evaluation + 1)
        for interval in surface_interval_scenario[0]:
            if interval[1] >= state.time and interval[0] <= state.time + self.knowledge.n_horizon_for_evaluation:
                whale_ups[ max(0, interval[0] - state.time ) : min(interval[1] - state.time + 1, self.knowledge.n_horizon_for_evaluation) ] = 1

        distances = np.zeros(self.knowledge.n_horizon_for_evaluation + 1)
        assigned_index = -1
        for n in range(1, self.knowledge.n_horizon_for_evaluation + 1):
            
            btheta = np.arctan2(wy_np[n-1] - by_np[n-1], wx_np[n-1] - bx_np[n-1])
            distances[n-1] = np.sqrt((wy_np[n-1] - by_np[n-1])**2 + (wx_np[n-1] - bx_np[n-1])**2)
            bv = min(self.knowledge.boat_max_speed_mtpm, distances[n-1])
            bx_np[n] = bx_np[n-1] + bv * np.cos(btheta)
            by_np[n] = by_np[n-1] + bv * np.sin(btheta) 
        
            if n == 1:
                btheta0 = btheta
                bv0 = bv
            if assigned_index == - 1 and whale_ups[n - 1] and distances[n-1] < state.knowledge.tagging_distance:
                assigned_index = n - 1
                break
            

        u0_final2 = np.array([btheta0, bv0])
        
        assigned = np.zeros(self.knowledge.n_horizon_for_evaluation + 1).astype(int)
        assigned[assigned_index:] = 1
        
        avg_cost = self.knowledge.n_horizon_for_evaluation + 1 - np.sum(assigned) #np.sum( [ 1 for 

        return avg_cost, u0_final2
    
    def get_control_assignment(self, state: Belief_State, bids_to_assign, wids_to_assign):

        nwid_to_wid = wids_to_assign # [wid for wid in range(state.number_of_whales) if wid not in state.assigned_whales]
        len_nwid_to_wid = len(wids_to_assign)
        number_of_boats = len(bids_to_assign)

        assignments = {}        
        for nbid in range(number_of_boats):
            bid = bids_to_assign[nbid]
            cost = np.zeros(len_nwid_to_wid)
            for nwid in range(len_nwid_to_wid):
                wid = nwid_to_wid[nwid]
                st : Belief_State = state.state_copy(bid, wid)
                future_scenario = {0:[]}
                if state.whale_up2[wid] == True:
                    interval_end_time = int(state.w_last_surface_start_time[wid] + state.knowledge.surface_time_mean)
                    future_scenario[0] = [(state.w_last_surface_start_time[wid], interval_end_time)]
                else:
                    interval_end_time = state.w_last_surface_end_time[wid]
                    future_scenario[0] = [(state.w_last_surface_start_time[wid], interval_end_time)]
                while interval_end_time < state.knowledge.n_horizon_for_evaluation:
                    interval_start_time = int(interval_end_time + state.knowledge.down_time_mean)
                    interval_end_time = int(interval_start_time + state.knowledge.surface_time_mean)
                    future_scenario[0].append((interval_start_time, interval_end_time))
                output = self.get_cost_of_single_assignment_for_a_scene(st, future_scenario)    

                cost[nwid] = output[0]

                # cost[nwid] = np.linalg.norm(np.array([state.w_x[wid], state.w_y[wid]]) - np.array([state.b_x[bid], state.b_y[bid]]))
            
            nwid_min = np.argmin(cost) 
            assignments[bid] = nwid_to_wid[nwid_min]

        return assignments

    # def get_control(self):
    #     if state.time % self.knowledge.observations_per_minute == 0:
    #         self.bid_whale_assignment = {}
    #         final_assignment = self.get_auction(state)
    #         for bid in final_assignment.keys():
    #             self.bid_whale_assignment[bid] = final_assignment[bid]

    #     bthetas = [state.b_theta[bid] for bid in range(state.number_of_boats)]
    #     bvs = [0 for _ in range(state.number_of_boats)]
    #     for bid in range(state.number_of_boats):
    #         if bid in self.bid_whale_assignment.keys():
    #             wid = self.bid_whale_assignment[bid]
    #             u = self.get_MPC_control(state, bid, wid)
    #             u0 = np.mod(u[0], 2 * np.pi)
    #             bthetas[bid] = float(u0)
    #             bvs[bid] = float(u[1])
            
    #     # print(state.time, bthetas, bvs)
    #     return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))

    def get_control(self, state: Belief_State):
        if state.time % self.knowledge.observations_per_minute == 0:
            bids_in_assigned = list(self.fixed_assignment.keys())
            for bid_in_assigned in bids_in_assigned:
                if self.fixed_assignment[bid_in_assigned] in state.assigned_whales:
                    del self.fixed_assignment[bid_in_assigned] 

            bids_to_assign = []
            wids_to_assign = []
            for bid in range(state.number_of_boats):
                if bid not in self.fixed_assignment.keys():
                    bids_to_assign.append(bid)
            for wid in range(state.number_of_whales):
                if wid not in state.assigned_whales and wid not in self.fixed_assignment.values():
                    wids_to_assign.append(wid)

            if len(bids_to_assign) > 0 and len(wids_to_assign) > 0:
                final_assignment = self.get_control_assignment(state, bids_to_assign, wids_to_assign)
                for new_bid in final_assignment.keys():
                    self.fixed_assignment[new_bid] = final_assignment[new_bid]
        

        bthetas = [state.b_theta[bid] for bid in range(state.number_of_boats)]
        bvs = [0 for _ in range(state.number_of_boats)]
        for bid in range(state.number_of_boats):
            if bid in self.fixed_assignment.keys():
                wid = self.fixed_assignment[bid]
                u = self.get_MPC_control(state, bid, wid)
                u0 = np.mod(u[0], 2 * np.pi)
                bthetas[bid] = float(u0)
                bvs[bid] = float(u[1])
            
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
    
        # bthetas = []
        # bvs = []
        # for bid in range(state.number_of_boats):
        #     if bid in self.fixed_assignment.keys():
        #         wid = self.fixed_assignment[bid]
        #         u = self.get_MPC_control(state, bid, wid)
        #         bthetas.append(float(u[0]))
        #         bvs.append(float(u[1]))
        #     else:
        #         bthetas.append(state.b_theta[bid])
        #         bvs.append(0)
        # # print(bthetas, bvs)
        # return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
