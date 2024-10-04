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
from scipy.optimize import linear_sum_assignment

class BLE2(Policy_base): #distance based greedy assignment
    def __init__(self, knowledge: Global_knowledge) -> None:
        self.knowledge = knowledge
        self.setup_MPC_for_last_mile(knowledge, 60 / self.knowledge.observations_per_minute)
        
        
        self.policy_name = 'BLE2'
        
        
        

    def get_auction(self, state: Belief_State):
        nbid_to_bid = [bid for bid in range(state.number_of_boats)]
        nwid_to_wid = [wid for wid in range(state.number_of_whales) if wid not in state.assigned_whales]
        len_nwid_to_wid = len(nwid_to_wid)
        len_nbid_to_bid = len(nbid_to_bid)
        cost = np.zeros((len_nbid_to_bid, len_nwid_to_wid))
        for nbid in range(len_nbid_to_bid):
            for nwid in range(len_nwid_to_wid):
                wid = nwid_to_wid[nwid]
                bid = nbid_to_bid[nbid]
                cost[nbid, nwid] = np.linalg.norm(np.array([state.w_x[wid], state.w_y[wid]]) - np.array([state.b_x[bid], state.b_y[bid]]))
        assinged_nbids, assigned_nwids = linear_sum_assignment(cost)
        assignment_bid_wid = {}
        for aid, nbid in enumerate(assinged_nbids):
            nwid = assigned_nwids[aid]
            assignment_bid_wid[nbid_to_bid[nbid]] = nwid_to_wid[nwid]

        return assignment_bid_wid
    
    def get_control(self, state: Belief_State):

        if state.time % self.knowledge.observations_per_minute == 0:
            self.bid_whale_assignment = {}
            final_assignment = self.get_auction(state)
            for bid in final_assignment.keys():
                self.bid_whale_assignment[bid] = final_assignment[bid]

        bthetas = [state.b_theta[bid] for bid in range(state.number_of_boats)]
        bvs = [0 for _ in range(state.number_of_boats)]
        for bid in range(state.number_of_boats):
            if bid in self.bid_whale_assignment.keys():
                wid = self.bid_whale_assignment[bid]
                u = self.get_MPC_control(state, bid, wid)
                u0 = np.mod(u[0], 2 * np.pi)
                bthetas[bid] = float(u0)
                bvs[bid] = float(u[1])
            
        # print(state.time, bthetas, bvs)
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))


      
