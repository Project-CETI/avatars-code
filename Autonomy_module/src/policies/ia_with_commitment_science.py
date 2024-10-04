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
        #self.n_horizon = 120
        # self.solver_base = pywraplp.Solver.CreateSolver("SCIP")
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
    
        # if self.knowledge.use_GPS_coordinate_from_dataset:
        #     wx_np = np.zeros(self.knowledge.n_horizon_for_evaluation)
        #     wy_np = np.zeros(self.knowledge.n_horizon_for_evaluation)
        #     wx_np[0] = state.w_x[0]
        #     wy_np[0] = state.w_y[0]
        #     for t in range(1, self.knowledge.n_horizon_for_evaluation):
        #         wx_np[t], wy_np[t] = self.knowledge.get_gps_from_start_vel_bearing(wx_np[t - 1], wy_np[t - 1], \
        #             state.w_v[0], state.w_theta[0])
        # else:
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
            # if self.knowledge.use_GPS_coordinate_from_dataset:
            #     distances[n - 1] = self.knowledge.get_distance_from_latLon_to_meter(by_np[n-1], bx_np[n-1], \
            #         wy_np[n-1], wx_np[n-1])
            #     btheta = self.knowledge.get_bearing_from_p1_p2(bx_np[n - 1], by_np[n - 1], wx_np[n - 1], wy_np[n - 1])
            #     bv = min(self.knowledge.boat_max_speed_mtpm, distances[n-1])                
            #     bx_np[n], by_np[n] = self.knowledge.get_gps_from_start_vel_bearing(bx_np[n - 1], by_np[n - 1], bv, btheta)
            # else:
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
            
        # distances[n-1] = np.sqrt((wy_np[n-1] - by_np[n-1])**2 + (wx_np[n-1] - bx_np[n-1])**2)
        # if assigned_index == - 1 and whale_ups[n - 1] and distances[n-1] < state.knowledge.tagging_distance:
        #     assigned_index = n - 1

        u0_final2 = np.array([btheta0, bv0])
        # if self.knowledge.use_GPS_coordinate_from_dataset:
        #     u0_final2 = np.array([btheta0, bv])
        # else:
        #     u0_final2 = np.array([ np.arctan2(wy_np[0] - by_np[0], wx_np[0] - bx_np[0]), \
        #         min(state.knowledge.boat_max_speed_mtpm, np.sqrt( (wy_np[0]- by_np[0])**2 + (wx_np[0]- bx_np[0])**2 )) ])


        assigned = np.zeros(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1).astype(int)
        assigned[assigned_index:] = 1
        # for n_ in range(1, self.knowledge.n_horizon + 1):
        #     assigned[n_] = np.minimum(1, assigned[n_ -1] + (1 - assigned[n_ -1]) * whale_ups[n_] * (distances[n_] <= self.knowledge.tagging_distance))

        avg_cost = int(self.knowledge.n_horizon_for_evaluation/self.div) + 1 - np.sum(assigned) #np.sum( [ 1 for st_id in range(self.knowledge.n_horizon + 1) if not assigned[st_id]  ] )

        # sum_d = sum(distances)
        #avg_cost_distance_based = np.sum( [ distances[st_id]/state.knowledge.boat_max_speed_mtpm for st_id in range(self.knowledge.n_horizon + 1) if not assigned[st_id]  ] )
        avg_cost_distance_based = np.sum(distances[:avg_cost])/state.knowledge.boat_max_speed_mtpm

        # avg_cost += 0.5 * abs((state.b_theta[0] - u0_final2[0]))

        return avg_cost, u0_final2, 'states_greedy', avg_cost_distance_based

    def get_cost_of_single_assignment_for_a_scene_slower(self, state: Belief_State):
    
        states_greedy = [ [state.b_x[0], state.b_y[0], state.w_x[0], state.w_v[0] * np.cos(state.w_theta[0]), \
            state.w_y[0], state.w_v[0] * np.sin(state.w_theta[0])]  ]
        u0_final2 = np.array([state.b_v[0] * np.cos(state.b_theta[0]), state.b_v[0] * np.sin(state.b_theta[0]) ]).reshape(2,1)
        for n in range(1, self.knowledge.n_horizon_for_evaluation + 1):
            bx = states_greedy[n-1][0]
            by = states_greedy[n-1][1]
            wx = states_greedy[n-1][2] # use particle 
            wxdot = states_greedy[n-1][3] # use motion uncertainty 
            wy = states_greedy[n-1][4]
            wydot = states_greedy[n-1][5]

            btheta = np.arctan2(wy - by, wx - bx)
            bv = min(self.knowledge.boat_max_speed_mtpm, np.sqrt((wy - by)**2 + (wx - bx)**2))

            if n==1:
                u0_final2[0] = btheta
                u0_final2[1] = bv

            states_greedy.append([bx + bv * np.cos(btheta), by + bv * np.sin(btheta), wx + wxdot, wxdot, wy + wydot, wydot ])
    
        distances = [np.sqrt((st[0]-st[2])**2 + (st[1]-st[4])**2) for st in states_greedy]

        whale_ups = np.zeros(self.knowledge.n_horizon_for_evaluation + 1)
        for interval in state.surface_interval_scenario[0]:
            if interval[1] >= state.time and interval[0] <= state.time + self.knowledge.n_horizon_for_evaluation:
                whale_ups[ max(0, interval[0] - state.time ) : min(interval[1] - state.time, self.knowledge.n_horizon_for_evaluation) ] = 1
            # if interval[1] >= 0 and interval[0] <= self.knowledge.n_horizon:
            #     whale_ups[ max(0, interval[0]) : min(interval[1], self.knowledge.n_horizon) ] = 1

        assigned = np.array([0]*(self.knowledge.n_horizon_for_evaluation + 1))
        for n_ in range(1, self.knowledge.n_horizon_for_evaluation + 1):
            assigned[n_] = min(1, assigned[n_ -1] + (1 - assigned[n_ -1]) * whale_ups[n_] * (distances[n_] <= self.knowledge.tagging_distance))

        avg_cost = np.sum( [ 1 for st_id in range(len(states_greedy)) if not assigned[st_id]  ] )

        sum_d = sum(distances)
        avg_cost_distance_based = np.sum( [ distances[st_id]/state.knowledge.boat_max_speed_mtpm for st_id in range(len(states_greedy)) if not assigned[st_id]  ] )

        # avg_cost += 0.5 * abs((state.b_theta[0] - u0_final2[0]))

        return avg_cost, u0_final2, states_greedy, avg_cost_distance_based

    def get_cost_of_single_assignment(self, state: Belief_State):
        
        avg_cost = 0

        avg_cost_distance_based = 0
        # if self.knowledge.use_GPS_coordinate_from_dataset:
        #     next_long_lat = self.knowledge.get_gps_from_start_vel_bearing(state.w_x[0], state.w_y[0], \
        #         state.w_v[0], state.w_theta[0])
        #     states_greedy = [ [state.b_x[0], state.b_y[0], state.w_x[0], next_long_lat[0] - state.w_x[0], \
        #         state.w_y[0], next_long_lat[1] - state.w_y[0]]]
            
        # else:
        states_greedy = [ [state.b_x[0], state.b_y[0], state.w_x[0], state.w_v[0] * np.cos(state.w_theta[0]), \
            state.w_y[0], state.w_v[0] * np.sin(state.w_theta[0])]  ]
    
        for n in range(1, int(self.knowledge.n_horizon_for_evaluation /self.div) + 1):
            bx = states_greedy[n-1][0]
            by = states_greedy[n-1][1]
            
            wx = states_greedy[n-1][2] # use particle 
            wxdot = states_greedy[n-1][3] # use motion uncertainty 
            wy = states_greedy[n-1][4]
            wydot = states_greedy[n-1][5]

            # if self.knowledge.use_GPS_coordinate_from_dataset:
            #     btheta = self.knowledge.get_bearing_from_p1_p2(bx, by, wx, wy)
            #     bv = min(self.knowledge.boat_max_speed_mtpm, self.knowledge.get_distance_from_latLon_to_meter(by, bx, wy, wx))
            #     b_next_long_lat = self.knowledge.get_gps_from_start_vel_bearing(bx, by, bv, btheta)
            #     w_next_long_lat = self.knowledge.get_gps_from_start_vel_bearing(wx, wy, state.w_v[0], state.w_theta[0])
            #     states_greedy.append([b_next_long_lat[0], b_next_long_lat[1], \
            #         w_next_long_lat[0], w_next_long_lat[0] - wx, w_next_long_lat[1], w_next_long_lat[1] - wy ])
            # else:
            btheta = np.arctan2(wy - by, wx - bx)
            bv = min(self.knowledge.boat_max_speed_mtpm, np.sqrt((wy - by)**2 + (wx - bx)**2))
            states_greedy.append([bx + bv * np.cos(btheta), by + bv * np.sin(btheta), wx + wxdot, wxdot, wy + wydot, wydot ])

            if n==1:
                u0_final2 = np.array([btheta, bv])

            
    
        # if self.knowledge.use_GPS_coordinate_from_dataset:
        #     distances = [self.knowledge.get_distance_from_latLon_to_meter(st[1], st[0], st[4], st[2]) for st in states_greedy]
        # else:
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
                # try:
                assigned[n_] = min(1, assigned[n_ -1] + (1 - assigned[n_ -1]) * whale_ups[n_] * (distances[n_] <= self.knowledge.tagging_distance))
                # except Exception as e:
                #     print(e)

    
            avg_cost += np.sum( [ 1 for st_id in range(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1) if not assigned[st_id]  ] )

            sum_d = sum(distances)
            # try:
            avg_cost_distance_based += np.sum( [ distances[st_id] for st_id in range(int(self.knowledge.n_horizon_for_evaluation/self.div) + 1) if not assigned[st_id]  ] )
            # except Exception as e:
            #     print(e)

        avg_cost /= self.num_scenarios
        avg_cost_distance_based /= self.num_scenarios

        # avg_cost += 0.5 * abs((state.b_theta[0] - u0_final2[0]))

        
        return avg_cost, u0_final2, states_greedy, avg_cost_distance_based

    def mip_solver_2(self, cost):
        solver = pywraplp.Solver.CreateSolver("SCIP") #self.solver_base
        if not solver:
            return
        infinity = solver.infinity()
        x = {}
        Ni = cost.shape[0]
        Nj = cost.shape[1]
        for i in range(Ni):
            for j in range(Nj):
                x[(i,j)] = solver.IntVar(0, 1, "x[("+str(i)+","+str(j)+")]")

        for i in range(Ni):
            constraint_expr = [x[(i,j)] for j in range(Nj)]
            if Ni <= Nj:
                solver.Add(sum(constraint_expr)==1)
            else:  
                solver.Add(sum(constraint_expr)<=1)
        for j in range(Nj):
            constraint_expr = [x[(i,j)] for i in range(Ni)]
            if Ni <= Nj:
                solver.Add(sum(constraint_expr) <= 1)
            else:
                solver.Add(sum(constraint_expr) == 1)
                
        objective = solver.Objective()
        for i in range(Ni):
            for j in range(Nj):
                objective.SetCoefficient(x[(i,j)], cost[i,j])
        objective.SetMinimization()    
        status = solver.Solve()

        assignment = []
        if status == pywraplp.Solver.OPTIMAL:
            #print("Objective value =", solver.Objective().Value())
            for i in range(Ni):
                for j in range(Nj):
                    if x[(i,j)].solution_value() > 0.5:
                        assignment.append([i,j])

            # print()
            # print("Problem solved in %f milliseconds" % solver.wall_time())
            # print("Problem solved in %d iterations" % solver.iterations())
            # print("Problem solved in %d branch-and-bound nodes" % solver.nodes())
        else:
            print("The problem does not have an optimal solution.")

        return assignment


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
        # wid_to_nwid = {wid: nwid for nwid, wid in enumerate(nwid_to_wid)}
        len_nwid_to_wid = len(nwid_to_wid)
        len_nbid_to_bid = len(nbid_to_bid)
        cost = np.zeros((len_nbid_to_bid, len_nwid_to_wid))
        u0 = np.zeros((len_nbid_to_bid, len_nwid_to_wid, 2))
        
        if self.CE:
            self.update_future_ia(state)

        # states = {(bid, nwid):None for bid in range(state.number_of_boats) for nwid in range(len(nwid_to_wid)) }
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
                cost[nbid, nwid] = output[0] #output[0] + output[3]
                u0[nbid, nwid, :] = output[1].reshape(2)
                #self.states[(bid, nwid)] = output[2]
        # assignments = self.mip_solver_2(cost)
        assinged_nbids, assigned_nwids = linear_sum_assignment(cost)
        #print("time:", state.time, " assignment_boat_wid", assignments)
        #print('cost: ', cost)
        # u0s = [np.zeros(2) for bid in range(state.number_of_boats)]
        assignment_bid_wid = {}
        cost_bid_wid = {(bid, wid): sys.maxsize for bid in range(state.number_of_boats) for wid in range(state.number_of_whales)}
        # for (nbid, nwid) in assignments:
        for aid, nbid in enumerate(assinged_nbids):
            nwid = assigned_nwids[aid]
            assignment_bid_wid[nbid_to_bid[nbid]] = nwid_to_wid[nwid]
            cost_bid_wid[(nbid_to_bid[nbid], nwid_to_wid[nwid])] = cost[nbid, nwid]
        #     u0s[assi[0]] = self.u0[assi[0], assi[1], :]
        #     print( 'boat:',assi[0], ' control:', self.u0[assi[0], assi[1], :], ' cost:', self.cost[assi[0], assi[1]])
        # return Boat_Control(b_theta = np.array([float(u0s[bid][0]) for bid in range(state.number_of_boats)]), b_v = np.array([float(u0s[bid][1]) for bid in range(state.number_of_boats)]))

        # print('base policy costs: ', cost)
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
        # print(bthetas, bvs)
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
