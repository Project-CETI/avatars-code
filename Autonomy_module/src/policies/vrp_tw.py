from src.global_knowledge import Global_knowledge
from src.belief_state import Belief_State, Policy_base
from src.configs.constants import Boat_Control
import numpy as np
import copy
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from geopy import Point, distance                                                                                                                                                                    
from geopy.distance import geodesic 
from geographiclib.geodesic import Geodesic

class VRP_TW(Policy_base):
    
    def __init__(self, knowledge: Global_knowledge, state: Belief_State) -> None:
        self.knowledge = knowledge
        self.setup_MPC_for_last_mile(knowledge, 60 / self.knowledge.observations_per_minute)

        self.policy_name = 'VRP_TW_no_replan_CE'
        self.assignment_new = {}
        self.assignment_without_replanning = True
        self.solve_vrp(state)
            
            
    
    def generate_data(self, state: Belief_State):
        nb = state.number_of_boats
        data = {}
        num_untagged_whales = state.number_of_whales - len(state.assigned_whales)
        data['whale_id_map_r'] = {}
        nwid = 0
        for wid in range(state.number_of_whales):
            if wid not in state.assigned_whales:
                data['whale_id_map_r'][nwid] = wid
                nwid += 1

        data['drone_id_map_r'] = {did: did for did in range(nb)}
        data['num_available_agents'] = nb

        data['time_windows'] = []
        for p in range(nb):
            data['time_windows'].append((0, 100)) #depot p
        for nwid in range(num_untagged_whales):
            wid = data['whale_id_map_r'][nwid]
            if state.whale_up2[wid] == True:
                next_end = int(state.w_last_surface_start_time[wid] \
                    + state.knowledge.surface_time_mean - state.time)/self.knowledge.observations_per_minute
                if next_end < 0:
                    next_end = 1
                data['time_windows'].append((0, next_end))
            else:
                next_start = int(state.w_last_surface_end_time[wid] \
                    + state.knowledge.down_time_mean - state.time)/self.knowledge.observations_per_minute
                if next_start < 0:
                    next_start = 1
                next_end = next_start + int(state.knowledge.surface_time_mean/self.knowledge.observations_per_minute)
                data['time_windows'].append((next_start , next_end))

        data['time_matrix'] = []
        whale_meet_time = [data['time_windows'][nb + nwid][0] \
            for nwid in range(num_untagged_whales)]
        data['whale_meet_loc'] = []
        for nwid in range(num_untagged_whales):
            wid = data['whale_id_map_r'][nwid]
            data['whale_meet_loc'].append((state.w_x[wid] + state.w_v[wid] * whale_meet_time[nwid] * np.cos(state.w_theta[wid]), \
                state.w_y[wid] + state.w_v[wid] * whale_meet_time[nwid] * np.sin(state.w_theta[wid])))
            # data['whale_meet_loc'].append((self.mean_bar_particles[wid][0] + self.gk.average_speed_whale * whale_meet_time[nwid] * np.cos(self.mean_bar_particles[wid][2]), \
            #         self.mean_bar_particles[wid][1] + self.gk.average_speed_whale * whale_meet_time[nwid] * np.sin(self.mean_bar_particles[wid][2]) ))
        for p1 in range(nb):
            line = np.zeros(nb + num_untagged_whales)

            if 1==1 : #not self.knowledge.use_GPS_coordinate_from_dataset:
                line[: nb] = np.ceil(np.linalg.norm(np.array([state.b_x[p1], state.b_y[p1]]) \
                    - np.array([state.b_x, state.b_y]).reshape(2, nb).transpose(), axis=1) /state.knowledge.boat_max_speed_mtpm).astype(int)

                line[nb: nb + num_untagged_whales] = np.ceil(np.linalg.norm(np.array([state.b_x[p1], state.b_y[p1]]) \
                    - np.array([data['whale_meet_loc']]).reshape(num_untagged_whales, 2) , axis=1) /state.knowledge.boat_max_speed_mtpm).astype(int)
            else:
                for p2 in range(nb):
                    line[p2] = int(np.ceil(distance.distance(Point(state.b_y[p1], state.b_x[p1]), Point(state.b_y[p2], state.b_x[p2])).m \
                        / state.knowledge.boat_max_speed_mtpm))
                for w in range(num_untagged_whales):
                    line[nb + w] = int(np.ceil(distance.distance(Point(state.b_y[p1], state.b_x[p1]), Point(data['whale_meet_loc'][w][1], data['whale_meet_loc'][w][0])).m \
                        / state.knowledge.boat_max_speed_mtpm))

            # for p2 in range(nb):
            #     line[p2] = int(round(np.linalg.norm(np.array([state.b_x[p1], state.b_y[p1]]) - np.array([state.b_x[p2], state.b_y[p2]]))/state.knowledge.boat_max_speed_mtpm))
            # for nwid in range(num_untagged_whales):
            #     wid = data['whale_id_map_r'][nwid]
            #     line[nb + nwid] = int(round(state.knowledge.calculate_distance((state.b_x[p1], state.b_y[p1]), \
            #         data['whale_meet_loc'][nwid]) / state.knowledge.boat_max_speed_mtpm))
            data['time_matrix'].append(line.tolist())
        for nwid1 in range(num_untagged_whales):
            line = np.zeros(nb + num_untagged_whales)
            if 1==1: # not self.knowledge.use_GPS_coordinate_from_dataset:
                line[0 : nb] = np.ceil(np.linalg.norm(np.array([data['whale_meet_loc'][nwid1]]) \
                    - np.array([state.b_x, state.b_y]).reshape(2, nb).transpose(), axis=1) / state.knowledge.boat_max_speed_mtpm).astype(int)
                line[nb : nb + num_untagged_whales] = np.ceil(np.linalg.norm(np.array([data['whale_meet_loc'][nwid1]]) \
                    - np.array([data['whale_meet_loc']]).reshape(num_untagged_whales, 2), axis=1) / state.knowledge.boat_max_speed_mtpm).astype(int)
            else:
                for p1 in range(nb):
                    line[p] = int(np.ceil(distance.distance(Point(state.b_y[p1], state.b_x[p1]), \
                        Point(data['whale_meet_loc'][nwid1][1], data['whale_meet_loc'][nwid1][0])).m / state.knowledge.boat_max_speed_mtpm))
                for nwid2 in range(num_untagged_whales):
                    line[nb + nwid2] = int(np.ceil(distance.distance(Point(data['whale_meet_loc'][nwid1][1], data['whale_meet_loc'][nwid1][0]), \
                        Point(data['whale_meet_loc'][nwid2][1], data['whale_meet_loc'][nwid2][0])).m / state.knowledge.boat_max_speed_mtpm))
            
            # for p2 in range(nb):
            #     line[p2] = int(round(state.knowledge.calculate_distance(data['whale_meet_loc'][nwid1], (state.b_x[p2], state.b_y[p2])) / state.knowledge.boat_max_speed_mtpm))
            # for nwid2 in range(num_untagged_whales):
            #     line[nb + nwid2] = int(round(state.knowledge.calculate_distance(data['whale_meet_loc'][nwid1], data['whale_meet_loc'][nwid2]) \
                    # /state.knowledge.boat_max_speed_mtpm))
            data['time_matrix'].append(line.tolist())

        
        data['num_vehicles'] = nb
        data['depot'] = [p for p in range(nb)]
        self.data = data
        # print(data)
        return data

    def get_control(self, state: Belief_State):

        if self.assignment_without_replanning == False:
            return self.get_control_with_replanning(state)
        bthetas = []
        bvs = []
        print(self.assignment_new)
        for bid in self.assignment_new.keys():
            # if self.assignment[bid] >= self.data['num_available_agents']:
            
            next_wid_to_visit = None
            for (next_wid, start_time, end_time) in self.assignment_new[bid]:
                if state.time/self.knowledge.observations_per_minute <= end_time:
                    next_wid_to_visit = next_wid
                    break
            if next_wid_to_visit is not None:    
                wid = self.data['whale_id_map_r'][next_wid_to_visit - self.data['num_available_agents']]
                u = self.get_MPC_control(state, bid, wid)
                bthetas.append(float(u[0]))
                bvs.append(float(u[1]))
            else:
                bthetas.append(state.b_theta[bid])
                bvs.append(0)
        for bid in range(state.number_of_boats):
            if bid not in self.assignment_new.keys():
                bthetas.append(state.b_theta[bid])
                bvs.append(0)

        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))

    def solve_vrp(self, state: Belief_State):
    
        self.data = self.generate_data(state)

        
        self.assignment = {}
        self.visit_times = {}
        
        manager = pywrapcp.RoutingIndexManager(len(self.data['time_matrix']), \
            self.data['num_vehicles'], self.data['depot'], self.data['depot'])
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return self.data['time_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        routing.AddDimension( 
            transit_callback_index,
            200,  # allow waiting time
            200,  # maximum time per vehicle
            False,  # Don't force start cumul to zero.
            time)
        time_dimension = routing.GetDimensionOrDie(time)
        for location_idx, time_window in enumerate(self.data['time_windows']):
            if location_idx in self.data['depot']:
                continue
            index = manager.NodeToIndex(location_idx)
	        #########Penalty
            routing.AddDisjunction([index], 100)
            #########Penalty
            time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))
            
            

        # Add time window constraints for each vehicle start node.
        depot_idx = self.data['depot']
        for vehicle_id in range(self.data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                self.data['time_windows'][depot_idx[vehicle_id]][0],
                self.data['time_windows'][depot_idx[vehicle_id]][1])
    
        for i in range(self.data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i)))
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i)))
            

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            time_dimension = routing.GetDimensionOrDie('Time')
            total_time = 0
            if self.assignment_without_replanning == False:
                for vehicle_id in range(self.data['num_vehicles']):
                    index = routing.Start(vehicle_id)
                    # plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                    n = 0
                    while not routing.IsEnd(index):
                        if n == 2:
                            break
                        time_var = time_dimension.CumulVar(index)
                        next_node = manager.IndexToNode(index)
                        visit_time = (solution.Min(time_var), solution.Max(time_var))
                        index = solution.Value(routing.NextVar(index))
                        n += 1
                    self.assignment[vehicle_id] = next_node
                    self.visit_times[vehicle_id] = visit_time
            else:
                self.assignment_new = {}
                for vehicle_id in range(self.data['num_vehicles']):
                    index = routing.Start(vehicle_id)
                    self.assignment_new[vehicle_id] = []
                    while not routing.IsEnd(index):
                        time_var = time_dimension.CumulVar(index)
                        next_node_id = manager.IndexToNode(index)
                        start_time = solution.Min(time_var)
                        end_time = solution.Max(time_var)
                        if next_node_id != vehicle_id:
                            self.assignment_new[vehicle_id].append((next_node_id, start_time, end_time))
                        index = solution.Value(routing.NextVar(index))
        else:
            print('no solution')

    def get_control_with_replanning(self, state: Belief_State):
        self.solve_vrp(state)

        bthetas = []
        bvs = []
        for bid in range(state.number_of_boats):
            if bid in self.assignment.keys() and self.assignment[bid] >= self.data['num_available_agents']:
                wid = self.data['whale_id_map_r'][self.assignment[bid] - self.data['num_available_agents']]
                # target_loc = (state.w_x[wid], state.w_y[wid])
                # source_loc = (state.b_x[bid], state.b_y[bid])
                u = self.get_MPC_control(state, bid, wid)
                bthetas.append(float(u[0]))
                bvs.append(float(u[1]))
            else:
                bthetas.append(state.b_theta[bid])
                bvs.append(0)
        for bid in range(state.number_of_boats):
            if bid not in self.assignment.keys():
                bthetas.append(state.b_theta[bid])
                bvs.append(0)
            #     if not self.knowledge.use_GPS_coordinate_from_dataset:
            #         bthetas.append(np.arctan2(target_loc[1] - source_loc[1], target_loc[0] - source_loc[0]))
            #         bvs.append(min(state.knowledge.boat_max_speed_mtpm, np.sqrt((target_loc[1] - source_loc[1])**2 + (target_loc[0] - source_loc[0])**2)))
            #     else:
            #         vel = distance.distance(Point(target_loc[1], target_loc[0]), Point(source_loc[1], source_loc[0])).m
            #         bearing = Geodesic.WGS84.Inverse(source_loc[1], source_loc[0], target_loc[1], target_loc[0])['azi1']
            #         bthetas.append(bearing * self.knowledge.deg_to_radian)
            #         bvs.append(min(state.knowledge.boat_max_speed_mtpm, vel))
            # else:
            #     bthetas.append(state.b_theta[bid])
            #     bvs.append(0)
        return Boat_Control(b_theta = np.array(bthetas), b_v = np.array(bvs))
        # return bel.construct_control_from_assignment(assignment)
        #return bel.construct_control_from_assignment_waithome_gohome(assignment, visit_times)