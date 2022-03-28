import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import traci.constants as tc
import numpy as np
import pandas as pd
import gc
import random
import itertools
from gym import spaces
from sklearn.preprocessing import MaxAbsScaler as scaler

class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, collect):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        self.last_measure = 0.0
        self.last_cars = 0.0
        self.out_cars = 0.0
        self.last_reward = [0,0]
        self.inGroup = False
        self.groupID = None
        self.dic_vehicles = {}
        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        self.num_green_phases = len(self.phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.flow = 0
        self.__normalizer = None
        self.__collect = collect
        self.collecting = None
        self.vehLanes = {}
        self.linearComb = []
        self.rewards = []
        for lane in self.lanes:
            self.vehLanes[lane] = {}

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 1 + 1*len(self.lanes)), high=np.ones(self.num_green_phases + 1 + 1*len(self.lanes)))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            spaces.Discrete(self.max_green//self.delta_time),            # Elapsed time of phase
            #*(spaces.Discrete(10) for _ in range(2*len(self.lanes))),    # Queue for each lane
            *(spaces.Discrete(4) for _ in range(1*len(self.lanes)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        programs = traci.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.phases
        traci.trafficlight.setProgramLogic(self.id, logic)
        # gc.collect()

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # print(self.id, self.green_phase)
            traci.trafficlight.setPhase(self.id, int(self.green_phase))
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases]
        """
        new_phase *= 2
        # print(self.time_since_last_phase_change)
        if self.time_since_last_phase_change < self.max_green and self.phase == new_phase or self.time_since_last_phase_change < self.min_green + self.yellow_time:# or self.time_to_act:
            self.green_phase = self.phase
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow
            self.next_action_time = self.env.sim_step + self.delta_time# + self.yellow_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        elapsed = [self.time_since_last_phase_change // self.max_green]
        # print(elapsed)
        # density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        # observation = np.array(phase_id + queue)
        # observation = np.array(phase_id + density + queue)
        # observation = np.array(phase_id + elapsed + density + queue)
        observation = np.array(phase_id + elapsed + queue)
        # print(phase_id , density , queue)
        return observation

    @property
    def normalizer(self) -> scaler:
        if self.__normalizer is None:
            # path = "./outputs/BASELINE/alpha0.3_gamma0.8_eps0.05_decay1.0/2021-11-08/09:05:03/"
            path = "./outputs/FIXED/gamma0.8_eps0.05_decay1.0/2021-12-16/18:44:31/"
            fit_file = f"{path}fit_data.csv"
            # print(fit_file)
            try:
                fit_data = pd.read_csv(fit_file).to_numpy()
                self.__normalizer = scaler().fit(fit_data)
            except FileNotFoundError:
                err_str = "Fit data must be in scenario directory."
                err_str += " Please run simulation with flag '--collect' before"
                raise RuntimeError(err_str) from FileNotFoundError
        return self.__normalizer

    def collector(self, last_reward):
        if self.collecting is None:
            self.collecting = pd.DataFrame({"0": [0.0], "1": [0]})

        df = pd.DataFrame({"0": [last_reward[0]], "1": [last_reward[1]]})
        self.collecting = pd.concat((self.collecting, df), ignore_index=True)
        # print(self.collecting)

    def compute_reward(self, normalize = True, testing = False):
        # self.last_reward = self._waiting_time_reward()
        # self.dic_vehicles = self.update_vehicles_state(self.dic_vehicles)
        # self.last_reward = self.get_rewards_from_sumo(self.dic_vehicles, self.phase)[0]
        # self.last_reward = [self._queue_reward(), self.get_rewards_from_sumo(self.dic_vehicles, self.phase)[0]]
        # self.last_reward = [self._queue_reward(), self._pressure_reward()]
        if testing:
            last_reward = [self._queue_reward(), self.get_flow(), -self.get_avg_travel_time(), -self.get_avg_CO2(), -self.get_avg_CO2(), -self.get_avg_CO(), -self.get_avg_HCE(), -self.get_avg_NOx(), -self.get_avg_PMx(), -self.get_avg_fuel(), self.get_avg_speed()]
        else:
            last_reward = [self._queue_reward(), self.get_flow()]
        # print(last_reward, testing)
        # self.last_reward = [self._queue_reward(), self._pressure_reward()]
        # last_reward = [self.get_flow(), self._queue_reward()]
        # self.get_flow()
        # self.last_reward = [self._queue_reward(), 0]

        # linear = self.normalizer.transform([np.array(last_reward)])[0] if normalize else np.array(last_reward)
        # m = 0.5
        # self.linearComb.append(-1*m*linear[0] + (1-m)*linear[1])

        # if last_reward[0] == 0 and last_reward[1] == 0:
        #     last_reward[0] = last_reward[0] + random.random()/1000
        #     last_reward[1] = last_reward[1] + random.random()/1000
        # self.linearComb.append(np.linalg.solve([[1, 1], last_reward], self.last_reward))
        # self.linearComb.append(np.linalg.solve([self.last_reward, last_reward], [1, 1]))
        # self.linearComb.append(np.linalg.norm(last_reward))
        # print(self.linearComb[-1])
        self.last_reward = last_reward
        self.rewards.append(last_reward)
        if self.__collect:
            self.collector(last_reward)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _out_lanes_reward(self):
        return self.get_out_lanes_density()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_lanes_queue()))#*10

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_flow(self):
        flow = 0
        keys = list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes]))
        for lane in self.lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            self.vehLanes[lane] = {}
            for veh in vehicles:
                if veh not in self.vehLanes[lane]:
                    self.vehLanes[lane][veh] = 1

# print(list(itertools.chain.from_iterable(a)))
        # print('now', list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes])), 'last', keys)
        # print(list(set(keys) - set(list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes])))))
        # print(len(list(set(keys) - set(list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes]))))))

        # new_cars = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        # out_cars = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        # reward = (out_cars - self.out_cars) - (new_cars - self.last_cars)
        self.last_cars = len(list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes])))
        # self.out_cars = out_cars
        self.flow = len(list(set(keys) - set(list(itertools.chain.from_iterable([list(self.vehLanes[lane].keys()) for lane in self.lanes])))))
        # print(self.flow)
        return self.flow

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return sum([min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes])

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def get_total_vehicles(self):
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def get_avg_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]) / len(self.lanes)

    def get_avg_speed(self):
        return sum([traci.lane.getLastStepMeanSpeed(lane) if traci.lane.getLastStepVehicleNumber(lane) else 0 for lane in self.lanes])/len(self.lanes)

    def get_avg_travel_time(self):
        return sum([traci.lane.getTraveltime(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_travel_time_2(self):
        vehicle_id_entering_list = self.get_vehicle_id_entering()
        return self.get_travel_time_duration(self.dic_vehicles, vehicle_id_entering_list)

    def get_travel_times(self):
        return [traci.lane.getTraveltime(lane) for lane in self.lanes]

    def get_avg_CO2(self):
        return sum([traci.lane.getCO2Emission(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_CO(self):
        return sum([traci.lane.getCOEmission(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_HCE(self):
        return sum([traci.lane.getHCEmission(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_NOx(self):
        return sum([traci.lane.getNOxEmission(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_PMx(self):
        return sum([traci.lane.getPMxEmission(lane) for lane in self.lanes])/len(self.lanes)

    def get_avg_fuel(self):
        return sum([traci.lane.getFuelConsumption(lane) for lane in self.lanes])/len(self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

# ------------------------------ HUA WEI --------------------------------

    def update_vehicles_state(self, dic_vehicles):
        vehicle_id_list = traci.vehicle.getIDList()
        vehicle_id_entering_list = self.get_vehicle_id_entering()
        for vehicle_id in (set(dic_vehicles.keys())-set(vehicle_id_list)):
            del(dic_vehicles[vehicle_id])

        for vehicle_id in vehicle_id_list:
            if (vehicle_id in dic_vehicles.keys()) == False:
                vehicle = Vehicles()
                vehicle.id = vehicle_id
                traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
                vehicle.speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
                current_sumo_time = traci.simulation.getCurrentTime()/1000
                vehicle.enter_time = current_sumo_time
                # if it enters and stops at the very first
                if (vehicle.speed < 0.1) and (vehicle.first_stop_time == -1):
                    vehicle.first_stop_time = current_sumo_time
                dic_vehicles[vehicle_id] = vehicle
            else:
                dic_vehicles[vehicle_id].speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
                if (dic_vehicles[vehicle_id].speed < 0.1) and (dic_vehicles[vehicle_id].first_stop_time == -1):
                    dic_vehicles[vehicle_id].first_stop_time = traci.simulation.getCurrentTime()/1000
                if (vehicle_id in vehicle_id_entering_list) == False:
                    dic_vehicles[vehicle_id].entering = False

        return dic_vehicles

    def get_rewards_from_sumo(self, vehicle_dict, action):
        listLanes = self.lanes
        reward = 0
        import copy
        reward_detail_dict ={  "delay": [
                                    True,
                                    -0.25
                                ],
                                "flickering": [
                                    True,
                                    -5
                                ],
                                "emergency": [
                                    False,
                                    0.1667
                                ],
                                "queue_length": [
                                    True,
                                    -0.25
                                ],
                                "wait_time": [
                                    True,
                                    -0.25
                                ],
                                "duration": [
                                    False,
                                    1
                                ],
                                "partial_duration": [
                                    False,
                                    1
                                ],
                                "num_of_vehicles_left": [
                                    True,
                                    1
                                ],
                                "duration_of_vehicles_left": [
                                    True,
                                    1
                                ]
                            }

        vehicle_id_entering_list = self.get_vehicle_id_entering()

        reward_detail_dict['queue_length'].append(self.get_overall_queue_length(listLanes))
        reward_detail_dict['wait_time'].append(self.get_overall_waiting_time(listLanes))
        reward_detail_dict['delay'].append(self.get_overall_delay(listLanes))
        reward_detail_dict['emergency'].append(self.get_num_of_emergency_stops(vehicle_dict))
        reward_detail_dict['duration'].append(self.get_travel_time_duration(vehicle_dict, vehicle_id_entering_list))
        reward_detail_dict['flickering'].append(self.get_flickering(action))
        reward_detail_dict['partial_duration'].append(self.get_partial_travel_time_duration(vehicle_dict, vehicle_id_entering_list))

        vehicle_id_list = traci.vehicle.getIDList()
        reward_detail_dict['num_of_vehicles_in_system'] = [False, 0, len(vehicle_id_list)]

        reward_detail_dict['num_of_vehicles_at_entering'] = [False, 0, len(vehicle_id_entering_list)]


        vehicle_id_leaving = self.get_vehicle_id_leaving(vehicle_dict)

        reward_detail_dict['num_of_vehicles_left'].append(len(vehicle_id_leaving))
        reward_detail_dict['duration_of_vehicles_left'].append(self.get_travel_time_duration(vehicle_dict, vehicle_id_leaving))



        for k, v in reward_detail_dict.items():
            if v[0]:  # True or False
                reward += v[1]*v[2]
        reward = self.restrict_reward(reward)#,func="linear")
        return reward, reward_detail_dict

    def restrict_reward(self, reward,func="unstrict"):
        if func == "linear":
            bound = -50
            reward = 0 if reward < bound else (reward/(-bound) + 1)
        elif func == "neg_log":
            reward = math.log(-reward+1)
        else:
            pass

        return reward

    def get_overall_queue_length(self, listLanes):
        overall_queue_length = 0
        for lane in listLanes:
            overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        return overall_queue_length

    def get_overall_waiting_time(self, listLanes):
        overall_waiting_time = 0
        for lane in listLanes:
            overall_waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
        return overall_waiting_time

    def get_overall_delay(self, listLanes):
        overall_delay = 0
        for lane in listLanes:
            overall_delay += 1 - traci.lane.getLastStepMeanSpeed(str(lane)) / traci.lane.getMaxSpeed(str(lane))
        return overall_delay

    def get_flickering(self, action):
        return action

    # calculate number of emergency stops by vehicle
    def get_num_of_emergency_stops(self, vehicle_dict):
        emergency_stops = 0
        vehicle_id_list = traci.vehicle.getIDList()
        for vehicle_id in vehicle_id_list:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            current_speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            if (vehicle_id in vehicle_dict.keys()):
                vehicle_former_state = vehicle_dict[vehicle_id]
                if current_speed - vehicle_former_state.speed < -4.5:
                    emergency_stops += 1
            else:
                # print("##New car coming")
                if current_speed - Vehicles.initial_speed < -4.5:
                    emergency_stops += 1
        if len(vehicle_dict) > 0:
            return emergency_stops/len(vehicle_dict)
        else:
            return 0

    def get_partial_travel_time_duration(self, vehicle_dict, vehicle_id_list):
        travel_time_duration = 0
        for vehicle_id in vehicle_id_list:
            if (vehicle_id in vehicle_dict.keys()) and (vehicle_dict[vehicle_id].first_stop_time != -1):
                travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].first_stop_time)/60.0
        if len(vehicle_id_list) > 0:
            return travel_time_duration#/len(vehicle_id_list)
        else:
            return 0


    def get_travel_time_duration(self, vehicle_dict, vehicle_id_list):
        travel_time_duration = 0
        for vehicle_id in vehicle_id_list:
            if (vehicle_id in vehicle_dict.keys()):
                travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].enter_time)/60.0
        if len(vehicle_id_list) > 0:
            return travel_time_duration/len(vehicle_id_list)
        else:
            return 0


    def get_vehicle_id_entering(self):
        vehicle_id_entering = []
        entering_lanes = self.lanes

        for lane in entering_lanes:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))

        return vehicle_id_entering

    def get_vehicle_id_leaving(self, vehicle_dict):
        vehicle_id_leaving = []
        vehicle_id_entering = self.get_vehicle_id_entering()
        for vehicle_id in vehicle_dict.keys():
            if not(vehicle_id in vehicle_id_entering) and vehicle_dict[vehicle_id].entering:
                vehicle_id_leaving.append(vehicle_id)

        return vehicle_id_leaving


class Vehicles:
    initial_speed = 5.0

    def __init__(self):
        # add what ever you need to maintain
        self.id = None
        self.speed = None
        self.wait_time = None
        self.stop_count = None
        self.enter_time = None
        self.has_read = False
        self.first_stop_time = -1
        self.entering = True
