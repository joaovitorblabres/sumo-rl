import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd
import gc

from .traffic_signal import TrafficSignal


class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                 time_to_teleport=-1, delta_time=5, yellow_time=2, min_green=5, max_green=50, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.val = 0
        self.neighbours = {}

        traci.start([sumolib.checkBinary('sumo'), '-n', self._net, '-r', self._route])  # start only to retrieve information

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        traci.close()

    def reset(self):
        # gc.collect()
        try:
            traci.close()
        except Exception as e:
            pass
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run, 0)
        # self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay),
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                     # '--max-num-vehicles', str(800),
                     # '--seed', str(4937),
                     '--random',
                     '--no-warnings',
                     '--quit-on-end']
        if self.use_gui:
            sumo_cmd.append('--start')

        # print(sumo_cmd)
        traci.start(sumo_cmd)

        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
                if self.sim_step % 5 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)
        else:
            self._apply_actions(action)

            time_to_act = False
            while not time_to_act:
                self._sumo_step()

                for ts in self.ts_ids:
                    self.traffic_signals[ts].update()
                    if self.traffic_signals[ts].time_to_act:
                        time_to_act = True

                if self.sim_step % 5 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time or traci.vehicle.getIDCount() == 0}
        done.update({ts_id: False for ts_id in self.ts_ids})

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], done['__all__'], {}
        else:
            return observations, rewards, done, {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        # return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids}

    def _compute_rewards(self):
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        # return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].discrete_observation_space

    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        if traci.vehicle.getIDCount() == 0:
            self.val = 1
        else:
            self.val = traci.vehicle.getIDCount()
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids),
            'flow': sum(self.traffic_signals[ts].flow for ts in self.ts_ids),
            'vehicles_on_network': traci.vehicle.getIDCount(),
            'teleported_vehicles': traci.simulation.getEndingTeleportNumber() ,
            'average_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids) / self.val
            # 'average_pressure': sum(self.traffic_signals[ts]._pressure_reward() for ts in self.ts_ids) / val,
        }

    def close(self):
        traci.close()

    def save_csv(self, out_csv_name, run, ep):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            os.makedirs(os.path.dirname(out_csv_name + '_run{}_ep{}'.format(run, ep) + '.csv'), exist_ok=True)
            df.to_csv(out_csv_name + '_run{}_ep{}'.format(run, ep) + '.csv', index=False)


    # Below functions are for discrete state space

    # def encode(self, state, ts_id):
    #     phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
    #     elapsed = self._discretize_elapsed_time(self.traffic_signals[ts_id].time_since_last_phase_change)
    #     density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases:]]
    #     # tuples are hashable and can be used as key in python dictionary
    #     # print(tuple([phase] + [elapsed] + density_queue))
    #     return tuple([phase] + [elapsed] + density_queue)
        # return tuple([phase] + density_queue)

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        elapsed = self._discretize_elapsed_time(self.traffic_signals[ts_id].time_since_last_phase_change)
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases:]]
        # return self.radix_encode([phase] + density_queue, ts_id)
        return self.radix_encode([phase] + [elapsed] + density_queue, ts_id)

    def _discretize_density(self, density):
        # print(density*100)
        return min(int(density*4), 3)

    def _discretize_elapsed_time(self, elapsed):
        # print(elapsed, self.delta_time, min(1, elapsed / self.delta_time))
        # return min(1, elapsed / self.delta_time)
        # elapsed *= self.max_green
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                # print(i)
                return i
        # print("?", elapsed, elapsed//self.delta_time -1)
        return elapsed//self.delta_time

    def radix_encode(self, values, ts_id):
        res = 0
        self.radix_factors = [s.n for s in self.traffic_signals[ts_id].discrete_observation_space.spaces]
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]

        # print(res)
        return int(res)

    """ def radix_decode(self, value):
        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res """
