import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl.environment.groups import Groups
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

np.random.seed(4937)

def groupingAgents(agents, g0, theta, env, threshold):
    groups = {}
    randomAgents = np.random.choice(agents, size=g0, replace=False)
    for i in range(0, g0):
        groups[i] = Groups(i, env, threshold)
        groups[i].addGroup(randomAgents[i])

    for i in range(0, g0):
        groups[i].checkNeighbours()

    while True:
        for i in range(0, g0):
            # print(groups[i], groups[i].neighbours)
            if groups[i].done == True:
                continue
            groups[i].addGroup(np.random.choice(groups[i].neighbours))
            groups[i].checkNeighbours()

            # print(groups[i], next, groups[i].neighbours)

        if all([groups[j].done == True for j in groups.keys()]):
            break
    return groups

def foundAllNeighbours(env):
    vizinhos = {}
    for i in range(0, len(env.ts_ids)):
        # print(traci.trafficlight.getControlledLinks(ts)[0][0][1]); exit()
        ts_i = env.ts_ids[i]
        # print(traci.trafficlight.getControlledLinks(ts_i)[0][0][0]); exit()
        for j in range(i+1, len(env.ts_ids)):
            ts_j = env.ts_ids[j]
            for link_i in traci.trafficlight.getControlledLinks(ts_i):
                for link_j in traci.trafficlight.getControlledLinks(ts_j):
                    if link_i[0][0] == link_j[0][1] or link_i[0][1] == link_j[0][0]:
                        if ts_i in vizinhos:
                            if ts_j not in vizinhos[ts_i]:
                                vizinhos[ts_i].append(ts_j)
                        else:
                            vizinhos[ts_i] = [ts_j]
                        if ts_j in vizinhos:
                            if ts_i not in vizinhos[ts_j]:
                                vizinhos[ts_j].append(ts_i)
                        else:
                            vizinhos[ts_j] = [ts_i]
    return vizinhos

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Diamond Network""")
    prs.add_argument("-route", dest="route", type=str, default='nets/diamond/DiamondTLs.rou.alt.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.1, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.95, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=8000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-eps", dest="eps", type=int, default=1, help="Number of episodes.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/diamond/{}_alpha{}_gamma{}_eps{}_decay{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay)
    g0 = 3
    theta = 2
    threshold = 0.2

    env = SumoEnvironment(net_file='nets/diamond/DiamondTLs.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=-1,
                          time_to_teleport=300)

    initial_states = env.reset()

    vizinhos = foundAllNeighbours(env)
    for ts in vizinhos.keys():
        env.neighbours[ts] = vizinhos[ts]

    for run in range(1, args.runs+1):
        initial_states = env.reset()

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_spaces(ts),
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        for ep in range(1, args.eps+1):
            groups = groupingAgents(env.ts_ids, g0, theta, env, threshold)
            # print(groups); exit()

            done = {'__all__': False}
            density = {ts: [] for ts in ql_agents.keys()}

            infos = []
            if args.fixed:
                while not done['__all__']:
                    _, _, done, _ = env.step({})
            else:
                while not done['__all__']:
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                    for ts in env.traffic_signals:
                        density[ts].append(env.traffic_signals[ts].get_lanes_density())

                    s, r, done, _ = env.step(action=actions)

                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(out_csv, run)
            density_csv = out_csv+'_{}_{}_densities.csv'.format(run, ep)
            os.makedirs(os.path.dirname(density_csv), exist_ok=True)
            df = pd.DataFrame(density)
            df.to_csv(density_csv, index=False)

            if ep != args.eps:
                initial_states = env.reset()
            if ep == args.eps:
                env.close()
