import argparse
import os
import sys
import copy
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
        groups[i] = Groups(i, env, threshold, args.alpha_group, args.gamma_group)
        groups[i].addGroup(randomAgents[i])

    for i in range(0, g0):
        groups[i].checkNeighbours()

    while True:
        for i in range(0, g0):
            # print(groups[i], groups[i].neighbours, groups[i].done)
            groups[i].checkNeighbours()
            if groups[i].done == True:
                continue
            if groups[i].neighbours:
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
    prs.add_argument("-ag", dest="alpha_group", type=float, default=0.1, required=False, help="Group Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-gg", dest="gamma_group", type=float, default=0.99, required=False, help="Group Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.1, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.95, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=6000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-t", dest="teleport", type=int, default=300, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-eps", dest="eps", type=int, default=1, help="Number of episodes.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/diamond_tests/{}_alpha{}_gamma{}_alphaG{}_gammaG{}_eps{}_decay{}'.format(experiment_time, args.alpha, args.gamma, args.alpha_group, args.gamma_group, args.epsilon, args.decay)
    g0 = 3
    theta = 2
    threshold = 0.2
    numberOfSingletons = 0
    backupGroups = {}
    lastSecond = args.seconds

    env = SumoEnvironment(net_file='nets/diamond/DiamondTLs.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=-1,
                          time_to_teleport=args.teleport)

    initial_states = env.reset()

    vizinhos = foundAllNeighbours(env)
    for ts in vizinhos.keys():
        env.neighbours[ts] = vizinhos[ts]

    for run in range(1, args.runs+1):
        initial_states = env.reset()
        env.run = run
        lastSecond = args.seconds

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_spaces(ts),
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        groups = groupingAgents(env.ts_ids, g0, theta, env, threshold)
        for g in groups.keys():
            for ts in groups[g].setTLs:
                groups[env.traffic_signals[ts].groupID].setNextStates.append(env.encode(initial_states[ts], ts))
            # print(groups); exit()
        for g in groups.keys():
            groups[g].addState(groups[g].setNextStates)
            groups[g].state = copy.deepcopy(groups[g].setNextStates)
            groups[g].setNextStates = []

        TSGroup = []
        for agent_id in range(0, len(env.ts_ids)):
            TSGroup.append(env.traffic_signals[env.ts_ids[agent_id]].groupID)

        twt = []

        for ep in range(1, args.eps+1):
            print("RUN =", run, "EP =", ep)

            done = {'__all__': False}
            density = {ts: [] for ts in ql_agents.keys()}

            for agent_id in range(0, len(env.ts_ids)):
                env.traffic_signals[env.ts_ids[agent_id]].groupID = TSGroup[agent_id]
            for g in groups.keys():
                groups[g].createdAt = env.sim_step

            infos = []
            if args.fixed:
                while not done['__all__']:
                    _, _, done, _ = env.step({})
            else:
                while not done['__all__']:
                    # print(env.sim_step)

                    # ORGANIZAR MELHOR
                    if numberOfSingletons/len(env.ts_ids) > 0.5:
                        print("REAGRUPANDO")
                        for g in groups.keys():
                            backupGroups[groups[g].printTLs()] = copy.deepcopy(groups[g])
                            backupGroups[groups[g].printTLs()].id = None

                        for agent_id in env.ts_ids:
                            env.traffic_signals[agent_id].groupID = None
                            env.traffic_signals[agent_id].inGroup = False

                        for g in list(groups):
                            del groups[g]

                        numberOfSingletons = 0
                        groups = {}
                        groups = groupingAgents(env.ts_ids, g0, theta, env, threshold)
                        for g in list(groups):
                            if groups[g].printTLs() in backupGroups.keys():
                                groups[g] = copy.deepcopy(backupGroups[groups[g].printTLs()])
                                groups[g].id = g
                                for agent_id in groups[g].setTLs:
                                    env.traffic_signals[agent_id].groupID = g
                                    env.traffic_signals[agent_id].inGroup = True
                                    ql_agents[agent_id].groupActing = False
                                    ql_agents[agent_id].epsilonGroup = 1
                                # print(groups[g], backupGroups)
                            else:
                                for agent_id in groups[g].setTLs:
                                    # print(groups[g], groups[g].setTLs, agent_id, next[agent_id])
                                    groups[g].setNextStates.append(next[agent_id])
                                    groups[g].setRewards[-1].append(r[agent_id])

                                groups[g].addState(groups[g].setNextStates)
                                groups[g].state = copy.deepcopy(groups[g].setNextStates)
                                groups[g].setNextStates = []
                            groups[g].createdAt = env.sim_step

                    # ORGANIZAR MELHOR
                    actionsGroups = {}
                    for g in groups.keys():
                        if env.sim_step > lastSecond*0.1 + groups[g].createdAt: # espera um tempo para começar a agir os agentes dos grupos
                            # print("tá entrando")
                            actionsGroups[g] = groups[g].act().replace('[', '').replace(']', '').split(',')
                            for agent_id in range(0, len(groups[g].setTLs)):
                                ql_agents[groups[g].setTLs[agent_id]].groupActing = True
                                ql_agents[groups[g].setTLs[agent_id]].groupAction = int(actionsGroups[g][agent_id])
                                # print(groups[g], actionsGroups[g][agent_id], ql_agents[groups[g].setTLs[agent_id]].groupAction, groups[g].setTLs[agent_id], env.ts_ids[agent_id])

                    # print(ql_agents.keys(), env.sim_step)
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                    # Updates groups actions
                    for g in groups.keys():
                        for agent_id in groups[g].setTLs:
                            # print(env.ts_ids[agent_id], groups, env.traffic_signals[env.ts_ids[agent_id]].groupID)
                            groups[g].action.append(actions[agent_id])

                    for ts in env.traffic_signals:
                        density[ts].append(env.traffic_signals[ts].get_lanes_density())

                    s, r, done, _ = env.step(action=actions)

                    next = {}
                    for agent_id in s.keys():
                        next[agent_id] = env.encode(s[agent_id], agent_id)
                        # print("-->>", next[agent_id], s)
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                    for g in groups.keys():
                        for agent_id in groups[g].setTLs:
                            # print(groups[g], groups[g].setTLs, agent_id, next[agent_id])
                            groups[g].setNextStates.append(next[agent_id])
                            groups[g].setRewards[-1].append(r[agent_id])

                    # ORGANIZAR MELHOR
                    for g in list(groups):
                        groups[g].addState(groups[g].setNextStates)
                        groups[g].addAction(groups[g].action)
                        groups[g].learn()

                        if env.sim_step > lastSecond*0.2 + groups[g].createdAt: # espera um tempo para começar a remover os agentes dos grupos
                            # print("tá entrando para remover")
                            removed = groups[g].removingGroup()
                            if removed:
                                print("GROUPS BEING REMOVED->", groups[g], removed)
                                backupGroups[groups[g].printTLs()] = copy.deepcopy(groups[g])
                                newGroupTLs = []
                                for agent_id in groups[g].setTLs:
                                    env.traffic_signals[agent_id].groupID = None
                                    env.traffic_signals[agent_id].inGroup = False
                                    ql_agents[agent_id].groupActing = False
                                    ql_agents[agent_id].epsilonGroup = 1

                                # for agent_id in s.keys():
                                    # print(env.traffic_signals[agent_id].groupID)
                                numberOfSingletons += len(removed)

                                for TL in groups[g].setTLs:
                                    if TL not in removed:
                                        newGroupTLs.append(TL)
                                if newGroupTLs:
                                    newGroupID = int(list(groups.keys())[-1]) + 1
                                    newGroupName = (';'.join(newGroupTLs))
                                    if newGroupName in backupGroups.keys():
                                        groups[newGroupID] = copy.deepcopy(backupGroups[newGroupName])
                                        groups[newGroupID].id = newGroupID
                                        for TL in newGroupTLs:
                                            env.traffic_signals[TL].groupID = newGroupID
                                            env.traffic_signals[TL].inGroup = True
                                            ql_agents[agent_id].groupActing = False
                                            ql_agents[agent_id].epsilonGroup = 1
                                    else:
                                        groups[newGroupID] = Groups(newGroupID, env, threshold, args.alpha_group, args.gamma_group)
                                        for TL in newGroupTLs:
                                            groups[newGroupID].addGroup(TL)
                                            groups[newGroupID].checkNeighbours()
                                            groups[newGroupID].setNextStates.append(next[TL])
                                            groups[newGroupID].action.append(actions[TL])
                                            groups[newGroupID].setRewards[-1].append(r[TL])
                                            # print(groups[g].state)
                                            groups[newGroupID].state.append(groups[g].state[groups[g].setTLs.index(TL)])

                                        # print(groups[newGroupID].setNextStates, groups[newGroupID].action, groups[newGroupID].state)
                                        # x = input()
                                        groups[newGroupID].addState(groups[newGroupID].state)
                                        groups[newGroupID].addState(groups[newGroupID].setNextStates)
                                        groups[newGroupID].addAction(groups[newGroupID].action)
                                        # groups[newGroupID].learn()
                                        groups[newGroupID].action = []
                                    groups[newGroupID].createdAt = env.sim_step
                                del groups[g]

                        if g in groups.keys():
                            groups[g].action = []
                            groups[g].setRewards.append([])

                    # for agent_id in s.keys():
                        # print(agent_id, env.traffic_signals[agent_id].groupID)

            env.save_csv(out_csv, run, ep)
            df = pd.DataFrame(env.metrics)
            twt.append(df['total_wait_time'].sum())
            density_csv = out_csv+'_densities_run{}_ep{}.csv'.format(run, ep)
            os.makedirs(os.path.dirname(density_csv), exist_ok=True)
            df = pd.DataFrame(density)
            df.to_csv(density_csv, index=False)
            lastSecond = env.sim_step

            TSGroup = []
            for agent_id in range(0, len(env.ts_ids)):
                TSGroup.append(env.traffic_signals[env.ts_ids[agent_id]].groupID)

            if ep != args.eps:
                initial_states = env.reset()
            if ep == args.eps:
                env.close()
        print(twt)
        env.run += 1
