import argparse
import os
import sys
import copy
import numpy as np
import pandas as pd
# import pickle
from datetime import datetime
import gc

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

# np.random.seed(4937)

def csv_make_dir(type, data, out_csv):
    type_csv = f'{out_csv}_DATA_{run}_{ep}/{type}.csv'
    os.makedirs( os.path.dirname(type_csv), exist_ok=True )
    df = pd.DataFrame( data )
    df.to_csv(type_csv, index=False)

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
    prs.add_argument("-route", dest="route", type=str, default='nets/diamond/DiamondTLs.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-ag", dest="alpha_group", type=float, default=0.1, required=False, help="Group Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-gg", dest="gamma_group", type=float, default=0.99, required=False, help="Group Gamma discount rate.\n")
    prs.add_argument("-g0", dest="g_zero", type=int, default=3, required=False, help="Groups initial amount.\n")
    prs.add_argument("-gt", dest="threshold", type=float, default=0.2, required=False, help="Performance threshold to remove an agent from a group (0, 1].\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1, required=False, help="Epsilon.\n")
    prs.add_argument("-eg", dest="groupRecommendation", type=float, default=0.2, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.1, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.95, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=6000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-t", dest="teleport", type=int, default=200, required=False, help="Time to teleport vehicles.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-debugger", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-recommendation", action="store_false", default=True, help="Follow group recommendations at a fixed time.\n")
    prs.add_argument("-eps", dest="eps", type=int, default=1, help="Number of episodes.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0].split(' ')
    out_csv = 'outputs/testgroups_diamond/alpha{}_gamma{}_alphaG{}_gammaG{}_eps{}_decay{}_g0{}_gt{}_gr{}/{}/{}/'.format(args.alpha, args.gamma, args.alpha_group, args.gamma_group, args.epsilon, args.decay, args.g_zero, args.threshold, args.groupRecommendation, experiment_time[0], experiment_time[1])
    g0 = args.g_zero
    theta = 2
    threshold = args.threshold
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
    # distances = [[0 for x in range(len(vizinhos.keys()))] for y in range(len(vizinhos.keys()))]
    # for ts in vizinhos.keys():
        # BFS(vizinhos, ts, distances)
    # for dist in distances:
    #     print(dist)
    # exit();
    for ts in vizinhos.keys():
        env.neighbours[ts] = vizinhos[ts]

    for run in range(1, args.runs+1):
        backupGroups = {}
        initial_states = env.reset()
        lastSecond = args.seconds

        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_spaces(ts),
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 groupRecommendation=args.groupRecommendation,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        groups = groupingAgents(env.ts_ids, g0, theta, env, threshold)
        for g in groups.keys():
            for ts in groups[g].setTLs:
                groups[env.traffic_signals[ts].groupID].setNextStates.append(env.encode(initial_states[ts], ts))
            # print(groups); exit()
        for g in groups.keys():
            # groups[g].addState(groups[g].setNextStates)
            groups[g].state = copy.copy(groups[g].setNextStates)
            groups[g].setNextStates = []

        TSGroup = []
        for agent_id in range(0, len(env.ts_ids)):
            TSGroup.append(env.traffic_signals[env.ts_ids[agent_id]].groupID)

        twt = []

        groupRecommendation = 0
        if args.debugger:
            import cProfile
            import pstats

            profile = cProfile.Profile()
            profile.enable()

        for ep in range(1, args.eps+1):
            if args.recommendation:
                groupRecommendation = args.groupRecommendation
            else:
                if ep > args.eps*0.0:
                    if ep % 300 == 0:
                        groupRecommendation = 0
                    elif ep % 300 == 100:
                        groupRecommendation = 0.5
                    elif ep % 300 == 200:
                        groupRecommendation = 1

            print("RUN =", run, "EP =", ep)

            for ts in ql_agents.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)
                ql_agents[ts].groupRecommendation = groupRecommendation

            done = {'__all__': False}
            density = {'step_time': []}
            for ts in ql_agents.keys():
                density[ts] = []
            for ts in ql_agents.keys():
                density[ts+"s_a_ns_r"] = []
            density['groups'] = []
            density['recommendations'] = []

            # total_queued = {ts: [] for ts in ql_agents.keys()}
            # lanes_density = {ts: [] for ts in ql_agents.keys()}
            # lanes_queue = {ts: [] for ts in ql_agents.keys()}
            # out_lanes_density = {ts: [] for ts in ql_agents.keys()}
            # pressure = {ts: [] for ts in ql_agents.keys()}
            # waiting_time_per_lane = {ts: [] for ts in ql_agents.keys()}

            groupAmout = len(groups)

            for agent_id in range(0, len(env.ts_ids)):
                env.traffic_signals[env.ts_ids[agent_id]].groupID = TSGroup[agent_id]
                if TSGroup[agent_id] is not None:
                    env.traffic_signals[env.ts_ids[agent_id]].inGroup = True
            for g in groups.keys():
                groups[g].createdAt = env.sim_step

            if args.fixed:
                while not done['__all__']:
                    _, _, done, _ = env.step({})
            else:
                while not done['__all__']:
                    # print(env.sim_step)
                    density['step_time'].append(env.sim_step)
                    # ORGANIZAR MELHOR
                    numberOfSingletons = 0
                    for agent_id in env.ts_ids:
                        if not env.traffic_signals[agent_id].inGroup:
                            numberOfSingletons += 1

                    if numberOfSingletons/len(env.ts_ids)*100 > 50:
                        print("REAGRUPANDO")
                        groupAmout = g0
                        for g in groups.keys():
                            backupGroups[groups[g].printTLs()] = copy.copy(groups[g])
                            backupGroups[groups[g].printTLs()].id = None
                            backupGroups[groups[g].printTLs()].done = False

                        for agent_id in env.ts_ids:
                            env.traffic_signals[agent_id].groupID = None
                            env.traffic_signals[agent_id].inGroup = False

                        for g in list(groups):
                            del groups[g]

                        groups = {}
                        groups = groupingAgents(env.ts_ids, g0, theta, env, threshold)
                        for g in list(groups):
                            groups[g].checkNeighbours()
                            if groups[g].printTLs() in backupGroups.keys():
                                # print(groups[g].printTLs(), backupGroups.keys())
                                groups[g] = copy.copy(backupGroups[groups[g].printTLs()])
                                groups[g].id = g
                                for agent_id in groups[g].setTLs:
                                    env.traffic_signals[agent_id].groupID = g
                                    env.traffic_signals[agent_id].inGroup = True
                                    ql_agents[agent_id].groupActing = False
                                    ql_agents[agent_id].groupRecommendation = groupRecommendation
                                # print(groups[g], backupGroups)
                            else:
                                for agent_id in groups[g].setTLs:
                                    # print(groups[g], groups[g].setTLs, agent_id, next[agent_id])
                                    groups[g].id = g
                                    groups[g].setNextStates.append(next[agent_id])
                                    groups[g].setRewards[-1].append(r[agent_id])

                                # groups[g].addState(groups[g].setNextStates)
                                groups[g].state = copy.copy(groups[g].setNextStates)
                                groups[g].setNextStates = []
                            groups[g].createdAt = env.sim_step
                            groups[g].checkNeighbours()

                    # ORGANIZAR MELHOR
                    actionsGroups = {}
                    gKeys = list(groups.keys())
                    for g in gKeys:
                        if env.sim_step > lastSecond*0.1 + groups[g].createdAt and groupRecommendation > 0: # espera um tempo para começar a agir os agentes dos grupos
                            # print("tá entrando")
                            actionsGroups[g] = groups[g].act().replace('[', '').replace(']', '').split(',')
                            for agent_id in range(0, len(groups[g].setTLs)):
                                if actionsGroups[g][agent_id] == '':
                                    actionsGroups[g][agent_id] = 0
                                ql_agents[groups[g].setTLs[agent_id]].groupActing = True
                                ql_agents[groups[g].setTLs[agent_id]].groupAction = int(actionsGroups[g][agent_id])
                                # print(groups[g], actionsGroups[g][agent_id], ql_agents[groups[g].setTLs[agent_id]].groupAction, groups[g].setTLs[agent_id], env.ts_ids[agent_id])

                    # print(ql_agents.keys(), env.sim_step)

                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                    # Updates groups actions
                    for g in gKeys:
                        groups[g].action = []
                        for agent_id in groups[g].setTLs:
                            # print(env.ts_ids[agent_id], groups, env.traffic_signals[env.ts_ids[agent_id]].groupID)
                            groups[g].action.append(actions[agent_id])

                    for ts in env.traffic_signals:
                        density[ts].append(env.traffic_signals[ts].get_lanes_density())
                        # total_queued[ts].append( env.traffic_signals[ts].get_total_queued() )
                        # pressure[ts].append( env.traffic_signals[ts].get_pressure() )
                        # lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_density()) } )
                        # lanes_queue[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_queue()) } )
                        # out_lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].out_lanes, env.traffic_signals[ts].get_out_lanes_density()) } )
                        # waiting_time_per_lane[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_waiting_time_per_lane()) } )

                    states = {ts: [] for ts in ql_agents.keys()}

                    for ts in env.traffic_signals:
                        if ql_agents[ts].action == env.traffic_signals[ts].phase//2 and env.traffic_signals[ts].time_since_last_phase_change >= env.traffic_signals[ts].max_green:
                            phases = [i for i in range(0,ql_agents[ts].action_space.n)]
                            phases.remove(ql_agents[ts].action)
                            ql_agents[ts].action = np.random.choice(phases)
                            actions[ts] = ql_agents[ts].action
                        states[ts] = ql_agents[ts].state


                    s, r, done, _ = env.step(action=actions)

                    for ts in env.traffic_signals:
                        next_state=env.encode(s[ts], ts)
                        density[ts+"s_a_ns_r"].append([states[ts], actions[ts], next_state, r[ts]])

                    density['groups'].append(str(groups))
                    density['recommendations'].append(groupRecommendation)
                    # print(density['groups'][-1], groups)

                    next = {}
                    for agent_id in s.keys():
                        next[agent_id] = env.encode(s[agent_id], agent_id)
                        # print("-->>", next[agent_id], s)
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                    for g in gKeys:
                        groups[g].setNextStates = []
                        for agent_id in groups[g].setTLs:
                            # print(groups[g], groups[g].setTLs, agent_id, next[agent_id])
                            groups[g].setNextStates.append(next[agent_id])
                            groups[g].setRewards[-1].append(r[agent_id])

                    # ORGANIZAR MELHOR
                    for g in list(groups):
                        if groups[g].setTLs:
                            # groups[g].addState(groups[g].setNextStates)
                            groups[g].addAction(groups[g].action)
                            # print(groups[g].action, groups[g], groups[g].setTLs)
                            groups[g].learn()

                        if env.sim_step > lastSecond*0.2 + groups[g].createdAt and groups[g].threshold > 0: # espera um tempo para começar a remover os agentes dos grupos
                            # print("tá entrando para remover")
                            removed = groups[g].removingGroup()
                            for tl in removed:
                                env.traffic_signals[agent_id].inGroup = False
                                env.traffic_signals[agent_id].groupID = None

                            if removed:
                                print("GROUPS BEING REMOVED->", groups[g], removed)
                                deletedGroupName = ';'.join(groups[g].setTLs)
                                backupGroups[deletedGroupName] = copy.copy(groups[g])
                                newGroupTLs = []
                                for agent_id in groups[g].setTLs:
                                    env.traffic_signals[agent_id].groupID = None
                                    env.traffic_signals[agent_id].inGroup = False
                                    ql_agents[agent_id].groupActing = False
                                    ql_agents[agent_id].groupRecommendation = groupRecommendation

                                for TL in groups[g].setTLs:
                                    if TL not in removed:
                                        newGroupTLs.append(TL)

                                # (len(groups[g].setTLs) - len(removed)) > 1

                                if newGroupTLs:
                                    newGroupID = groups[list(groups.keys())[-1]].id+1
                                    # print("ID:", newGroupID)
                                    # print(backupGroups, deletedGroupName)
                                    # groupAmout += 1
                                    newGroupName = (';'.join(newGroupTLs))
                                    if newGroupName in backupGroups.keys():
                                        groups[newGroupID] = copy.copy(backupGroups[newGroupName])
                                        groups[newGroupID].id = newGroupID
                                        groups[newGroupID].action = []
                                        groups[newGroupID].state = []
                                        groups[newGroupID].setNextStates = []
                                        for TL in newGroupTLs:
                                            env.traffic_signals[TL].groupID = newGroupID
                                            env.traffic_signals[TL].inGroup = True
                                            groups[newGroupID].action.append(actions[TL])
                                            ql_agents[agent_id].groupActing = False
                                            ql_agents[agent_id].groupRecommendation = groupRecommendation
                                            groups[newGroupID].state.append(states[TL])
                                            groups[newGroupID].setNextStates.append(next[TL])
                                            groups[newGroupID].setRewards[-1].append(r[TL])
                                    else:
                                        # print("NEW", newGroupName, backupGroups, newGroupTLs)
                                        groups[newGroupID] = Groups(newGroupID, env, threshold, args.alpha_group, args.gamma_group)
                                        groups[newGroupID].action = []
                                        groups[newGroupID].state = []
                                        groups[newGroupID].setNextStates = []
                                        for TL in newGroupTLs:
                                            groups[newGroupID].addGroup(TL)
                                            groups[newGroupID].checkNeighbours()
                                            groups[newGroupID].setNextStates.append(next[TL])
                                            groups[newGroupID].action.append(actions[TL])
                                            groups[newGroupID].setRewards[-1].append(r[TL])
                                            # print(groups[newGroupID].setNextStates)
                                            groups[newGroupID].state.append(states[TL])

                                        # print(groups[newGroupID].setNextStates, groups[newGroupID].action, groups[newGroupID].state)
                                        # x = input()
                                        # groups[newGroupID].addState(groups[newGroupID].state)
                                        # groups[newGroupID].addState(groups[newGroupID].setNextStates)
                                        groups[newGroupID].addAction(groups[newGroupID].action)
                                        # groups[newGroupID].learn()
                                        # groups[newGroupID].action = []
                                    groups[newGroupID].createdAt = env.sim_step
                                    # groups[newGroupID].learn()

                                del groups[g]

                        if g in groups.keys():
                            groups[g].action = []
                            groups[g].setRewards.append([])

                    # for agent_id in s.keys():
                        # print(agent_id, env.traffic_signals[agent_id].groupID)

            env.save_csv(out_csv, run, ep)
            # types = [   ['total_queued', total_queued],
            #         ['lanes_queue', lanes_queue],
            #         ['lanes_density', lanes_density],
            #         # ['out_lanes_density', out_lanes_density],
            #         # ['pressure', pressure],
            #         ['waiting_time_per_lane', waiting_time_per_lane]
            #     ]
            # for type,data in types:
                # csv_make_dir( type, data, out_csv  )
            # for g in groups.keys():
            #     with open(out_csv+"QTable_"+str(g)+"_"+str(ep)+'.pickle', 'wb') as handle:
            #         pickle.dump(groups[g].qTable, handle, protocol=pickle.HIGHEST_PROTOCOL)

            df = pd.DataFrame(env.metrics)
            twt.append(df['total_wait_time'].sum())
            density_csv = out_csv+'_densities_run{}_ep{}.csv'.format(run, ep)
            os.makedirs(os.path.dirname(density_csv), exist_ok=True)
            den = pd.DataFrame(density)
            den.to_csv(density_csv, index=False)
            lastSecond = env.sim_step
            del density
            del den
            del df
            del TSGroup
            # del types

            TSGroup = []
            for agent_id in range(0, len(env.ts_ids)):
                TSGroup.append(env.traffic_signals[env.ts_ids[agent_id]].groupID)
            gc.collect()
            if ep != args.eps:
                initial_states = env.reset()
            if ep == args.eps:
                env.close()
        if args.debugger:
            profile.disable()
            ps = pstats.Stats(profile)
            ps.print_stats()
        print(twt)
