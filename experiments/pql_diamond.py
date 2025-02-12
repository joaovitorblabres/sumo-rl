import argparse
import os
import sys
import numpy as np
import copy
import pandas as pd
from datetime import datetime
import cProfile, pstats, io
from pstats import SortKey

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import mPQLAgent
from sumo_rl.agents import PQLAgent
from sumo_rl.exploration import MOSelection
# from sumo_rl.util import Integral

def csv_make_dir(type, data, out_csv):
    type_csv = f'{out_csv}_DATA_{run}_{ep}/{type}.csv'
    os.makedirs( os.path.dirname(type_csv), exist_ok=True )
    df = pd.DataFrame( data )
    df.to_csv(type_csv, index=False)

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Diamond Network""")
    prs.add_argument("-route", dest="route", type=str, default='nets/diamond/DiamondTLs.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.15, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.05, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=0.95, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-algType", dest="algType", type=int, default=0, required=False, help="Int: PQL action selection. 0 = Hypervolume, 1 = Pareto Selection, default: 0.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-debug", action="store_true", default=False, help="Debug spent time in each function.\n")
    prs.add_argument("-optimize", action="store_true", default=False, help="Debug spent time in each function.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=200000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-eps", dest="eps", type=int, default=1, help="Number of episodes.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0].split(' ')
    debug = args.debug
    out_csv = 'outputs/{}{}{}{}PQL/gamma{}_eps{}_decay{}/{}/{}/'.format(['OPT_' if args.optimize else ''][0],['PO_' if args.algType else 'HV_'][0], ['FIXED_' if args.fixed else ''][0], ['DEBUG_' if args.debug else ''][0], args.gamma, args.epsilon, args.decay, experiment_time[0], experiment_time[1])

    env = SumoEnvironment(net_file='nets/diamond/DiamondTLs.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=-1,
                          time_to_teleport=300)

    for run in range(1, args.runs+1):
        if debug:
            pr = cProfile.Profile()
            pr.enable()
        initial_states = env.reset()
        # print(initial_states, env.ts_ids[6])
        # ql_agents = {ts: mPQLAgent(starting_state=env.encode(initial_states[ts], ts),
        #                          state_space=env.observation_spaces(ts),
        #                          action_space=env.action_spaces(ts),
        #                          alpha=args.alpha,
        #                          gamma=args.gamma,
        #                          ref_point=np.array([-10000, -10000]),
        #                          exploration_strategy=MOSelection(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay, ref=np.array([-10000, -10000]))) for ts in env.ts_ids}
        ql_agents = {ts: PQLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_spaces(ts),
                                 action_space=env.action_spaces(ts),
                                 gamma=args.gamma,
                                 ref_point=np.asarray([-80, 0]),
                                 exploration_strategy=MOSelection(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay, ref=np.asarray([-80, 0]), algType=args.algType)) for ts in env.ts_ids}

        twt = []
        volumes = []
        for ep in range(1, args.eps+1):
            print("RUN =", run, "EP =", ep)
            done = {'__all__': False}
            # density = {'step_time': []}
            # for ts in ql_agents.keys():
                # density[ts] = []
            # for ts in ql_agents.keys():
                # density[ts+"s_a_ns_r"] = []
            # print(density,len(env.ts_ids))

            # total_queued = {ts: [] for ts in ql_agents.keys()}
            # lanes_density = {ts: [] for ts in ql_agents.keys()}
            # lanes_queue = {ts: [] for ts in ql_agents.keys()}
            # out_lanes_density = {ts: [] for ts in ql_agents.keys()}
            # pressure = {ts: [] for ts in ql_agents.keys()}
            # waiting_time_per_lane = {ts: [] for ts in ql_agents.keys()}

            if args.fixed:
                while not done['__all__']:
                    density['step_time'].append(env.sim_step)
                    actions = {ts: [] for ts in ql_agents.keys()}
                    states = {ts: [] for ts in ql_agents.keys()}
                    for ts in ql_agents.keys():
                        ql_agents[ts].action = actions[ts] = env.traffic_signals[ts].green_phase
                        states[ts] = ql_agents[ts].state

                    s, r, done, _ = env.step({})
                    for ts in ql_agents.keys():
                        next_state=env.encode(s[ts], ts)
                        density[ts+"s_a_ns_r"].append([states[ts], actions[ts], next_state, r[ts], env.traffic_signals[ts].flow, sum(env.traffic_signals[ts].get_waiting_time_per_lane())/max(1,env.traffic_signals[ts].last_cars), env.traffic_signals[ts].last_cars, env.traffic_signals[ts].get_avg_speed(), env.traffic_signals[ts].get_avg_travel_time(), env.traffic_signals[ts].get_lanes_queue(), env.traffic_signals[ts].get_avg_CO2(), env.traffic_signals[ts].get_avg_CO(), env.traffic_signals[ts].get_avg_HCE(), env.traffic_signals[ts].get_avg_NOx(), env.traffic_signals[ts].get_avg_PMx(), env.traffic_signals[ts].get_avg_fuel(), env.traffic_signals[ts].get_avg_queued(), env.traffic_signals[ts].get_avg_travel_time_2()])
                        # if env.traffic_signals[ts].get_avg_travel_time() > 10000:
                        #     print(density[ts+"s_a_ns_r"][-1], env.traffic_signals[ts].get_travel_times(), env.traffic_signals[ts].get_avg_travel_time_2())
                            # input()
                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
            else:
                # j = 0
                while not done['__all__']:
                    if env.sim_step % 1000 == 0:
                        print(env.sim_step, env.val)
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                    # density['step_time'].append(env.sim_step)
                    # for ts in ql_agents.keys():
                    #     # density[ts].append(env.traffic_signals[ts].get_lanes_density())
                    #     total_queued[ts].append( env.traffic_signals[ts].get_total_queued() )
                    #     # pressure[ts].append( env.traffic_signals[ts].get_pressure() )
                    #     lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_density()) } )
                    #     lanes_queue[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_queue()) } )
                    #     # out_lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].out_lanes, env.traffic_signals[ts].get_out_lanes_density()) } )
                    #     waiting_time_per_lane[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_waiting_time_per_lane()) } )

                    states = {ts: [] for ts in ql_agents.keys()}

                    for ts in ql_agents.keys():
                        if ql_agents[ts].action == env.traffic_signals[ts].phase//2 and env.traffic_signals[ts].time_since_last_phase_change >= env.traffic_signals[ts].max_green:
                            phases = [i for i in range(0,ql_agents[ts].action_space.n)]
                            phases.remove(ql_agents[ts].action)
                            ql_agents[ts].action = np.random.choice(phases)
                            actions[ts] = ql_agents[ts].action
                        states[ts] = ql_agents[ts].state

                    s, r, done, _ = env.step(action=actions)

                    # print(s, r, actions)

                    for ts in ql_agents.keys():
                        # print(env.traffic_signals[ts].linearComb[-1])
                        # print(ql_agents[ts].non_dominated[0][:])
                        next_state=env.encode(s[ts], ts)
                        # density[ts+"s_a_ns_r"].append([states[ts], actions[ts], next_state, r[ts], env.traffic_signals[ts].flow, sum(env.traffic_signals[ts].get_waiting_time_per_lane())/max(1,env.traffic_signals[ts].last_cars), env.traffic_signals[ts].last_cars, env.traffic_signals[ts].get_avg_speed(), env.traffic_signals[ts].get_avg_travel_time(), env.traffic_signals[ts].get_lanes_queue(), env.traffic_signals[ts].get_avg_CO2(), env.traffic_signals[ts].get_avg_CO(), env.traffic_signals[ts].get_avg_HCE(), env.traffic_signals[ts].get_avg_NOx(), env.traffic_signals[ts].get_avg_PMx(), env.traffic_signals[ts].get_avg_fuel(), env.traffic_signals[ts].get_avg_queued(), env.traffic_signals[ts].get_avg_travel_time_2()])

                    # print(density); exit()

                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(out_csv, run, ep)
        #     types = [
        #             ['total_queued', total_queued],
        #             ['lanes_queue', lanes_queue],
        #             ['lanes_density', lanes_density],
        #             # ['out_lanes_density', out_lanes_density],
        #             # ['pressure', pressure],
        #             ['waiting_time_per_lane', waiting_time_per_lane]
        #         ]
        #     # for type,data in types:
        #         # csv_make_dir( type, data, out_csv  )
        #
        #     df = pd.DataFrame(env.metrics)
        #     twt.append(df['average_wait_time'].sum())
        #     twt.append(df['flow'].sum())
        #     density_csv = out_csv+'_{}_{}_densities.csv'.format(run, ep)
        #     os.makedirs(os.path.dirname(density_csv), exist_ok=True)
        #     df = pd.DataFrame(density)
        #     df.to_csv(density_csv, index=False)
        #
        #     for ts in ql_agents.keys():
        #          NDt = out_csv+'NDt_{}_{}_{}.csv'.format(ts, run, ep)
        #          avgr = out_csv+'AVGr_{}_{}_{}.csv'.format(ts, run, ep)
        #          os.makedirs(os.path.dirname(NDt), exist_ok=True)
        #          os.makedirs(os.path.dirname(avgr), exist_ok=True)
        #
        #          # print(ql_agents[ts].non_dominated[0][:], ql_agents[ts].avg_r[0][:])
        #          non = pd.DataFrame(np.concatenate(ql_agents[ts].non_dominated[0][:]))
        #          avg = pd.DataFrame(ql_agents[ts].avg_r[0][:])
        #          non.to_csv(NDt, index=False)
        #          avg.to_csv(avgr, index=False)
        #
        #     for ts in ql_agents.keys():
        #          LC = out_csv+'LC_{}_{}_{}.csv'.format(ts, run, ep)
        #          os.makedirs(os.path.dirname(LC), exist_ok=True)
        #
        #          non = pd.DataFrame(env.traffic_signals[ts].linearComb[:])
        #          non.to_csv(LC, index=False)
        #
        #     for ts in ql_agents.keys():
        #          LC = out_csv+'REWARDS_{}_{}_{}.csv'.format(ts, run, ep)
        #          os.makedirs(os.path.dirname(LC), exist_ok=True)
        #
        #          non = pd.DataFrame(env.traffic_signals[ts].rewards[:])
        #          non.to_csv(LC, index=False)
        #
        #     NDt = out_csv+'NDt_ALL_{}_{}.csv'.format(run, ep)
        #     os.makedirs(os.path.dirname(NDt), exist_ok=True)
        #     main_df = pd.DataFrame()
        #     for ts in ql_agents.keys():
        #         sets = []
        #         for action in range(len(ql_agents[ts].non_dominated[0])):
        #             for set in ql_agents[ts].non_dominated[0][action]:
        #                 sets.append(np.absolute(set))
        #                 # if np.any(np.isclose(ql_agents[ts].avg_r[0][action],0)):
        #                 #     sets.append(set/1)
        #                 # else:
        #                 #     sets.append(set/ql_agents[ts].avg_r[0][action])
        #                 # print(set/ql_agents[ts].avg_r[0][action])
        #         df = pd.DataFrame(sets)
        #         main_df = pd.concat((main_df, df), ignore_index=True)
        #          # print(pd.DataFrame(np.concatenate(ql_agents[ts].non_dominated[0][:])))
        #
        #     main_df.sort_values(by=1, ascending=False).to_csv(NDt, index=False)
        #     integ_cls = Integral(file=NDt, y=0, x=1, method='trapz')
        #     vals = integ_cls.integrate_files()
        #     volumes.append(vals[0][0])
        #
        #     ACT = out_csv+'ACT_{}_{}.csv'.format(run, ep)
        #     os.makedirs(os.path.dirname(ACT), exist_ok=True)
        #     epLen = range(len(ql_agents['B5'].lenAct))
        #     lenAct = []
        #     for ts in ql_agents.keys():
        #         for i in epLen:
        #             if len(lenAct) == i:
        #                 lenAct.append(ql_agents[ts].lenAct[i])
        #             else:
        #                 lenAct[i] += ql_agents[ts].lenAct[i]
        #     non = pd.DataFrame(lenAct)
        #     non.to_csv(ACT, index=False)
        #
        #     RNDACT = out_csv+'RNDACT_{}_{}.csv'.format(run, ep)
        #     os.makedirs(os.path.dirname(RNDACT), exist_ok=True)
        #     epLen = range(len(ql_agents['B5'].lenAct))
        #     actRnd = []
        #     for ts in ql_agents.keys():
        #         for i in epLen:
        #             if len(actRnd) == i:
        #                 actRnd.append(ql_agents[ts].actRnd[i])
        #             else:
        #                 actRnd[i] += ql_agents[ts].actRnd[i]
        #     non = pd.DataFrame(actRnd)
        #     non.to_csv(RNDACT, index=False)
        #
            if ep != args.eps:
                initial_states = env.reset()
            if ep == args.eps:
                env.close()
        # print(twt, volumes)
        # volume = out_csv+'Vol_{}_{}.csv'.format(run, ep)
        # vol = pd.DataFrame(volumes)
        # vol.to_csv(volume)
        # for i in s.keys():
        #     for m in range(0,len(ql_agents[i].avg_r)):
        #         if np.any(ql_agents[i].avg_r[m] < 0):
        #             print(m, ql_agents[i].avg_r[m])
        if debug:
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
