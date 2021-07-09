import argparse
import os
import sys
import pandas as pd
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

def csv_make_dir(type, data, out_csv):
    type_csv = f'{out_csv}_DATA_{run}_{ep}/{type}.csv'
    os.makedirs( os.path.dirname(type_csv), exist_ok=True )
    df = pd.DataFrame( data )
    df.to_csv(type_csv, index=False)

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Diamond Network""")
    prs.add_argument("-route", dest="route", type=str, default='nets/diamond/DiamondTLs.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=1, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.05, required=False, help="Minimum epsilon.\n")
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
    experiment_time = str(datetime.now()).split('.')[0].split(' ')
    out_csv = 'outputs/diamondWT/alpha{}_gamma{}_eps{}_decay{}/{}/{}/'.format(args.alpha, args.gamma, args.epsilon, args.decay, experiment_time[0], experiment_time[1])

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
        initial_states = env.reset()
        # print(initial_states, env.action_space)
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_spaces(ts),
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}

        twt = []
        for ep in range(1, args.eps+1):
            print("RUN =", run, "EP =", ep)
            done = {'__all__': False}
            density = {'step_time': []}
            for ts in ql_agents.keys():
                density[ts] = []
            for ts in ql_agents.keys():
                density[ts+"s_a_ns_r"] = []
            # print(density,len(env.ts_ids))

            total_queued = {ts: [] for ts in ql_agents.keys()}
            lanes_density = {ts: [] for ts in ql_agents.keys()}
            lanes_queue = {ts: [] for ts in ql_agents.keys()}
            out_lanes_density = {ts: [] for ts in ql_agents.keys()}
            pressure = {ts: [] for ts in ql_agents.keys()}
            waiting_time_per_lane = {ts: [] for ts in ql_agents.keys()}

            if args.fixed:
                while not done['__all__']:
                    _, _, done, _ = env.step({})
            else:
                while not done['__all__']:
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                    density['step_time'].append(env.sim_step)
                    for ts in env.traffic_signals:
                        density[ts].append(env.traffic_signals[ts].get_lanes_density())
                        total_queued[ts].append( env.traffic_signals[ts].get_total_queued() )
                        pressure[ts].append( env.traffic_signals[ts].get_pressure() )
                        lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_density()) } )
                        lanes_queue[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_lanes_queue()) } )
                        out_lanes_density[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].out_lanes, env.traffic_signals[ts].get_out_lanes_density()) } )
                        waiting_time_per_lane[ts].append( { lane: data for lane, data in zip(env.traffic_signals[ts].lanes, env.traffic_signals[ts].get_waiting_time_per_lane()) } )


                    states = {ts: [] for ts in ql_agents.keys()}
                    for ts in env.traffic_signals:
                        states[ts] = ql_agents[ts].state

                    s, r, done, _ = env.step(action=actions)

                    # print(r, env.traffic_signals)

                    for ts in env.traffic_signals:
                        next_state=env.encode(s[ts], ts)
                        density[ts+"s_a_ns_r"].append([states[ts], actions[ts], next_state, r[ts]])

                    # print(density)#; exit()

                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(out_csv, run, ep)
            types = [   ['total_queued', total_queued],
                    ['lanes_queue', lanes_queue],
                    ['lanes_density', lanes_density],
                    ['out_lanes_density', out_lanes_density],
                    ['pressure', pressure],
                    ['waiting_time_per_lane', waiting_time_per_lane]
                ]
            for type,data in types:
                csv_make_dir( type, data, out_csv  )

            df = pd.DataFrame(env.metrics)
            twt.append(df['total_wait_time'].sum())
            density_csv = out_csv+'_{}_{}_densities.csv'.format(run, ep)
            os.makedirs(os.path.dirname(density_csv), exist_ok=True)
            df = pd.DataFrame(density)
            df.to_csv(density_csv, index=False)

            if ep != args.eps:
                initial_states = env.reset()
            if ep == args.eps:
                env.close()
        print(twt)
