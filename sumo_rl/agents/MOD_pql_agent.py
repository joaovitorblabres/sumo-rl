import numpy as np
from pygmo import hypervolume
import random
import warnings

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

def paretoEfficient(points, return_mask = True, repeated = False, minimize = True):
    """
    Find the (minimizing) pareto-efficient points
    :param points: An (n_points, n_points) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(points.shape[0])
    n_points = points.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(points):
        if minimize:
            nondominated_point_mask = np.any(points < points[next_point_index], axis=1)
        else:
            nondominated_point_mask = np.any(points > points[next_point_index], axis=1)
        if repeated:
            for i in range(points.shape[0]):
                if np.array_equal(points[next_point_index], points[i]):
                    nondominated_point_mask[i] = True
        else:
            nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        points = points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

    else:
        return is_efficient

def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    # print(solutions)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(c > solutions[is_efficient], axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return solutions[is_efficient]


def compute_hypervolume(q_set, nA, ref):
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.asarray(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

class modPQLAgent:

    def __init__(self, starting_state, state_space, action_space, ref_point, gamma=0.95, exploration_strategy=EpsilonGreedy(), number_obejctives=2, grouped=0, gt=0.5):
        self.state = starting_state
        self.state_space = 1
        for space in state_space:
            self.state_space *= space.n

        self.action_space = action_space
        self.action = None
        self.lenAct = []
        self.actRnd = []
        self.gamma = gamma
        self.groupTheta = gt
        self.minEpsilonGroup = 0.05
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = np.zeros(number_obejctives)
        self.followed = False
        self.number_obejctives = number_obejctives
        self.grouped = grouped
        np.set_printoptions(suppress=True)

        # self.non_dominated = [[[np.random.uniform(-100,100,2)] for _ in range(self.action_space.n)] for _ in range(self.state_space)]
        self.non_dominated = [[[np.array([0,0])] for _ in range(self.action_space.n)] for _ in range(self.state_space)]
        # self.non_dominated = {self.state: [np.zeros(number_obejctives) for _ in range(self.action_space.n)]}
        # self.avg_others = np.array([np.array([np.array([0.,0.]) for _ in range(self.action_space.n)]) for _ in range(self.state_space)])
        self.avg_others = np.zeros((self.state_space, self.action_space.n, number_obejctives))
        # self.avg_r = np.array([np.array([np.array([100.,0.]) for _ in range(self.action_space.n)]) for _ in range(self.state_space)])
        self.avg_r = np.zeros((self.state_space, self.action_space.n, number_obejctives))
        # self.avg_r = {self.state: [np.zeros(number_obejctives) for _ in range(self.action_space.n)]}
        # self.n_visits = np.array([np.array([np.array(0) for _ in range(self.action_space.n)]) for _ in range(self.state_space)])
        self.n_visits = np.zeros((self.state_space, self.action_space.n))
        # print(self.avg_r)
        # self.n_visits = {self.state: [np.zeros(number_obejctives) for _ in range(self.action_space.n)]}
        self.ref_point = ref_point
        # print(self.non_dominated, self.action_space.n)

    def compute_q_set(self, s):
        m = self.groupTheta
        warnings.simplefilter("ignore")
        q_set = []
        for a in range(self.action_space.n):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s][a]

            if self.grouped:
                rew = (m*rew) + ((1-m)*self.avg_others[s][a])
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
            # print([nd for nd in nd_sa])
        return np.asarray(q_set)

    def update_non_dominated(self, s, a, s_n):
        q_set_n = self.compute_q_set(s_n)
        # update for all actions, flatten
        solutions = np.concatenate(q_set_n, axis=0)
        # append to current pareto front
        # print(solutions, self.non_dominated[s][a])
        # solutions = np.concatenate([solutions, self.non_dominated[s][a]])
        solutions = np.unique(solutions.round(decimals=4), axis=0)

        # compute pareto front
        # print(solutions, s_n, q_set_n, paretoEfficient(solutions, False, True))
        # nonD = np.array([list(solutions[x]) for x in paretoEfficient(solutions, False, True, False)])
        # nonD = paretoEfficient(solutions, False, False, False)
        # myset = solutions[nonD]
        # print("NON", solutions[nonD], myset, get_non_dominated(solutions))
        # self.non_dominated[s][a] = myset
        self.non_dominated[s][a] = get_non_dominated(solutions)

    def act(self):
        q_set = self.compute_q_set(self.state)
        # print("PRE", q_set, self.state, self.action_space)
        self.lenAct.append(0)
        self.actRnd.append(0)
        self.action = self.exploration.choose(q_set, self.state, self.action_space)
        self.lenAct[-1] = self.exploration.lenAct
        self.actRnd[-1] = self.exploration.actRnd
        return self.action

    def learn(self, next_state, reward, done=False):
        # if next_state not in self.non_dominated:
        #     self.non_dominated[next_state] = [[np.zeros(self.number_obejctives)] for _ in range(self.action_space.n)]
        # if next_state not in self.avg_r:
        #     self.avg_r[next_state] = [[np.zeros(self.number_obejctives)] for _ in range(self.action_space.n)]
        # if next_state not in self.n_visits:
        #     self.n_visits[next_state] = [[np.zeros(self.number_obejctives)] for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        # print(s, a, s1, self.action_space.n)

        self.update_non_dominated(s, a, s1)
        self.n_visits[s][a] += 1
        # print(self.avg_r[s][a], (reward - self.avg_r[s][a]) / self.n_visits[s][a])
        self.avg_r[s][a] += (reward - self.avg_r[s][a]) / self.n_visits[s][a]
        # print(self.avg_r[s][a], self.non_dominated[s][a])
        # self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
