import numpy as np
from pygmo import hypervolume
import random

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

def get_non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1

    return solutions[is_efficient]


def compute_hypervolume(q_set, nA, ref):
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

class PQLAgent:

    def __init__(self, starting_state, state_space, action_space, ref_point, gamma=0.95, exploration_strategy=EpsilonGreedy(), groupRecommendation=0.2, number_obejctives=2):
        self.state = starting_state
        self.state_space = 1
        for space in state_space:
            self.state_space *= space.n

        self.action_space = action_space
        self.action = None
        self.gamma = gamma
        self.groupAction = None
        self.groupActing = False
        self.groupRecommendation = groupRecommendation
        self.decayGroup = 1
        self.minEpsilonGroup = 0.05
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = np.zeros(number_obejctives)
        self.followed = False

        self.non_dominated = [[[np.zeros(number_obejctives)] for _ in range(self.action_space.n)] for _ in range(self.state_space)]
        self.avg_r = np.zeros((self.state_space, self.action_space.n, number_obejctives))
        self.n_visits = np.zeros((self.state_space, self.action_space.n))
        self.ref_point = ref_point
        # print(self.state_space, self.action_space.n)

    def compute_q_set(self, s):
        q_set = []
        for a in range(self.action_space.n):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s, a]
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
        return np.array(q_set)

    def update_non_dominated(self, s, a, s_n):
        q_set_n = self.compute_q_set(s_n)
        # update for all actions, flatten
        solutions = np.concatenate(q_set_n, axis=0)
        # append to current pareto front
        # solutions = np.concatenate([solutions, self.non_dominated[s][a]])

        # compute pareto front
        self.non_dominated[s][a] = get_non_dominated(solutions)

    def act(self):
        if self.groupActing:
            # print(self.groupAction, self.state, self.action_space, self.groupRecommendation)
            if self.followGroup:
                self.followed = True
                self.action = self.groupAction
                # print("GROUP", self.action, self.groupAction)
            else:
                self.followed = False
                self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
                # print("GREEDY", self.action)
            self.groupRecommendation = max(self.groupRecommendation*self.decayGroup, self.minEpsilonGroup)
        else:
            q_set = self.compute_q_set(self.state)
            # print(q_set, self.state, self.action_space)
            self.action = self.exploration.choose(q_set, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        # if next_state not in self.q_table:
        #     self.q_table[next_state] = [random.uniform(0, 0) for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        # print(s, a, s1, self.action_space.n)

        self.update_non_dominated(s, a, s1)
        self.n_visits[s, a] += 1
        self.avg_r[s, a] += (reward - self.avg_r[s, a]) / self.n_visits[s, a]
        # self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
