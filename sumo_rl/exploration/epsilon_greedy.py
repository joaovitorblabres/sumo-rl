import numpy as np
from gym import spaces
import random

class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon

class EpsilonGreedyGroups:

    def __init__(self, initial_epsilon=0.05, min_epsilon=0.05, decay=1):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def choose(self, q_table, state, action_space):
        # print("choose", action_space)
        if np.random.rand() < self.epsilon:
            action = random.choice(list(action_space.values()))
        else:
            action = np.argmax(q_table[state])
            # print("choose", action_space, q_table[state])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        #print(self.epsilon)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon

#----------------------
from pygmo import hypervolume

def compute_hypervolume(q_set, nA, ref):
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        # print(q_set, i)
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

class MOSelection:
    def __init__(self, initial_epsilon=0.05, min_epsilon=0.05, decay=1, ref=np.array([-100, -100])):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.ref = ref

    def choose(self, q_set, state, action_space):
        q_values = compute_hypervolume(q_set, action_space.n, self.ref)

        if np.random.rand() >= self.epsilon:
            self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
            return np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())
        else:
            self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
            return np.random.choice(range(q_set.shape[0]))

        #print(self.epsilon)
        return np.random.choice(range(q_set.shape[0]))

    def reset(self):
        self.epsilon = self.initial_epsilon
