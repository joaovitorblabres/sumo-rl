import numpy as np
from gym import spaces
import random
import copy

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
        points = np.asarray(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

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
    # solutions = copy.deepcopy(solution)
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    # print(solutions)
    # nonA = []
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(c > solutions[is_efficient], axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
            # nonA.append(i)

    return solutions[is_efficient]
    # return nonA

class MOSelection:
    '''
        algType:Int - {0 = Hypervolume, 1 = Pareto Selection, others = Random, default: 0}
    '''
    def __init__(self, initial_epsilon=0.05, min_epsilon=0.05, decay=1, ref=np.asarray([100, -100]), algType=0):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.ref = ref
        self.lenAct = 0
        self.actRnd = 0
        self.algType = algType

    def choose(self, q_set, state, action_space):
        if np.random.rand() >= self.epsilon:
            if self.algType == 0:
                qSet = copy.deepcopy(q_set)
                q_values = compute_hypervolume(qSet, action_space.n, self.ref)
                # print(q_values)
            elif self.algType == 1:
                solutions = np.concatenate(q_set, axis=0)
                p = paretoEfficient(solutions, False, False, False)
                nonD = solutions[p]
                actions = []
                for s, q in enumerate(q_set):
                    if len(actions) == action_space.n:
                        break
                    for n in nonD:
                        if len(actions) == action_space.n:
                            break
                        for v in q:
                            # print(v, n, q, s)
                            if np.array_equal(v, n):
                                if s not in actions:
                                    actions.append(s)
                            if s in actions:
                                break
                        if s in actions:
                            break

                self.lenAct = len(actions)
                self.actRnd = 0
                if self.lenAct == q_set.shape[0]:
                    self.actRnd = 1

            # print(actions)
            self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
            if self.algType == 0:
                return np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())
            elif self.algType == 1:
                try:
                    return np.random.choice(actions)
                except:
                    return np.random.choice(range(action_space.n))
            else:
                return np.random.choice(range(action_space.n))
        else:
            self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
            # print(q_set, q_values, nonD)
            self.lenAct = q_set.shape[0]
            return np.random.choice(range(q_set.shape[0]))

        #print(self.epsilon)
        return np.random.choice(range(q_set.shape[0]))

    def reset(self):
        self.epsilon = self.initial_epsilon
