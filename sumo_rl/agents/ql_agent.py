import numpy as np

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class QLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.groupAction = None
        self.groupActing = False
        self.epsilonGroup = 1
        self.decayGroup = 0.99
        self.minEpsilonGroup = 0.2
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        if self.groupActing:
            # print(self.groupAction, self.state, self.action_space, self.epsilonGroup)
            if np.random.rand() > self.epsilonGroup:
                self.action = self.groupAction
                # print("GROUP", self.action, self.groupAction)
            else:
                self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
                # print("GREEDY", self.action)
            self.epsilonGroup = max(self.epsilonGroup*self.decayGroup, self.minEpsilonGroup)
        else:
            self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        # print(s, a, s1, self.action_space.n)
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
