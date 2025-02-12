import numpy as np
import random

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class QLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.95, exploration_strategy=EpsilonGreedy(), groupRecommendation=0.2):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.groupAction = None
        self.groupActing = False
        self.groupRecommendation = groupRecommendation
        self.decayGroup = 1
        self.minEpsilonGroup = 0.05
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0
        self.followed = False

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
            self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [random.uniform(0, 0) for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        # print(s, a, s1, self.action_space.n)
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward[0] + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward[0]
