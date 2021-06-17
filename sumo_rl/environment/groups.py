import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
import copy
from gym import spaces
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

class Groups:
    """
    This class represents a Group of Traffic Signals
    It is responsible for coordenate Traffc Signals that are in the group
    """

    def __init__(self, id, env, threshold, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        self.id = id
        self.env = env
        self.setTLs = []
        self.setActions = [[]]
        self.state = []
        self.setStates = {}
        self.setNextStates = []
        self.setRewards = [[]]
        self.action_space = []
        self.action = None
        self.qTable = {}
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.neighbours = []
        self.threshold = threshold
        self.done = False
        self.acc_reward = 0

    def act(self):
        self.action = self.exploration.choose(self.qTable, self.state, self.setActions)
        return self.action

    def learn(self, done=False):
        print("GROUP: ", self, "\n s", self.setStates, "\n a", self.setActions, "\n r", self.setRewards, "\n n", self.setNextStates)
        s = self.setStates[repr(self.state)]
        s1 = self.setStates[repr(self.setNextStates)]

        if s1 not in self.qTable:
            self.qTable[s1] = [0 for _ in range(len(self.setActions))]

        if self.action is None:
            self.action = repr(self.setActions[-1])

        a = self.action
        print(s, a, s1)
        self.state = copy.deepcopy(s1)
        self.setNextStates = []

        self.qTable[s][a] = self.qTable[s][a] + self.alpha*(self.setRewards[-1] + self.gamma*max(self.qTable[s1]) - self.qTable[s][a])
        self.acc_reward += self.setRewards[-1]

    def addGroup(self, TL):
        if self.env.traffic_signals[TL].inGroup == False:
            self.setTLs.append(TL)
            self.env.traffic_signals[TL].groupID = self.id
            for tl in self.env.neighbours[TL]:
                if tl not in self.neighbours and self.env.traffic_signals[tl].inGroup == False:
                    self.neighbours.append(tl)
            self.env.traffic_signals[TL].inGroup = True

    def addState(self, state):
        if repr(state) not in self.setStates:
            self.setStates[repr(state)] = len(self.setStates)
        print("group state", self.setStates, len(self.state), state)

    def checkNeighbours(self):
        for neighbour in self.neighbours:
            if self.env.traffic_signals[neighbour].inGroup == True:
                self.neighbours.remove(neighbour)
                if not self.neighbours:
                    self.done = True


    def removingGroup(self):
        last = self.setRewards[-10:]
        avg = np.average(last)
        for tl in self.setTLs:
            if avg < self.threshold:
                pass

    def printTLs(self):
        return (';'.join(self.setTLs))

    def printNeighbours(self):
        return (';'.join(self.neighbours))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.id) + " : " + self.printTLs() + " - " + self.printNeighbours()
