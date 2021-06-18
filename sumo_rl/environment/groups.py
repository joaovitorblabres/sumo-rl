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
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedyGroups

class Groups:
    """
    This class represents a Group of Traffic Signals
    It is responsible for coordenate Traffc Signals that are in the group
    """

    def __init__(self, id, env, threshold, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedyGroups()):
        self.id = id
        self.env = env
        self.setTLs = []
        self.state = []
        self.action = []
        self.actionToTLs = []
        self.actionToInt = {}
        self.intToAction = {}
        self.stateToInt = {}
        self.intToState = {}
        self.setNextStates = []
        self.setRewards = [[]]
        self.action_space = 1
        self.qTable = {}
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.neighbours = []
        self.threshold = threshold
        self.done = False
        self.acc_reward = 0

    def act(self):
        if list(self.actionToInt):
            self.actionToTLs = self.intToAction[self.exploration.choose(self.qTable, self.stateToInt[repr(self.state)], self.actionToInt)]
            return self.actionToTLs

    def learn(self, done=False):
        # print("GROUP: ", self, "\n s", self.stateToInt, self.state, "\n a", self.actionToInt, "\n r", self.setRewards, "\n s1", self.setNextStates)
        s = self.stateToInt[repr(self.state)]
        s1 = self.stateToInt[repr(self.setNextStates)]
        a = self.actionToInt[repr(self.action)]

        if s1 not in self.qTable:
            self.qTable[s1] = [0 for _ in range(self.action_space)]

        self.state = copy.deepcopy(self.setNextStates)
        self.setNextStates = []

        self.qTable[s][a] = self.qTable[s][a] + self.alpha*(sum(self.setRewards[-1]) + self.gamma*max(self.qTable[s1]) - self.qTable[s][a])
        self.acc_reward += sum(self.setRewards[-1])

    def addGroup(self, TL):
        if self.env.traffic_signals[TL].inGroup == False:
            self.setTLs.append(TL)
            self.action_space *= self.env.traffic_signals[TL].action_space.n
            self.env.traffic_signals[TL].groupID = self.id
            for tl in self.env.neighbours[TL]:
                if tl not in self.neighbours and self.env.traffic_signals[tl].inGroup == False:
                    self.neighbours.append(tl)
            self.env.traffic_signals[TL].inGroup = True

    def addState(self, state):
        s = repr(state)
        l = len(self.stateToInt)
        if s not in self.stateToInt:
            self.stateToInt[s] = l
            self.intToState[l] = s
        # print("group state", self.stateToInt, len(self.state), s, self.intToState)

    def addAction(self, action):
        a = repr(action)
        l = len(self.actionToInt)
        # print("action state", a)
        if a not in self.actionToInt:
            self.actionToInt[a] = l
            self.intToAction[l] = a

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
