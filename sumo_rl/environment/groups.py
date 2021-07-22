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
import random
from gym import spaces
from statistics import mean
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedyGroups

class Groups:
    """
    This class represents a Group of Traffic Signals
    It is responsible for coordenate Traffc Signals that are in the group
    """

    def __init__(self, id, env, threshold = 0.1, alpha=0.1, gamma=0.95, exploration_strategy=EpsilonGreedyGroups()):
        self.id = id
        self.env = env
        self.setTLs = []
        self.state = []
        self.action = []
        self.actionToTLs = []
        self.intToAction = {}
        self.intToState = {}
        self.setNextStates = []
        self.setRewards = [[]]
        self.rewards = []
        self.action_space = 1
        self.qTable = {}
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.neighbours = []
        self.threshold = threshold
        self.done = False
        self.acc_reward = 0
        self.createdAt = 0
        self.performance = 0

    def act(self):
        try:
            if list(self.intToAction):
                # print("GROUP:", self, "\n i to a", self.intToAction, "\n a to i", self.actionToInt,"\n s to i", self.stateToInt,"\n s", self.state, "\n q", self.qTable)
                self.choice = self.exploration.choose(self.qTable, self.intToState[repr(self.state)], self.intToAction)
                if self.choice in self.intToAction.keys():
                    self.actionToTLs = self.intToAction[self.choice]
                else:
                    self.actionToTLs = self.intToAction[np.random.choice(list(self.intToAction.keys()))]
                    # self.actionToTLs = self.choice
                # print(self.choice, self.actionToTLs, self.action_space)
        except Exception as e:
            # print(self, self.intToAction, self.intToState, self.state, e)
            # print(self.intToAction, self.actionToInt, self.choice, self, self.stateToInt)
            exit()
        return self.actionToTLs

    def learn(self, done=False):
        # print("GROUP: ", self, "\n s", self.stateToInt, self.state, "\n a to i", self.actionToInt, "\n a", self.action, "\n r", self.setRewards[-1], "\n s1", self.setNextStates)
        # print("GROUP: ", self, "\n s", self.state, "\n a", self.action, "\n r", self.setRewards[-1], "\n s1", self.setNextStates, "\n q", self.qTable)
        # s = [rep for rep, state in self.intToState.items() if state == repr(self.state)][0]
        # s1 = [rep for rep, state in self.intToState.items() if state == repr(self.setNextStates)][0]
        # print(self.state, self.setNextStates, s, s1)
        s = self.intToState[repr(self.state)]
        s1 = self.intToState[repr(self.setNextStates)]
        a = [rep for rep, state in self.intToAction.items() if state == repr(self.action)][0]
        # print(self.action_space*10, self.intToAction, a)

        if s1 not in self.qTable:
            self.qTable[s1] = [random.uniform(0, 0) for _ in range(self.action_space)]

        if s not in self.qTable:
            self.qTable[s] = [random.uniform(0, 0) for _ in range(self.action_space)]

        self.state = copy.deepcopy(self.setNextStates)
        self.setNextStates = []

        rewardNormalized = []
        totalVehicles = 1
        if self.setRewards:
            for i, tl in enumerate(self.setTLs):
                if i+1 < len(self.setRewards[-1]):
                    vehicles = self.env.traffic_signals[tl].get_total_vehicles() + 1
                    rewardNormalized.append(self.setRewards[-1][i]*vehicles)
                    totalVehicles += vehicles
            # print(self.setRewards[-1], sum(rewardNormalized)/totalVehicles)
        self.rewards.append(sum(rewardNormalized)/totalVehicles)
        try:
            self.qTable[s][a] = self.qTable[s][a] + self.alpha*(self.rewards[-1] + self.gamma*max(self.qTable[s1]) - self.qTable[s][a])
        except Exception as e:
            print(s, a, s1, self.qTable, self.intToAction, self.printTLs)
            exit();
        self.acc_reward += self.rewards[-1]

        rewardPerformance = []
        for r in self.rewards[-10:]:
            rewardPerformance.append(r)

        self.rewards = self.rewards[-10:]
        # print(mean(rewardPerformance))
        self.performance = mean(rewardPerformance)

    def addGroup(self, TL):
        if self.env.traffic_signals[TL].inGroup == False:
            self.setTLs.append(TL)
            self.setTLs.sort()
            self.action_space *= self.env.traffic_signals[TL].action_space.n
            self.env.traffic_signals[TL].groupID = self.id
            for tl in self.env.neighbours[TL]:
                if tl not in self.neighbours and self.env.traffic_signals[tl].inGroup == False:
                    self.neighbours.append(tl)
            self.env.traffic_signals[TL].inGroup = True
        if not self.neighbours:
            self.done = True

    def addState(self, state):
        l = repr(state)
        s = len(self.intToState)
        # print("group state", len(self.state), l, self.intToState)
        if l not in self.intToState.keys():
            # self.stateToInt[s] = l
            self.intToState[l] = s

    def addAction(self, action):
        a = repr(action)
        l = len(self.intToAction)
        # print("action", a, self.intToAction.values())
        if a not in self.intToAction.values():
            # self.actionToInt[a] = l
            self.intToAction[l] = a

    def checkNeighbours(self):
        for neighbour in self.neighbours:
            if self.env.traffic_signals[neighbour].inGroup == True:
                self.neighbours.remove(neighbour)
        if not self.neighbours:
            self.done = True
        else:
            self.done = False

    def removingGroup(self):
        removed = []
        for tl in range(0, len(self.setTLs)):
            tlPerformance = []
            for r in self.setRewards[-10:]:
                tlPerformance.append(r[tl])
            gpPerformance = 0
            if self.performance > 0:
                gpPerformance = self.performance*self.threshold
            else:
                gpPerformance = self.performance*self.threshold + self.performance

            # print(mean(tlPerformance), self.performance, gpPerformance, self.setTLs)
            if self.threshold != 0:
                if mean(tlPerformance) < gpPerformance and abs(self.setRewards[-1][tl]) != 0:
                    removed.append(self.setTLs[tl])
                # print("MUITO RUIM!!!", self.setTLs[tl])
        return removed

    def printTLs(self):
        return (';'.join(self.setTLs))

    def printNeighbours(self):
        return (';'.join(self.neighbours))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.id) + " : " + self.printTLs() + " - " + self.printNeighbours()
