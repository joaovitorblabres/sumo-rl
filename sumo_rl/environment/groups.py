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
        self.actionToInt = {}
        self.intToAction = {}
        self.stateToInt = {}
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
            if list(self.actionToInt):
                # print("GROUP:", self, "\n i to a", self.intToAction, "\n a to i", self.actionToInt,"\n s to i", self.stateToInt,"\n s", self.state, "\n q", self.qTable)
                self.choice = self.exploration.choose(self.qTable, [rep for rep, state in self.intToState.items() if state == repr(self.state)][0], self.actionToInt)
                if self.choice < len(self.intToAction.keys()):
                    self.actionToTLs = self.intToAction[self.choice]
                else:
                    self.actionToTLs = self.intToAction[0]
        except Exception as e:
            print(self, self.intToAction, self.actionToInt, self.stateToInt, self.state, e)
            # print(self.intToAction, self.actionToInt, self.choice, self, self.stateToInt)
            exit()
        return self.actionToTLs

    def learn(self, done=False):
        # print("GROUP: ", self, "\n s", self.stateToInt, self.state, "\n a to i", self.actionToInt, "\n a", self.action, "\n r", self.setRewards[-1], "\n s1", self.setNextStates)
        # print("GROUP: ", self, "\n s", self.state, "\n a", self.action, "\n r", self.setRewards[-1], "\n s1", self.setNextStates, "\n q", self.qTable)
        s = [rep for rep, state in self.intToState.items() if state == repr(self.state)][0]
        s1 = [rep for rep, state in self.intToState.items() if state == repr(self.setNextStates)][0]
        # print(self.state, self.setNextStates, s, s1)
        # s = self.stateToInt[repr(self.state)]
        # s1 = self.stateToInt[repr(self.setNextStates)]
        a = self.actionToInt[repr(self.action)]
        # print(self.action_space)

        if s1 not in self.qTable:
            self.qTable[s1] = [random.uniform(-1, 1) for _ in range(self.action_space*10)]

        if s not in self.qTable:
            self.qTable[s] = [random.uniform(-1, 1) for _ in range(self.action_space*10)]

        self.state = copy.deepcopy(self.setNextStates)
        self.setNextStates = []

        rewardNormalized = []
        totalVehicles = 1
        for i, tl in enumerate(self.setTLs):
            vehicles = self.env.traffic_signals[tl].get_total_vehicles() + 1
            rewardNormalized.append(abs(self.setRewards[-1][i])*vehicles)
            totalVehicles += vehicles
        # print(self.setRewards[-1], sum(rewardNormalized)/totalVehicles)
        self.rewards.append(sum(rewardNormalized)/totalVehicles)

        self.qTable[s][a] = self.qTable[s][a] + self.alpha*(self.rewards[-1] + self.gamma*max(self.qTable[s1]) - self.qTable[s][a])
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
            self.action_space *= self.env.traffic_signals[TL].action_space.n
            self.env.traffic_signals[TL].groupID = self.id
            for tl in self.env.neighbours[TL]:
                if tl not in self.neighbours and self.env.traffic_signals[tl].inGroup == False:
                    self.neighbours.append(tl)
            self.env.traffic_signals[TL].inGroup = True

    def addState(self, state):
        s = repr(state)
        l = len(self.intToState)
        if l not in self.intToState:
            # self.stateToInt[s] = l
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
        else:
            self.done = False

    def removingGroup(self):
        removed = []
        for tl in range(0, len(self.setTLs)):
            tlPerformance = []
            # print(self.setRewards[-10:], self.setTLs)
            for r in self.setRewards[-10:]:
                tlPerformance.append(abs(r[tl]))
            if mean(tlPerformance) < self.performance*self.threshold and abs(self.setRewards[-1][tl]) != 0:
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
