import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces

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
        self.setStates = [[]]
        self.setRewards = [[]]
        self.action_space = []
        self.action = None
        self.qTable = None
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.neighbours = []
        self.threshold = threshold
        self.done = False

    def act(self):
        self.action = self.exploration.choose(self.q_table, self.state, self.setActions)
        return self.action

    def addGroup(self, TL):
        if self.env.traffic_signals[TL].inGroup == False:
            self.setTLs.append(TL)
            self.env.traffic_signals[TL].groupID = self.id
            for tl in self.env.neighbours[TL]:
                if tl not in self.neighbours and self.env.traffic_signals[tl].inGroup == False:
                    self.neighbours.append(tl)
            self.env.traffic_signals[TL].inGroup = True

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
