# Referred to -
# 1. RLDM course homework and projects for code snippets
# 2. https://github.com/sawcordwell/pymdptoolbox/blob/master/src/mdptoolbox/mdp.py#L1091-L1095
#

import sys
import time
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import mdptoolbox
from mdptoolbox.mdp import ValueIteration, QLearning, PolicyIteration
from mdptoolbox.example import forest


class RLMDPToolBox(object):
    def __init__(self, states):
        self.states = states
        self.gamma = 0.90
        self.select_states = [3, 4, 5, 6, 7, 8, 9, 10,
                              11, 12, 13, 14, 15, 20,
                              30, 50, 70, 100, 200, 300,
                              400, 500, 600, 700, 800, 900, 1000]

    def mtb_value_iteration(self):
        # P, R = forest(S=self.states, p=0.01)
        iteration = list()
        policy = list()
        value = list()
        total_times = list()
        discount_rate_gamma = list()
        discount_rates = [(i + 0.5) / 10 for i in range(0, 10)]
        for i in range(0, 10):
            P, R = forest(S=self.states, p=0.01)  # Probability p=0.01 that a fire burns the forest
            discount_rate_gamma.append(discount_rates[i])
            start = time.time()
            pi = ValueIteration(P, R, discount_rates[i])
            pi.run()
            end = time.time()
            value.append(np.mean(pi.V))
            policy.append(pi.policy)
            iteration.append(pi.iter)
            total_times.append(end - start)
            print("done-%d-VI", i)
        return discount_rates, value, iteration, total_times

    def mtb_policy_iteration(self):
        # P, R = forest(S=self.states, p=0.01)
        iteration = list()
        policy = list()
        value = list()
        total_times = list()
        discount_rate_gamma = list()
        discount_rates = [(i + 0.5) / 10 for i in range(0, 10)]
        for i in range(0, 10):
            P, R = forest(S=self.states, p=0.01)  # Probability p=0.01 that a fire burns the forest
            discount_rate_gamma.append(discount_rates[i])
            start = time.time()
            pi = PolicyIteration(P, R, discount_rates[i])
            pi.run()
            end = time.time()
            value.append(np.mean(pi.V))
            policy.append(pi.policy)
            iteration.append(pi.iter)
            total_times.append(end - start)
            print("done-%d-PI", i)
        return discount_rates, value, iteration, total_times

    def mtb_q_learning(self):
        iteration = list()
        policy = list()
        value = list()
        total_times = list()
        discount_rate_gamma = list()
        discount_rates = [(i + 0.5) / 10 for i in range(0, 10)]
        for i in range(0, 10):
            P, R = forest(S=self.states, p=0.01)  # Probability p=0.01 that a fire burns the forest
            discount_rate_gamma.append(discount_rates[i])
            start = time.time()
            pi = QLearning(P, R, discount=discount_rates[i])
            pi.run()
            end = time.time()
            value.append(np.mean(pi.V))
            policy.append(pi.policy)
            iteration.append(10000) # Max_iterations = 10000 by default
            total_times.append(end - start)
            print("done-%d-QI", i)
        return discount_rates, value, iteration, total_times

    def compare_vpq_learning(self):
        compare_VI_QI_policy = []  # True or False
        compare_VI_PI_policy = []

        for state in self.select_states:
            P, R = forest(state, p=0.01)
            VI = ValueIteration(P, R, self.gamma)
            PI = PolicyIteration(P, R, self.gamma)
            QL = QLearning(P, R, self.gamma)
            VI.run()
            PI.run()
            QL.run()
            compare_VI_QI_policy.append(QL.policy == VI.policy)
            compare_VI_PI_policy.append(VI.policy == PI.policy)
        print("Forest Management - VI and PI Comparison", compare_VI_PI_policy)
        print("Forest Management - VI and PI Comparison", compare_VI_QI_policy)
