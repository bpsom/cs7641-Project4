# Script that runs the experiment
import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mdptoolbox
import mdptoolbox.example

import gym
from gym import envs
from gym.envs.toy_text import discrete
from RLmdp import RLMDP
import RLmdptoolbox


def plot_rl(dirname, fp, env_name, gamma, scores, values, times, episodes, states):
    plot_name = '{}-Convergence Analysis'.format(env_name)
    plt.plot(gamma, episodes)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Iterations to Converge')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-Optimal State-Value,gamma,convergence'.format(env_name)
    k = 0
    for gam in gamma:
        plt.plot(states, values[k], label='{}'.format(gam))
        k += 1
    # plt.plot(gamma, values)
    plt.title(plot_name)
    plt.xlabel('states')
    plt.ylabel('Optimal State Value')
    plt.legend()
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-Reward Analysis'.format(env_name)
    plt.plot(gamma, scores)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-Execution Time Analysis'.format(env_name)
    plt.plot(gamma, times)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Execution Time (sec)')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()


def plot_ql(dirname, fp, env_name, epsilons, rewards, iterations, total_times, averages, sizes, gamma):
    plot_name = '{}-{}- reward Analysis'.format(env_name, gamma)
    plt.plot(iterations, rewards)
    plt.title(plot_name)
    plt.xlabel('episode')
    plt.ylabel('Training cumulative reward')
    # plt.legend()
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-{}- time Analysis'.format(env_name, gamma)
    plt.plot(epsilons, total_times)
    plt.title(plot_name)
    plt.xlabel('greedy epsilons')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()


def plot_ql_gamma(dirname, fp, env_name, epsilons, rewards, iterations, total_times, averages, sizes, gamma):
    plot_name = '{}- rewards for learning rate'.format(env_name)
    x0 = range(0, len(rewards[0]), sizes[0])
    x1 = range(0, len(rewards[1]), sizes[1])
    x2 = range(0, len(rewards[2]), sizes[2])
    x3 = range(0, len(rewards[3]), sizes[3])
    x4 = range(0, len(rewards[4]), sizes[4])
    x5 = range(0, len(rewards[5]), sizes[5])
    plt.plot(x0, averages[0], label='greedy-epsilon=0.05')
    plt.plot(x1, averages[1], label='greedy-epsilon=0.15')
    plt.plot(x2, averages[2], label='greedy-epsilon=0.30')
    plt.plot(x3, averages[3], label='greedy-epsilon=0.50')
    plt.plot(x4, averages[4], label='greedy-epsilon=0.75')
    plt.plot(x5, averages[5], label='greedy-epsilon=0.90')
    plt.title(plot_name)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()


def mdp_frozen_lake(dirname, fp, episodes):  # Grid MDP
    alphas = [0.05, 0.15, 0.30, 0.5, 0.75, 0.90]
    states = [i for i in range(0,16)]
    convergence_factor = 1e-20
    frozen_lake = RLMDP(env='FrozenLake-v0',
                        gamma=0.9,
                        eps=convergence_factor,
                        max_episodes=episodes)
    # Value Iteration
    discount_rate_gamma, scores, optimal_values, total_times, iterations, time_to_policy = frozen_lake.mdp_value_iteration()
    plot_rl(dirname,
            fname,
            'Frozen Lake (Value Iteration)',
            discount_rate_gamma, scores, optimal_values, total_times, iterations, states)

    # Policy Iteration
    discount_rate_gamma, scores, optimal_values, total_times, iterations = frozen_lake.mdp_policy_iteration()
    plot_rl(dirname,
            fname,
            'Frozen Lake (Policy Iteration)',
            discount_rate_gamma, scores, optimal_values, total_times, iterations, states)
    # Q-learning 1
    # Training
    epsilons = [0.05, 0.15, 0.30, 0.5, 0.75, 0.90]
    gammas = [0.75, 0.80, 0.85, 0.90]
    for gamma in gammas:
        rewards, iterations, total_times, averages, sizes = frozen_lake.mdp_q_learning1(epsilons, gamma)
        plot_ql(dirname,
                fname,
                'Frozen Lake (Q learning)-training',
                epsilons, rewards, iterations, total_times, averages, sizes, gamma)
        if gamma == 0.90:
            plot_ql_gamma(dirname,
                          fname,
                          'Frozen Lake (Q learning)-training',
                          epsilons, rewards, iterations, total_times, averages, sizes, gamma)
    # Testing
    frozen_lake.test_q_learning()
    print("done-All-FrozenLake")


def plot_forest_management(dirname, fp, env_name, gamma, value, episodes, times):
    plot_name = '{}-Convergence Analysis'.format(env_name)
    plt.plot(gamma, episodes)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Iterations to Converge')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-Reward Analysis'.format(env_name)
    plt.plot(gamma, value)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Average Rewards')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()

    plot_name = '{}-Execution Time Analysis'.format(env_name)
    plt.plot(gamma, times)
    plt.title(plot_name)
    plt.xlabel('Discount rate')
    plt.ylabel('Execution Time (sec)')
    plt.grid()
    plt.savefig(dirname + plot_name + '.png')
    # plt.show()
    plt.close()


def mdp_forest_management(dirname, fp):
    forest = RLmdptoolbox.RLMDPToolBox(states=1000)
    discount_rates, value, iteration, times = forest.mtb_value_iteration()
    plot_forest_management(dirname, fp, 'ForestManagement-ValueIteration',
                           discount_rates, value, iteration, times)
    discount_rates, value, iteration, times = forest.mtb_policy_iteration()
    plot_forest_management(dirname, fp, 'ForestManagement-PolicyIteration',
                           discount_rates, value, iteration, times)

    discount_rates, value, iteration, times = forest.mtb_q_learning()
    plot_forest_management(dirname, fp, 'ForestManagement-Q-Learning',
                           discount_rates, value, iteration, times)

    forest.compare_vpq_learning()
    return 0


if __name__ == '__main__':
    print("--- Reinforcement Learning ---" + "\n" + "\n")
    # sns.set()

    folder = "result"
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        fname = './{}/Results.txt'.format(folder)
        os.remove(fname)
    except OSError:
        pass

    dirname = './{}/'.format(folder)
    fname = './{}/Results.txt'.format(folder)
    fp = open(fname, "a+")
    fp.write("--- Reinforcement Learning ---" + "\n" + "\n")

    Episodes = 1e6
    mdp_frozen_lake(dirname, fp, Episodes)
    mdp_forest_management(dirname, fp)

    print("done")
    fp.close()

