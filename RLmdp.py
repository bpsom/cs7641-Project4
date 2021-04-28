# Referred to -
# 1. RLDM course homework and projects for code snippets
# 2. https://medium.com/analytics-vidhya/solving-the-frozenlake-environment-from-openai-gym-using-value-iteration-5a078dffe438
#

import sys
import time
import random
import numpy as np
import gym
import matplotlib.pyplot as plt


class RLMDP(object):
    def __init__(self, env, gamma=0.9, eps=0.00001, alpha=0.7, max_episodes=1e6):
        self.gamma = gamma  # discount rate
        self.cv_factor = eps  # convergence factor
        self.alpha = alpha  # learning rate
        self.episodes = int(max_episodes)  # max iterations

        # Exploration / Exploitation Parameters
        self.epsilon = 1  # Exploration rate
        self.max_epsilon = 1  # Exploration probability at start
        self.min_epsilon = 0.01  # Minimum exploration probability
        self.decay_rate = 0.01  # Exponential decay rate for exploration prob
        self.train_episodes = 20000  # Total train episodes
        self.test_episodes = 1000  # Total test episodes
        self.max_steps = 100  # Max steps per episode

        print("env=", env)
        self.env_name = env
        self.env = gym.make(env)
        # if env is not 'Taxi-v3':
        if env == 'FrozenLake-v0':
            self.env = self.env.unwrapped
            self.desc = self.env.unwrapped.desc
            self.env.render()

        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
        # if env != 'CartPole-v0':
        #     self.state_size = self.env.observation_space.n
        # else:
        #     self.state_size = self.env.observation_space.shape[0]
        print("Action space size: ", self.action_size)
        print("State space size: ", self.state_size)
        self.Q = np.zeros((self.state_size, self.action_size))
        self.value = None
        self.policy = None

    def __del__(self):
        self.env.close()

    def close_env(self):
        self.env.close()

    def reinit_env(self):
        self.env = gym.make(self.env_name)
        if self.env_name == 'FrozenLake-v0':
            self.env = self.env.unwrapped
            self.desc = self.env.unwrapped.desc
            # self.env.render()

    def take_action(self, policy, discount_rate):
        observation = self.env.reset()
        total_reward = 0
        steps = 0
        misses=0
        while True:
            action = int(policy[observation])
            observation, reward, done, i = self.env.step(action)
            total_reward += (discount_rate ** steps * reward)
            steps += 1
            if done and reward == 1:
                break
            elif done and reward == 0:
                misses = 1
                break
        return total_reward, steps, misses

    def policy_evaluation(self, policy, gamma, n=100):
        scores = list()
        steps = list()
        misses = 0
        for i in range(n):
            score, step, miss = self.take_action(policy, gamma)
            scores.append(score)
            steps.append(step)
            misses += miss
        # scores = [self.take_action(policy, gamma) for i in range(n)]
        # avg_scores = np.mean(scores)
        avg_scores = np.mean(scores)
        print('----------------------------------------------')
        print('For Discount rate (Gamma) = {:.2f} Average score = {:.10f}'.format(gamma, avg_scores))
        print('Average of {:.0f} steps to reach goal'.format(np.mean(steps)))
        print('Fell in the hole {:.2f} % of the times'.format((misses / n * 100)))
        return avg_scores

    def policy_extraction(self, value, gamma):
        policy = np.zeros(self.env.nS)
        for s in range(self.env.nS):
            utility = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                utility[a] = sum([p * (r + gamma * value[new_state]) for p, new_state, r, d in self.env.P[s][a]])
            policy[s] = np.argmax(utility)
        return policy

    def policy_value_calculate(self, policy, discount_rate):
        value = np.zeros(self.env.nS)
        eps = 1e-5
        while True:
            prev_v = np.copy(value)
            for s in range(self.env.nS):
                p_action = policy[s]
                value[s] = sum([p * (r + discount_rate * prev_v[new_state])
                                for p, new_state, r, d in self.env.P[s][p_action]])
            if eps >= np.sum((np.fabs(prev_v - value))):
                break
        return value

    def bellman_update(self, prob, reward, discount_rate, p_val):
        utility = prob * (reward + discount_rate * p_val)
        return utility

    def value_iteration(self, discount_rate):
        value = np.zeros(self.env.nS)  # initialize value-function
        k = 0
        for i in range(self.episodes):
            prev_value = np.copy(value)
            for s in range(self.env.nS):
                utility = [sum([self.bellman_update(p, r, discount_rate, prev_value[new_state])
                           for p, new_state, r, d in self.env.P[s][a]])
                           for a in range(self.env.nA)]
                value[s] = max(utility)
            if self.cv_factor >= np.sum(np.fabs(prev_value - value)):
                k = i + 1
                break
        return value, k

    def policy_iteration(self, discount_rate):
        policy = np.random.choice(self.env.nA, size=self.env.nS)
        k = 0
        for i in range(self.episodes):
            prev_policy_value = self.policy_value_calculate(policy, discount_rate)
            new_policy = self.policy_extraction(prev_policy_value, discount_rate)
            if np.all(policy == new_policy):
                k = i + 1
                break
            policy = new_policy
        return policy, k

    # 1. Value iteration
    def mdp_value_iteration(self):
        discount_rate_gamma = list()
        scores = list()
        optimal_values = list()
        total_times = list()
        iterations = list()
        time_to_policy = list()
        for idx in range(0, 10):
            self.reinit_env()
            start = time.time()
            discount_rate = (idx + 0.5) / 10
            start1 = time.time()
            opt_value, k = self.value_iteration(discount_rate)
            policy = self.policy_extraction(opt_value, discount_rate)  # Make policy
            end1 = time.time()
            time_to_policy = end1 - start1
            policy_score = self.policy_evaluation(policy, discount_rate, n=1000)  # Test policy
            end = time.time()
            discount_rate_gamma.append(discount_rate)
            iterations.append(k)  # Iterations taken to converge
            optimal_values.append(opt_value)
            scores.append(policy_score)
            total_times.append(end - start)
            self.close_env()
            print("done-%d-VI", idx)
        return discount_rate_gamma, scores, optimal_values, total_times, iterations, time_to_policy

    # 2. Policy iteration
    def mdp_policy_iteration(self):
        discount_rate_gamma = list()
        scores = list()
        optimal_values = list()
        total_times = list()
        iterations = list()
        for idx in range(0, 10):
            self.reinit_env()
            start = time.time()
            discount_rate = (idx + 0.5) / 10
            start1 = time.time()
            opt_value, k = self.policy_iteration(discount_rate)  # Gives the Policy
            end1 = time.time()
            time_to_policy = end1 - start1
            policy_score = self.policy_evaluation(opt_value, discount_rate)  # Test policy
            end = time.time()
            discount_rate_gamma.append(discount_rate)
            iterations.append(k)
            optimal_values.append(opt_value)
            scores.append(np.mean(policy_score))
            total_times.append(end - start)
            self.close_env()
            print("done-%-PI", idx)
        return discount_rate_gamma, scores, optimal_values, total_times, iterations

    # 3. Q-learning
    def greedy_policy(self, state):
        return np.argmax(self.Q[state])

    def rewards_batch_average(self, rew, size):
        for i in range(0, len(rew), size):
            yield rew[i:i+size]

    def get_q_action(self, state, eps):
        if np.random.rand() < eps:
            action = np.argmax(self.Q[state, :])
        else:
            action = self.env.action_space.sample()
        return action

    def mdp_q_learning1(self, epsilons, gamma):
        start = time.time()
        rewards = list()  # list of rewards
        total_times = list()
        all_Q = list()
        iterations = list()
        averages = list()
        q_batch = list()
        sizes = list()
        alpha = 0.85
        for eps in epsilons:
            Q = np.zeros((self.state_size, self.action_size))
            optimal_values = list()
            nsteps = list()
            episode_reward = list()
            episodes = 30000
            self.reinit_env()
            for episode in range(episodes):
                state = self.env.reset()
                max_steps = 1000000
                done = False
                step_reward = 0
                ki = 0
                for k in range(max_steps):
                    if done:
                        ki = k
                        break
                    action = self.get_q_action(state, eps)
                    new_state, reward, done, info = self.env.step(action)
                    step_reward = step_reward + reward
                    Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
                eps = (1 - 2.71 ** (-episode / 1000))  # euler's number e=2.71
                episode_reward.append(step_reward)
                nsteps.append(ki)
            all_Q.append(Q)
            rewards.append(episode_reward)
            iterations.append(nsteps)

            for k in range(self.state_size):
                optimal_values.append(np.argmax(Q[k, :]))

            self.close_env()
            end = time.time()
            total_times.append(end - start)

            def rewards_batch_average(rew, n):
                for p in range(0, len(rew), n):
                    yield rew[p:p + n]

            size = int(episodes / 50)
            batches = list(rewards_batch_average(episode_reward, size))
            q_batch.append(batches)
            avg = list()
            for batch in batches:
                avg.append(np.sum(batch) / len(batch))
            # avg = [np.sum(batch) / len(batch) for batch in batches]
            averages.append(avg)
            sizes.append(size)
            print("done-%-epsilon", eps)
        return rewards, iterations, total_times, averages, sizes

    def test_q_learning(self):
        # TEST PHASE
        alpha = 0.85
        gamma = 0.90
        for episode in range(1000):
            state = self.env.reset()
            episode_rewards = 0
            n_steps = 1000
            for t in range(n_steps):
                action = self.greedy_policy(state)
                new_state, reward, done, _ = self.env.step(action)
                self.Q[state, action] += alpha * (reward + gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
                state = new_state
                episode_rewards += reward
                # at the end of the episode
                if done:
                    print('Test episode finished with a total reward of: {}'.format(episode_rewards))
                    break
        self.env.close()
