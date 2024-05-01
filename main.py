# Sasha Koroleva
# 04/30/2024
# Using Radial Basis Functions (RBF) to approximate the state space

import numpy as np
import matplotlib.pyplot as plt
import random
import math



# FUNCTIONS FOR NORMAL QL
def choose_action_QL_normal(s, Q_table, epsilon, p_array, iteration, episode):
    # select the action with highest estimated action value
    # if several actions with same Q value: randomize
    # get maximum value
    max_action_value = max(Q_table[s].values())

    # get all keys with maximum value
    max_action_keys = [key for key, value in Q_table[s].items() if value == max_action_value]

    # decaying epsilon. Higher epsilon at start of training for more exploration. In later episodes uses exploitation.
    if episode == 0 or episode == 1:
        epsilon = 0.5
    elif episode < 10:
        epsilon = 0.01
    else:
        epsilon = 0.0

    p = p_array[iteration]
    if p >= epsilon:
        # if more than one maximum value action found
        if len(max_action_keys) > 1:
            # randomize
            action = random.choice(max_action_keys)
        else:
            action = max_action_keys[0]
    else:
        # randomize
        action = random.choice(list(Q_table[s]))
    return action


def take_action_QL_normal(action, s):
    # set new state s'
    if action == 'up':
        sprime = (s[0] - 1, s[1])
    elif action == 'down':
        sprime = (s[0] + 1, s[1])
    elif action == 'left':
        sprime = (s[0], s[1] - 1)
    elif action == 'right':
        sprime = (s[0], s[1] + 1)
    else:
        raise ValueError("Invalid action")

    # set reward r
    # • Action that makes the robot tend to go out of the grid will get a reward of -1 (when the robot is in the border cells)
    # • Action that makes the robot reach the goal will get a reward of 100
    # • All other actions will get a reward of 0
    if sprime == (4, 4):
        r = 100
    elif sprime[0] == -1 or sprime[1] == -1 or sprime[0] == 5 or sprime[1] == 5:
        r = -1
    else:
        r = 0

    # if action would be out of the border, robot stays in current cell
    if r == -1:
        sprime = s
    return r, sprime


def QL_normal(gamma, alpha, epsilon, p_array, s, Q_table, episode):
    reward_sum_episode_QL_normal = 0
    action_sum_episode_QL_normal = 0
    iteration = 0
    iteration_terminate = 0
    """ Repeat (for each step of episode): """
    # while loop until goal is reached
    while s != (4, 4) and iteration_terminate < 10000:
        """ Choose a from s using policy derived from Q (e.g. epsilon-greedy) """
        action = choose_action_QL_normal(s, Q_table, epsilon, p_array, iteration, episode)
        """ Take action a, observe r, s' """
        r, sprime = take_action_QL_normal(action, s)
        """ Q[s,action] += alpha * (reward + (gamma * predicted_value) - Q[s,action]) """
        predicted_value = max(Q_table[sprime].values(), default=0)
        Q_table[s][action] += alpha * (r + (gamma * predicted_value) - Q_table[s][action])
        action_sum_episode_QL_normal += 1
        reward_sum_episode_QL_normal += r
        if iteration < 199:
            iteration += 1
        else:
            iteration = 0
        iteration_terminate += 1
        """ s = s' """
        s = sprime
    """ Until s is terminal """
    return action_sum_episode_QL_normal, reward_sum_episode_QL_normal


# FUNCTIONS FOR RBF-QL WITH 4RBF

def choose_action_4RBF(s, theta, epsilon, p_array, c, mu, iteration, episode):
    # select the action with highest estimated action value
    # if several actions with same value: randomize
    # calculate phi for all actions
    phi_all_actions = []
    # for all 4 actions
    for i in range(4):
        phi = np.zeros(16)
        for l in range(4):
            phi[i*4+l] = math.exp(-np.linalg.norm(s - c[l,:])**2 / (2 * mu[l]**2))
        phi_all_actions.append(phi)
    # calculate phi_transp*theta for each action
    phi_t_mult_theta_list = [phi @ theta for phi in phi_all_actions]
    max_action_keys = [jj for jj, j in enumerate(phi_t_mult_theta_list) if j == max(phi_t_mult_theta_list)]
    # decaying epsilon. Higher epsilon at start of training for more exploration. In later episodes uses exploitation.
    if episode == 0 or episode == 1:
        epsilon = 0.5
    elif episode < 10:
        epsilon = 0.01
    else:
        epsilon = 0.0
    p = p_array[iteration]
    if p >= epsilon:
        # if more than one maximum value action found
        if len(max_action_keys) > 1:
            # randomize
            action = random.choice(max_action_keys)
        else:
            action = max_action_keys[0]
    else:
        # randomize
        action = random.randint(0, 3)
    return action


def take_action_4RBF(action, s):
    # set new state s'
    if action == 0:
        sprime = (s[0]-1, s[1])
    elif action == 1:
        sprime = (s[0]+1, s[1])
    elif action == 2:
        sprime = (s[0], s[1]-1)
    elif action == 3:
        sprime = (s[0], s[1]+1)
    else:
        raise ValueError("Invalid action")
    # set reward r
    # • Action that makes the robot tend to go out of the grid will get a reward of -1 (when the robot is in the border cells)
    # • Action that makes the robot reach the goal will get a reward of 100
    # • All other actions will get a reward of 0
    if sprime == (4,4):
        r = 100
    elif sprime[0] == -1 or sprime[1] == -1 or sprime[0] == 5 or sprime[1] == 5:
        r = -1
    else:
        r = 0
    # if action would be out of the border, robot stays in current cell
    if r == -1:
        sprime = s
    return r, sprime


def QL_4RBF(gamma, alpha, epsilon, p_array, s, c, mu, theta, episode):
    reward_sum_episode_4RBF = 0
    action_sum_episode_4RBF = 0
    iteration = 0
    iteration_terminate = 0
    """ Repeat (for each step of episode): """
    # while loop until goal is reached
    while s != (4,4) and iteration_terminate < 10000:
        """ Choose a from A using greedy policy with probability p """
        action = choose_action_4RBF(s, theta, epsilon, p_array, c, mu, iteration, episode)
        """ Take action a, observe r,s' """
        r, sprime = take_action_4RBF(action, s)
        """ Estimate phi_s """
        phi_s = np.zeros(16)
        for i in range(4):
            if action == i:
                for l in range(4):
                    phi_s[i*4+l] = math.exp( - (np.linalg.norm( s - c[l,:])**2 ) /(2*(mu[l]**2)) )
        """ Update """
        # calculate predicted value
        # calculate phi_sprime for all actions
        phi_sprime_all_actions = []
        # for all 4 actions
        for i in range(4):
            phi_sprime = np.zeros(16)
            for l in range(4):
                phi_sprime[i*4+l] = math.exp( - (np.linalg.norm( sprime - c[l,:])**2 ) /(2*(mu[l]**2)) )
            phi_sprime_all_actions.append(phi_sprime)
        # calculate phi_sprime_transp*theta for each action
        phi_sprime_t_mult_theta_list = [phi_sprime @ theta for phi_sprime in phi_sprime_all_actions]
        predicted_value = max(phi_sprime_t_mult_theta_list)
        # theta update
        theta += alpha * (r + gamma * predicted_value - phi_s @ theta) * phi_s
        action_sum_episode_4RBF += 1
        reward_sum_episode_4RBF += r
        if iteration < 199:
            iteration += 1
        else:
            iteration = 0
        iteration_terminate += 1
        """ s = s' """
        s = sprime
        """ Until s is terminal """
    return action_sum_episode_4RBF, reward_sum_episode_4RBF


# FUNCTIONS FOR RBF-QL WITH 9RBF
def choose_action_9RBF(s, theta, epsilon, p_array, c, mu, iteration, episode):
    phi_all_actions = []
    # for all 4 actions
    for i in range(4):
        phi = np.zeros(36)  # Adjusted for 9 RBFs per action
        for l in range(9):  # Adjusted for 9 RBFs
            phi[i * 9 + l] = math.exp(-np.linalg.norm(np.array(s) - c[l, :]) ** 2 / (2 * mu[l] ** 2))
        phi_all_actions.append(phi)

    phi_t_mult_theta_list = [phi @ theta for phi in phi_all_actions]
    max_action_keys = [jj for jj, j in enumerate(phi_t_mult_theta_list) if j == max(phi_t_mult_theta_list)]

    if episode == 0 or episode == 1:
        epsilon = 0.5
    elif episode < 10:
        epsilon = 0.01
    else:
        epsilon = 0.0

    p = p_array[iteration]
    if p >= epsilon:
        if len(max_action_keys) > 1:
            action = random.choice(max_action_keys)
        else:
            action = max_action_keys[0]
    else:
        action = random.randint(0, 3)
    return action


def take_action_9RBF(action, s):
    # Actions are defined as 0: up, 1: down, 2: left, 3: right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sprime = (s[0] + moves[action][0], s[1] + moves[action][1])

    if sprime == (4, 4):
        r = 100
    elif sprime[0] < 0 or sprime[1] < 0 or sprime[0] > 4 or sprime[1] > 4:
        r = -1
    else:
        r = 0

    if r == -1:
        sprime = s
    return r, sprime


def QL_9RBF(gamma, alpha, epsilon, p_array, s, c, mu, theta, episode):
    reward_sum_episode_9RBF = 0
    action_sum_episode_9RBF = 0
    iteration = 0
    iteration_terminate = 0

    while s != (4, 4) and iteration_terminate < 10000:
        action = choose_action_9RBF(s, theta, epsilon, p_array, c, mu, iteration, episode)
        r, sprime = take_action_9RBF(action, s)

        phi_s = np.zeros(36)  # Adjusted for 9 RBFs per action
        for i in range(4):
            if action == i:
                for l in range(9):  # Adjusted for 9 RBFs
                    phi_s[i * 9 + l] = math.exp(-np.linalg.norm(np.array(s) - c[l, :]) ** 2 / (2 * mu[l] ** 2))

        phi_sprime_all_actions = []
        for i in range(4):
            phi_sprime = np.zeros(36)  # Adjusted for 9 RBFs per action
            for l in range(9):
                phi_sprime[i * 9 + l] = math.exp(-np.linalg.norm(np.array(sprime) - c[l, :]) ** 2 / (2 * mu[l] ** 2))
            phi_sprime_all_actions.append(phi_sprime)

        phi_sprime_t_mult_theta_list = [phi_sprime @ theta for phi_sprime in phi_sprime_all_actions]
        predicted_value = max(phi_sprime_t_mult_theta_list)

        theta += alpha * (r + gamma * predicted_value - phi_s @ theta) * phi_s

        action_sum_episode_9RBF += 1
        reward_sum_episode_9RBF += r
        iteration += 1
        iteration_terminate += 1

        s = sprime

    return action_sum_episode_9RBF, reward_sum_episode_9RBF


if __name__ == "__main__":

    # Define the grid size, actions, and initialize Q-table
    grid_size = 5
    actions = ['up', 'down', 'left', 'right']
    Q_table = {(i, j): {action: 0 for action in actions} for i in range(grid_size) for j in range(grid_size)}

    # Initialize RBF parameters
    num_RBFs = 4
    centers = np.array([(i, j) for i in range(2) for j in range(2)])  # Four centers, e.g., (0,0), (0,1), (1,0), (1,1)
    mus = np.ones(num_RBFs)  # Spread of each RBF
    theta = np.zeros(16)  # Coefficients for the RBFs

    num_RBFs_9 = 9
    centers_9 = np.array([(i, j) for i in range(3) for j in range(3)])  # Nine centers
    mus_9 = np.ones(num_RBFs_9)  # Spread of each RBF
    theta_9 = np.zeros(4 * num_RBFs_9)  # Coefficients for 9RBFs


    # Simulation parameters
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.1
    episodes = 100
    p_array = [random.random() for _ in range(200)]

    # PLOT OF NUMBER OF ACTIONS THE ROBOT TAKES IN EACH EPISODE
    # START

    # # Actions storage
    # actions_per_episode_normal = []
    # actions_per_episode_RBF_4 = []
    # actions_per_episode_RBF_9 = []
    #
    # for episode in range(episodes):
    #     start_state = (0, 0)  # Starting at the top-left corner of the grid
    #
    #     # Normal Q-learning
    #     action_count_normal, _ = QL_normal(gamma, alpha, epsilon, p_array, start_state, Q_table, episode)
    #     actions_per_episode_normal.append(action_count_normal)
    #
    #     # RBF Q-learning 4RBF
    #     action_count_RBF_4, _ = QL_4RBF(gamma, alpha, epsilon, p_array, start_state, centers, mus, theta, episode)
    #     actions_per_episode_RBF_4.append(action_count_RBF_4)
    #
    #     # RBF Q-learning with 9 RBFs
    #     action_count_RBF_9, _ = QL_9RBF(gamma, alpha, epsilon, p_array, start_state, centers_9, mus_9, theta_9, episode)
    #     actions_per_episode_RBF_9.append(action_count_RBF_9)
    #
    # # Plotting the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(actions_per_episode_normal, label='QL')
    # plt.plot(actions_per_episode_RBF_4, label='RBF-QL 4BF', linestyle='--')
    # plt.plot(actions_per_episode_RBF_9, label='RBF-QL 9BF', linestyle='-.')
    # plt.xlabel('Episodes')
    # plt.ylabel('Number of Steps')
    # plt.title('Number of Actions the Robot Takes in Each Episode')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # END

    # PLOT THE REWARD OF ALL LEARNING EPISODES
    # START

    # Rewards storage
    rewards_per_episode_normal = []
    rewards_per_episode_RBF_4 = []
    rewards_per_episode_RBF_9 = []

    for episode in range(episodes):
        start_state = (0, 0)  # Starting at the top-left corner of the grid

        # Normal Q-learning
        _, reward_sum_episode_normal = QL_normal(gamma, alpha, epsilon, p_array, start_state, Q_table, episode)
        rewards_per_episode_normal.append(reward_sum_episode_normal)

        # RBF Q-learning with 4 RBFs
        _, reward_sum_episode_RBF_4 = QL_4RBF(gamma, alpha, epsilon, p_array, start_state, centers, mus, theta, episode)
        rewards_per_episode_RBF_4.append(reward_sum_episode_RBF_4)

        # RBF Q-learning with 9 RBFs
        _, reward_sum_episode_RBF_9 = QL_9RBF(gamma, alpha, epsilon, p_array, start_state, centers_9, mus_9, theta_9,
                                              episode)
        rewards_per_episode_RBF_9.append(reward_sum_episode_RBF_9)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode_normal, label='QL')
    plt.plot(rewards_per_episode_RBF_4, label='RBF-QL (4RBF)', linestyle='--')
    plt.plot(rewards_per_episode_RBF_9, label='RBF-QL (9RBF)', linestyle='-.')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward of all learning episodes')
    plt.legend()
    plt.grid(True)
    plt.show()
# END