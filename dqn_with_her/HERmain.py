import os
import matplotlib.pyplot as plt
import numpy as np

import BitFlipEnv as bflip
from dqn_with_her import DQNAgentWithHER as dqnHER

if __name__ == '__main__':

    n_bits = 8
    env = bflip.BitFlipEnv(n_bits)

    n_episodes = 30000
    epsilon_history = []
    episodes = []
    win_percent = []
    success = 0

    load_checkpoint = False

    checkpoint_dir = os.path.join(os.getcwd(), '/checkpoint/')

    # Initializes the DQN agent with Hindsight Experience Replay
    agent = dqnHER.DQNAgentWithHER(learning_rate=0.0001, n_actions=n_bits,
                                   input_dims=n_bits, gamma=0.99,
                                   epsilon=0.9, batch_size=64, memory_size=10000,
                                   replace_network_count=50,
                                   checkpoint_dir=checkpoint_dir)

    if load_checkpoint:
        agent.load_model()

    # Iterate through the episodes
    for episode in range(n_episodes):
        env.reset_env()
        state = env.state
        goal = env.goal
        done = False
        transitions = []

        for p in range(n_bits):
            if not done:
                action = agent.choose_action(state, goal)
                next_state, reward, done = env.take_step(action)
                if not load_checkpoint:
                    agent.store_experience(state, action, reward, next_state, done, goal)
                    transitions.append((state, action, reward, next_state))
                    agent.learn()
                state = next_state

                if done:
                    success += 1

        if not done:
            new_goal = np.copy(state)
            if not np.array_equal(new_goal, goal):
                for p in range(n_bits):
                    transition = transitions[p]
                    if np.array_equal(transition[3], new_goal):
                        agent.store_experience(transition[0], transition[1], 0.0,
                                               transition[3], True, new_goal)
                        agent.learn()
                        break

                    agent.store_experience(transition[0], transition[1], transition[2],
                                           transition[3], False, new_goal)
                    agent.learn()

        # Average over last 500 episodes to avoid spikes
        if episode % 500 == 0:
            print('success rate for last 500 episodes after', episode, ':', success / 5)
            if len(win_percent) > 0 and (success / 500) > win_percent[len(win_percent) - 1]:
                agent.save_model()
            epsilon_history.append(agent.epsilon)
            episodes.append(episode)
            win_percent.append(success / 500.0)
            success = 0

    print('Epsilon History:', epsilon_history)
    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(episodes, win_percent)

    plt.title('DQN with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), '/plots/'))
