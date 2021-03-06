import os
import matplotlib.pyplot as plt
import numpy as np

from ddpg_with_her import DDPGAgent, ContinuousEnv as cenv

if __name__ == '__main__':
    size = 5
    env = cenv.ContinuousEnv(size=size)

    n_episodes = 10000
    print(n_episodes)
    episodes = []
    win_percent = []
    success = 0

    load_checkpoint = False

    checkpoint_dir = os.path.join(os.getcwd(), '/checkpoint/')

    # Initializes the DDPG agent
    agent = DDPGAgent.DDPGAgent(actor_learning_rate=0.0001, critic_learning_rate=0.001, n_actions=2,
                                input_dims=2, gamma=0.99,
                                memory_size=10000, batch_size=64,
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

        for p in range(10):
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
                for q in range(4):
                    transition = transitions[q]
                    good_done = np.linalg.norm(new_goal) <= 0.5
                    bad_done = np.max(np.abs(transition[3])) > size
                    if good_done or bad_done:
                        agent.store_experience(transition[0], transition[1], transition[2],
                                               transition[3], True, new_goal)
                        agent.learn()
                        break

                    agent.store_experience(transition[0], transition[1], transition[2],
                                           transition[3], False, new_goal)
                    agent.learn()

        # Average over last 100 episodes to avoid spikes
        if episode > 0 and episode % 100 == 0:
            print('success rate for last 100 episodes after', episode, ':', success)
            if len(win_percent) > 0 and (success / 100) > win_percent[len(win_percent) - 1]:
                agent.save_model()
            episodes.append(episode)
            win_percent.append(success / 100)
            success = 0

    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(episodes, win_percent)

    plt.title('DDPG with HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig(os.path.join(os.getcwd(), '/plots/'))
