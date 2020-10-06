import matplotlib.pyplot as plt

import BitFlipEnv as bflip
import DQNAgent as dqn


if __name__ == '__main__':

    n_bits = 8
    env = bflip.BitFlipEnv(n_bits)

    n_episodes = 30000
    epsilon_history = []
    episodes = []
    win_percent = []
    success = 0

    load_checkpoint = False

    agent = dqn.DQNAgent(learning_rate=0.0001, n_actions=n_bits,
                         input_dims=n_bits, gamma=0.99,
                         epsilon=0.9, batch_size=64, memory_size=10000,
                         replace_network_count=50,
                         checkpoint_dir='/home/blackreaper/Documents/temp/duelingdqn')

    if load_checkpoint:
        agent.load_model()

    for episode in range(n_episodes):
        env.reset_env()
        state = env.state
        goal = env.goal
        done = False

        for p in range(n_bits):
            if not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.take_step(action)
                if not load_checkpoint:
                    agent.store_experience(state, action, reward, next_state, done)
                    agent.learn()
                state = next_state

                if done:
                    success += 1

        if episode % 500 == 0:
            print('success rate for last 500 episodes after', episode, ':', success/5)
            if len(win_percent) > 0 and (success / 500) > win_percent[len(win_percent) - 1]:
                agent.save_model()
            epsilon_history.append(agent.epsilon)
            episodes.append(episode)
            win_percent.append(success/500.0)
            success = 0

    print('Epsilon History:', epsilon_history)
    print('Episodes:', episodes)
    print('Win percentage:', win_percent)

    figure = plt.figure()
    plt.plot(episodes, win_percent)

    plt.title('DQN without HER')
    plt.ylabel('Win Percentage')
    plt.xlabel('Number of Episodes')
    plt.ylim([0, 1])

    plt.savefig('/home/blackreaper/Documents/temp/duelingdqn/abc1.png')
