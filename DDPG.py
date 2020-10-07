import numpy as np

from ActorCritic import Actor, Critic
from dqn_with_her import HERMemory as her
import OUNoise as noise


class DDPG:
    def __init__(self, learning_rate, n_actions, input_dims,
                 memory_size, batch_size, checkpoint_dir='/tmp/ddqn/'):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.actor = Actor(input_dims=input_dims, n_actions=n_actions,
                           learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,
                           name='actor')

        self.critic = Critic(input_dims=input_dims, n_actions=n_actions,
                             learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,
                             name='critic')

        self.target_actor = Actor(input_dims=input_dims, n_actions=n_actions,
                                  learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,
                                  name='actor')

        self.target_critic = Critic(input_dims=input_dims, n_actions=n_actions,
                                    learning_rate=learning_rate, checkpoint_dir=checkpoint_dir,
                                    name='critic')

        self.memory = her.HindsightExperienceReplayMemory(memory_size=memory_size,
                                                          input_dims=input_dims)

        self.ou_noise = noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))
