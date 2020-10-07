import numpy as np

import torch

from ActorCritic import Actor, Critic
from dqn_with_her import HERMemory as her
import OUNoise as noise


class DDPG:
    def __init__(self, learning_rate, n_actions, input_dims, gamma,
                 memory_size, batch_size, tau=0.99,
                 checkpoint_dir='/tmp/ddqn/'):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau

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

    def store_experience(self, state, action, reward, next_state, done, goal):
        """
        Saves the experience to the replay memory
        """
        self.memory.add_experience(state=state, action=action,
                                   reward=reward, next_state=next_state,
                                   done=done, goal=goal)

    def get_sample_experience(self):
        """
        Gives a sample experience from the experience replay memory
        """
        state, action, reward, next_state, done, goal = self.memory.get_random_experience(
            self.batch_size)

        t_state = torch.tensor(state).to(self.actor.device)
        t_action = torch.tensor(action).to(self.actor.device)
        t_reward = torch.tensor(reward).to(self.actor.device)
        t_next_state = torch.tensor(next_state).to(self.actor.device)
        t_done = torch.tensor(done).to(self.actor.device)
        t_goal = torch.tensor(goal).to(self.actor.device)

        return t_state, t_action, t_reward, t_next_state, t_done, t_goal

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(state)
        action = mu + torch.tensor(self.ou_noise(), dtype=torch.float).to(self.actor.device)

        self.actor.train()
        return action

    def learn(self):
        if self.memory.counter < self.batch_size:
            return

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        state, action, reward, next_state, done, goal = self.get_sample_experience()

        target_actions = self.target_actor.forward(state)
        critic_next_value = self.target_critic.forward(next_state, target_actions).view(-1)

        actor_value = self.actor.forward(state)
        critic_value = self.critic.forward(state, action)

        critic_value[done] = 0.0

        target = reward + self.gamma * critic_next_value

        loss_critic = self.critic.loss(target, critic_value)
        loss_critic.backward()
        self.critic.optimizer.step()

        loss_actor = -torch.mean(self.critic.forward(state, actor_value))
        loss_actor.backward()
        self.actor.optimizer.step()

        actor_parameters = dict(self.actor.named_parameters())
        critic_parameters = dict(self.critic.named_parameters())
        target_actor_parameters = dict(self.target_actor.named_parameters())
        target_critic_parameters = dict(self.target_critic.named_parameters())

        for i in actor_parameters:
            actor_parameters[i] = self.tau * actor_parameters[i] + (1 - self.tau) * target_actor_parameters[i]

        for i in critic_parameters:
            critic_parameters[i] = self.tau * critic_parameters[i] + (1 - self.tau) * target_critic_parameters[i]

        self.target_actor.load_state_dict(actor_parameters)
        self.target_critic.load_state_dict(critic_parameters)
