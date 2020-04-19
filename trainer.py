"""Trainer for policy gradient agent."""
from argparse import ArgumentParser
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.replay_buffer import ReplayBuffer
from actor import Actor
from critic import Critic
from agent import Agent


class Trainer:
    """Trainer for policy gradient agent."""

    def __init__(self, agent, writer, params):
        """Initialize the trainer."""
        self.agent = agent
        self.writer = writer

        self._seed = params['seed']
        self.render = params['render']
        self.max_episode = params['max_episode']
        self.random_expl = params['random_expl']
        self.critic_update = params['critic_update']
        self.actor_update = params['actor_update']
        self.critic_threshold = params['critic_threshold']
        self.savepath = params['savepath']

        self.device = params['device']
        self.n_iter = 0  # number of iteration of sampling
        self.episode = 0

        self._fix_random_seed(self._seed)

    def _fix_random_seed(self, seed):
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run_one_episode(self, episode):
        """Train one episode."""
        accumulated_rewards = 0
        # initialize the environment
        ob = self.agent.env.reset()
        if self.render:
            self.agent.env.render()
        done = False
        t = 0
        while not done:
            self.n_iter += 1  # n_iter starts from 1 not from 0
            ob_tsr = torch.tensor([ob],
                                  dtype=torch.float32,
                                  device=self.device)
            # if the number of iteration is small, we choose random action
            if self.n_iter < self.random_expl:
                action = self.agent.env.action_space.sample()
            else:
                action = self.agent.actor.get_action(ob_tsr,
                                                     'epsilon-greedy',
                                                     0.1)

            ob_next, reward, done, _ = self.agent.env.step(action)
            if self.render:
                self.agent.env.render()
            accumulated_rewards += reward

            self.agent.buffer.add_paths([t, ob, action, reward, ob_next, done])
            ob = ob_next
            # if the agent is exploring randomly or there is no enough data,
            # continue collecting data without learning.
            if (self.n_iter < self.random_expl or
                    self.agent.buffer.length < self.agent.buffer.capacity//5):
                continue

            if t % self.critic_update == 0:
                if not self.agent.buffer.can_sample(
                        self.agent.critic.params['batch_size']):
                    continue
                critic_loss = self.agent.critic_update()
                self.writer.add_scalar('loss/critic', critic_loss, self.n_iter)
                print("episode-{} \t\t critic loss: {}".format(episode,
                                                               critic_loss))
            if t % self.actor_update == 0:
                if not self.agent.buffer.can_sample(
                        self.agent.actor.params['batch_size']):
                    continue
                actor_loss = self.agent.actor_update()
                self.writer.add_scalar('loss/actor', actor_loss, self.n_iter)
                print("episode-{} \t\t actor loss: {}".format(episode,
                                                              actor_loss))
            # if done, close environment
            self.agent.env.close()
        return accumulated_rewards

    def run_training_loop(self):
        """Run training loop."""
        assert self.episode < self.max_episode,\
            'already done for maximal episode'

        for ep in range(self.episode, self.max_episode):
            # train one episode
            accumulated_rewards = self.run_one_episode(ep)
            self.writer.add_scalar('Total Rewards', accumulated_rewards, ep)
            self.agent.history['rewards'].append(accumulated_rewards)
            # Stop learning if critic "converges".
            if len(self.agent.history['critic_loss']) > 10:
                recent_crit_loss = np.array(
                        self.agent.history['critic_loss'][-10:])
                mean_loss = recent_crit_loss.mean()
                if mean_loss < self.critic_threshold:
                    print("Early stopping with mean loss : {}!!".format(
                        mean_loss))
                    break
            # Save the parameters of models if ...
            if ep % 5 == 0:
                self.agent.save(self.savepath, ep)
        self.episode = ep

        print("Stop learning after {} episodes!".format(ep))


def get_args():
    """Get arguments from the shell command."""
    parser = ArgumentParser()
    parser.add_argument('--render', help="show rendered image",
                        action='store_true')
    parser.add_argument('--capacity', help="capacity of replay buffer\
                        (default=100)", type=int, default=100)
    parser.add_argument('--max_episode', help="maximal number of episode\
                        for training (default=10)", type=int, default=10)
    parser.add_argument('--random_exploration', help="number of iterations to\
                        explore randomly without learning (default=50)",
                        type=int, default=50)
    parser.add_argument('--critic_update', help='frequency for updating critic\
                        (default=1)', type=int, default=1)
    parser.add_argument('--actor_update', help='frequency for updating actor\
                        (default=1)', type=int, default=1)
    parser.add_argument('--critic_threshold', help='early stopping criterion\
                        on convergence of critic (default=1e-1)', type=float,
                        default=1e-1)

    parser.add_argument('--actor_lr', help='learning rate of actor\
                        (default=1e-3)', type=float, default=1e-3)
    parser.add_argument('--critic_lr', help='learning rate of critic\
                        (default=1e-3)', type=float, default=1e-3)
    parser.add_argument('--batch_size', help='batch size for training actor &\
                        critic (default=16)', type=int, default=16)
    parser.add_argument('--env', help="environment (default='CartPole-v0')",
                        type=str, choices=['CartPole-v0', 'Pendulum-v0'],
                        default='CartPole-v0')
    parser.add_argument('--gamma', help='decreasing rate of future rewards\
                        (default=0.99)', type=float, default=0.99)
    parser.add_argument('--seed', help="random seed (default=2020)", type=int,
                        default=2021)
    parser.add_argument('--load', help="filename for loading the agent\
                        (default=None)", default=None)
    parser.add_argument('--savepath', help="Path to save outputs\
                        (default=backup.pth)", default='backup.pth')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    LOAD = args.load
    CAPACITY = args.capacity

    env = gym.make(args.env)
    # set configuration parameters
    params = {
            'obser_n': env.observation_space.shape[0],
            'action_n': env.action_space.n,
            }
    actor_params = {
            'learning_rate': args.actor_lr,
            'batch_size': args.batch_size,
            }
    critic_params = {
            'learning_rate': args.critic_lr,
            'batch_size': args.batch_size,
            }
    agent_params = {
            'gamma': args.gamma,
            }
    actor_params.update(params)
    critic_params.update(params)

    # Use GPU if it is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    actor = Actor(env, actor_params)
    critic = Critic(critic_params, q=True)  # Q-function
    actor.to(device)
    critic.to(device)

    actor_optim = torch.optim.Adam(actor.parameters(),
                                   lr=actor.params['learning_rate'])
    critic_optim = torch.optim.Adam(critic.parameters(),
                                    lr=critic.params['learning_rate'])

    # create the actor-critic agent
    agent_params.update(params)
    memory = ReplayBuffer(CAPACITY)
    agent = Agent(env, actor, critic, memory, actor_optim, critic_optim,
                  agent_params, device)

    # Initialize or load the model
    if not LOAD:
        agent.actor._initialize()
        agent.critic._initialize()
    else:
        agent.load(LOAD)

    # summary writer for tensorboard
    writer = SummaryWriter()
    # Parameters for trainer
    trainer_params = {
            'seed': args.seed,
            'render': args.render,
            'max_episode': args.max_episode,
            'random_expl': args.random_exploration,
            'critic_update': args.critic_update,
            'actor_update': args.actor_update,
            'critic_threshold': args.critic_threshold,
            'savepath': args.savepath,
            'device': device,
            }
    trainer = Trainer(agent, writer, trainer_params)
    trainer.run_training_loop()
