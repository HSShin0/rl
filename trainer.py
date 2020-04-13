from argparse import ArgumentParser
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.replay_buffer import ReplayBuffer
from actor import Actor
from critic import Critic
from agent import Agent


def get_args():
    """Get arguments from the shell command."""
    parser = ArgumentParser() 
    parser.add_argument('--render', help="show rendered image", action='store_true')
    parser.add_argument('--capacity', help="capacity of replay buffer (default=100)",
                        type=int, default=100)
    parser.add_argument('--max_episode', help="maximal number of episode for training\
                        (default=10)", type=int, default=10)
    parser.add_argument('--random_exploration', help="number of episode to explore \
                        randomly without learning (default=5)", type=int, default=5)
    parser.add_argument('--critic_update', help='frequency for updating critic (default=1)',
                        type=int, default=1)
    parser.add_argument('--actor_update', help='frequency for updating actor (default=1)',
                        type=int, default=1)
    parser.add_argument('--critic_threshold', help='early stopping criterion on convergence \
                        of critic (default=1e-1)', type=float, default=1e-1)

    parser.add_argument('--actor_lr', help='learning rate of actor (default=1e-3)',
                        type=float, default=1e-3)
    parser.add_argument('--critic_lr', help='learning rate of critic (default=1e-3)',
                        type=float, default=1e-3)
    parser.add_argument('--batch_size', help='batch size for training actor & critic \
                        default=16', type=int, default=16)
    parser.add_argument('--env', help="environment (default='CartPole-v0')", type=str,
                        choices=['CartPole-v0', 'Pendulum-v0'], default='CartPole-v0')
    parser.add_argument('--gamma', help='decreasing rate of future rewards (default=0.99)',
                        type=float, default=0.99)
    parser.add_argument('--seed', help="random seed (default=2020)", type=int,
                        default=2020)
    parser.add_argument('--load', help="filename for loading the agent (default=None)",
                        default=None)
    parser.add_argument('--savepath', help="Path to save outputs\
                        (default=backup.pth)", default='backup.pth')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # fix random seed
    SEED = args.seed
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # summary writer for tensorboard
    writer = SummaryWriter()

    RENDER = args.render
    LOAD = args.load
    SAVEPATH = args.savepath

    CAPACITY = args.capacity
    MAX_EPISODE = args.max_episode
    RANDOM_EXPL = args.random_exploration

    CRITIC_UPDATE = args.critic_update
    CRITIC_THRESHOLD = args.critic_threshold
    ACTOR_UPDATE = args.actor_update

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
    critic = Critic(critic_params, q=True) # Q-function
    actor.to(device)
    critic.to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor.params['learning_rate'])
    critic_optim = torch.optim.Adam(critic.parameters(), lr=critic.params['learning_rate'])

    # create the actor-critic agent
    agent_params.update(params)
    memory = ReplayBuffer(CAPACITY)
    agent = Agent(env, actor, critic, memory, actor_optim, critic_optim, agent_params, device)

    # Initialize or load the model
    if not LOAD:
        agent.actor._initialize()
        agent.critic._initialize()
    else:
        agent.load(LOAD)

    # Collect samples using actor.
    n_iter = -1
    for episode in range(MAX_EPISODE):
        accumlated_rewards = 0

        ob = agent.env.reset()
        if RENDER:
            agent.env.render()
        done = False
        t = 0
        while not done:
            n_iter += 1
            ob_tsr = torch.tensor([ob], dtype=torch.float32, device=device)
            if episode < RANDOM_EXPL:
                action = agent.env.action_space.sample()
            else:
                action = agent.actor.get_action(ob_tsr, 'epsilon-greedy', 0.1)
            ob_next, reward, done, _ = agent.env.step(action)
            accumlated_rewards += reward
            if RENDER:
                agent.env.render()
            memory.add_paths([t, ob, action, reward, ob_next, done])
            ob = ob_next
            # If there is no enough data, continue collecting data without learning.
            if memory.length < memory.capacity // 5:
                continue

            if episode >= RANDOM_EXPL:
                if t % CRITIC_UPDATE == 0:
                    if not agent.buffer.can_sample(agent.critic.params['batch_size']):
                        continue
                    critic_loss = agent.critic_update()
                    writer.add_scalar('Loss/critic', critic_loss, n_iter)
                    print("Episode-{} \t\t Critic loss: {}".format(episode, critic_loss))
                if t % ACTOR_UPDATE == 0:
                    if not agent.buffer.can_sample(agent.actor.params['batch_size']):
                        continue
                    actor_loss = agent.actor_update()
                    writer.add_scalar('Loss/actor', actor_loss, n_iter)
                    print("Episode-{} \t\t Actor loss: {}".format(episode, actor_loss))

        # If done, add the toal rewards.
        writer.add_scalar('Total Rewards', accumlated_rewards, episode)
        agent.history['rewards'].append(accumlated_rewards)

        # Stop learning if loss of critic is sufficiently small for several batches.
        if len(agent.history['critic_loss']) > 10:
            recent_critic_loss_np = np.array(agent.history['critic_loss'][-10: ])
            mean_loss = recent_critic_loss_np.mean()
            if mean_loss < CRITIC_THRESHOLD:
                print("Early stopping with mean loss : {}!!".format(mean_loss))
                break

        # Save the parameters of models if ...
        if episode % 5 == 0:
            agent.save(SAVEPATH, episode)

    print("Stop learning after {} episodes!".format(episode))
