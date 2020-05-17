"""
Note:
    1. advantage method
        It is used for updating both of actor and critic.
        However, we do not consider "done" when we update actor.
        Note that advantage value is used as a weight for gradient of
        log policy when we fit actor. The reason for that is as follows.
        Let (s,a,s',r,done=True) be the sample for fitting.
        If we "zero out" the values of Q or V for terminal states,
        the weight for the gradient "r + gamma * V(s') - V(s)"
        becomes "r - V(s)", which looks inadequate.
        MORE RIGOROUS REASON ??
    2. target networks
        How to implement it for "soft" targeting?
    3. episode
        Need to change the :meth load: because the :param episode: is not
        used in the proper manner.
"""
import numpy as np
import os
import torch


class Agent:
    """Actor-Critic agent."""

    def __init__(self, env, actor, critic, buffer, actor_optim,
                 critic_optim, params, device, target_update_lag=5):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.device = device

        if isinstance(critic, list):
            assert (isinstance(critic_optim, list) and
                    len(critic_optim) == len(critic) == 2), \
                    'Wrong number of critics and optimizers \
                    for target networks'
            self.use_target = True
            self.critics = self.critic
            self.critic = self.critics[0]
            self.target = self.critics[1]
            self.critic_optims = self.critic_optim
            self.critic_optim = self.critic_optims[0]
            self.target_optim = self.critic_optims[1]
            self._every_target_update = target_update_lag
        else:
            self.use_target = False
        # We will exchange critic and target after updating
        # critic self._every_target_update times.
        self.n_critic_update = 0

        self.params = params
        self.obser_n = params['obser_n']
        self.action_n = params['action_n']
        self.gamma = params['gamma']  # Discount factor

        self.history = {
                        'rewards': [],
                        'critic_loss': [],
                        'actor_loss': [],
                        }

    def _update_target(self):
        """Update target network."""
        if self.use_target:
            for cri_param, tar_param in zip(self.critic.parameters(),
                                            self.target.parameters()):
                with torch.no_grad():
                    tar_param.copy_(cri_param)

            '''
            def _test():
                batch_size = self.critic.params['batch_size']
                if not self.buffer.can_sample(batch_size):
                    return None
                # Sampling a batch of paths from the replay buffer
                t, obs, act, rew, obs_next, done = \
                        self.get_batch_samples(batch_size, recent=False)
                err = torch.mean(abs(self.critic(obs) - self.target(obs)))
                assert err < 1e-6, 'wrong update'
                return 0
            with torch.no_grad():
                _test()
            '''

    def advantage(self, obser, reward, obser_next, done=None,
                  critic_learning=False):
        """
        Compute advantage.

        obser (torch.FloatTensor): shape [N, (shape of observation_space)]

        return (torch.Tensor): shape [N, 1]
        """
        batch_size = len(obser)
        val = self.critic.forward(obser)
        if critic_learning and self.use_target:
            val_next = self.target.forward(obser_next)
        else:
            val_next = self.critic.forward(obser_next)
        # If the critic is Q-function, use "max Q" for estimating "V".
        if self.critic._q:
            val, _ = val.max(dim=1, keepdim=True)
            val_next, _ = val_next.max(dim=1, keepdim=True)
        # If done, do not add (V or) Q-value of next observation state.
        if isinstance(done, np.ndarray):
            val_next[done] = 0
        advantage = reward.reshape(batch_size, 1) + self.gamma * val_next - val

        return advantage

    def get_batch_samples(self, batch_size, recent=False):
        """Sample a batch of paths from the replay buffer."""
        samples = self.buffer.get_samples(batch_size, recent)

        timestep = np.zeros(batch_size, dtype=np.int16)
        obser = torch.zeros((batch_size, self.obser_n),
                            dtype=torch.float32, device=self.device)
        action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        reward = torch.zeros(batch_size,
                             dtype=torch.float32, device=self.device)
        obser_next = torch.zeros((batch_size, self.obser_n),
                                 dtype=torch.float32, device=self.device)
        done = np.empty(batch_size, dtype=np.bool)

        for idx, sample in enumerate(samples):
            t, o, a, r, o1, d = sample
            timestep[idx] = t
            obser[idx] = torch.tensor(o)
            action[idx] = torch.tensor(a)
            reward[idx] = r
            obser_next[idx] = torch.tensor(o1)
            done[idx] = d

        return timestep, obser, action, reward, obser_next, done

    def actor_update(self):
        batch_size = self.actor.params['batch_size']
        if not self.buffer.can_sample(batch_size):
            return None

        timestep, obser, action, reward, obser_next, done = self.get_batch_samples(batch_size, recent=True)

        # Compute present policy
        policy_logit = self.actor.forward(obser)
        # Compute advantage
        advantage = self.advantage(obser,
                                   reward,
                                   obser_next)  # (batch_size, n_action)
        # policy logit for actions in batch samples
        logit = policy_logit[range(batch_size), action]  # (batch_size,)
        loss = -(logit.unsqueeze(1) * advantage).mean(dim=0)[0]
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        loss_np = loss.detach().cpu().numpy()
        self.history['actor_loss'].append(loss_np)

        return loss_np

    def critic_update(self):
        batch_size = self.critic.params['batch_size']
        if not self.buffer.can_sample(batch_size):
            return None
        # Sampling a batch of paths from the replay buffer
        t, obs, act, rew, obs_next, done = \
            self.get_batch_samples(batch_size, recent=False)
        advantage = self.advantage(obs, rew, obs_next, done,
                                   critic_learning=True)
        loss = torch.norm(advantage, p=2) / batch_size
        # Gradient descent
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        loss_np = loss.detach().cpu().numpy()
        self.history['critic_loss'].append(loss_np)

        self.n_critic_update += 1
        # Update target network
        if (self.use_target and
                self.n_critic_update % self._every_target_update == 0):
            self._update_target()
        return loss_np

    def train_one_episode(self):
        pass

    def save(self, savepath, episode):
        if not os.path.exists('./save'):
            os.makedirs('./save')
        path = os.path.join('./save', savepath)
        torch.save({
                    'episode': episode,
                    'history': self.history,
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optim.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    }, path
                   )
        print("Saved the agent to {}".format(path))

    def load(self, filename):
        if not os.path.exists(filename):
            filename = os.path.join('./save', filename)
            assert os.path.exists(filename),\
                "wrong path for loading model."
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        episode = checkpoint['episode']
        self.history = checkpoint['history']
        print("Loaded parameters from {}".format(filename))
