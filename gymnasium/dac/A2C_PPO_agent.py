#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from typing import Callable, Iterable

import torch
from torch import nn, Tensor

import numpy as np
import pickle

from skimage import color
from skimage.io import imsave

from gymnasium.vector.vector_env import VectorEnv
from gymnasium import Env

import time

from .util.torch_utils import tensor
from .util.misc import random_sample, to_np
from .component.normalizer import BaseNormalizer, RescaleNormalizer
from .component.replay import Storage

# 'hat' is the high-MDP
# 'bar' is the low-MDP


class ASquaredCPPOAgentConfig:
    def __init__(
        self,
        num_o: int,
        state_dim: int,
        action_dim: int,

        opt_ep: int,
        save_interval: int,
        log_interval: int,
        eval_interval: int,

        gate: object,
        entropy_weight: float,
        beta_weight: float,
        ppo_ratio_clip: float,

        max_steps: int,
        discount: float,
        rollout_length: int,
        optimization_epochs: int,
        mini_batch_size: int,

        env: VectorEnv,
        eval_env: Env,
        num_workers: int,

        network: nn.Module,
        optimizer_fn: Callable[[Iterable[Tensor]], torch.optim.Optimizer],

        use_gae: bool = False,
        gae_tau: float = 1.0,
        gradient_clip: float = 0.5,

        learning: str = 'all',
        log_level: int = 1,
        freeze_v: bool = True,
        tasks: bool = False,

        state_normalizer: BaseNormalizer = RescaleNormalizer(),
        reward_normalizer: BaseNormalizer = RescaleNormalizer(),
    ) -> None:
        self.learning = learning
        self.log_level = log_level
        self.num_o = num_o
        self.opt_ep = opt_ep
        self.freeze_v = freeze_v
        self.tasks = tasks
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.gate = gate
        self.entropy_weight = entropy_weight
        self.max_steps = max_steps
        self.beta_weight = beta_weight

        self.env = env
        self.eval_env = eval_env
        self.num_workers = num_workers

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network = network
        self.optimizer_fn = optimizer_fn

        self.discount = discount
        self.use_gae = use_gae
        self.gae_tau = gae_tau
        self.gradient_clip = gradient_clip
        self.rollout_length = rollout_length
        self.optimization_epochs = optimization_epochs
        self.mini_batch_size = mini_batch_size
        self.ppo_ratio_clip = ppo_ratio_clip
        self.log_interval = log_interval
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer


class ASquaredCPPOAgent:
    def __init__(self, config: ASquaredCPPOAgentConfig):
        self.config = config

        self.env = config.env
        self.eval_env = config.eval_env

        self.network = config.network
        self.opt = config.optimizer_fn(self.network.parameters())

        self.total_steps = 0

        self.worker_index = tensor(np.arange(config.num_workers)).long()
        self.states, _ = self.env.reset()
        self.states = config.state_normalizer(self.states)
        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = tensor(np.zeros(config.num_workers)).long()

        self.count = 0

        self.all_options = []

        self.storage = Storage(config.rollout_length, [
            's', 'a', 'r', 'm', 'init', 'o', 'prev_o',

            'pi_bar', 'pi_hat',
            'log_pi_bar', 'log_pi_hat',

            'v_bar', 'v_hat',
            'adv_bar', 'adv_hat',
            'ret_bar', 'ret_hat',

            'mean', 'std', 'q_o', 'u_o',
            'inter_pi', 'log_inter_pi', 'beta',
        ])
        # ] + self.network.output_keys)

    def compute_pi_hat(self, prediction, prev_option, is_intial_states):
        inter_pi = prediction['inter_pi']
        mask = torch.zeros_like(inter_pi)
        mask[self.worker_index, prev_option] = 1
        beta = prediction['beta']
        pi_hat = (1 - beta) * mask + beta * inter_pi
        is_intial_states = is_intial_states.view(
            -1, 1).expand(-1, inter_pi.size(1))
        pi_hat = torch.where(is_intial_states, inter_pi, pi_hat)
        return pi_hat

    def compute_pi_bar(self, options, action, mean, std):
        options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
        mean = mean.gather(1, options).squeeze(1)
        std = std.gather(1, options).squeeze(1)
        dist = torch.distributions.Normal(mean, std)
        pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
        return pi_bar

    def compute_log_pi_a(self, options, pi_hat, action, mean, std, mdp):
        if mdp == 'hat':
            return pi_hat.add(1e-5).log().gather(1, options)
        elif mdp == 'bar':
            pi_bar = self.compute_pi_bar(options, action, mean, std)
            return pi_bar.add(1e-5).log()
        else:
            raise NotImplementedError

    def compute_adv(self, storage, mdp):
        config = self.config
        v = storage.__getattribute__('v_%s' % (mdp))
        adv = storage.__getattribute__('adv_%s' % (mdp))
        all_ret = storage.__getattribute__('ret_%s' % (mdp))

        ret = v[-1].detach()
        advantages = tensor(np.zeros((config.num_workers, 1)))
        for i in reversed(range(config.rollout_length)):
            ret = storage.r[i] + config.discount * storage.m[i] * ret
            if not config.use_gae:
                advantages = ret - v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * \
                    storage.m[i] * v[i + 1] - v[i]
                advantages = advantages * config.gae_tau * \
                    config.discount * storage.m[i] + td_error
            adv[i] = advantages.detach()
            all_ret[i] = ret.detach()

    def learn(self, storage, mdp, freeze_v=False):
        config = self.config
        states, actions, options, log_probs_old, \
            returns, advantages, prev_options, inits, \
            pi_hat, mean, std = \
            storage.cat([
                's', 'a', 'o', f'log_pi_{mdp}',
                f'ret_{mdp}', f'adv_{mdp}', 'prev_o', 'init',
                'pi_hat', 'mean', 'std'
            ])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        pi_hat = pi_hat.detach()
        mean = mean.detach()
        std = std.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        total_policy_loss = 0
        total_value_loss = 0

        total_batch = 0

        for _ in range(config.optimization_epochs):
            sampler = random_sample(
                np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()

                sampled_pi_hat = pi_hat[batch_indices]
                sampled_mean = mean[batch_indices]
                sampled_std = std[batch_indices]
                sampled_states = states[batch_indices]
                sampled_prev_o = prev_options[batch_indices]
                sampled_init = inits[batch_indices]

                sampled_options = options[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states)

                if mdp == 'hat':
                    cur_pi_hat = self.compute_pi_hat(
                        prediction, sampled_prev_o.view(-1), sampled_init.view(-1))
                    entropy = - \
                        (cur_pi_hat * cur_pi_hat.add(1e-5).log()).sum(-1).mean()
                    log_pi_a = self.compute_log_pi_a(
                        sampled_options, cur_pi_hat, sampled_actions, sampled_mean, sampled_std, mdp)
                    beta_loss = prediction['beta'].mean()
                elif mdp == 'bar':
                    log_pi_a = self.compute_log_pi_a(
                        sampled_options, sampled_pi_hat, sampled_actions, prediction['mean'], prediction['std'], mdp)
                    entropy = 0
                    beta_loss = 0
                else:
                    raise NotImplementedError

                if mdp == 'bar':
                    v = prediction['q_o'].gather(1, sampled_options)
                elif mdp == 'hat':
                    v = (prediction['q_o'] *
                         sampled_pi_hat).sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError

                ratio = (log_pi_a - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - config.ppo_ratio_clip,
                                          1.0 + config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * entropy + \
                    config.beta_weight * beta_loss

                discarded = (obj > obj_clipped).float().mean()
                # self.logger.add_scalar('clipped_%s' % (mdp), discarded, log_level=5)

                value_loss = 0.5 * (sampled_returns - v).pow(2).mean()
                # self.logger.add_scalar('v_loss', value_loss.item(), log_level=5)
                if freeze_v:
                    value_loss = 0

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    self.network.parameters(), config.gradient_clip)
                self.opt.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batch += 1

        avg_policy_loss = total_policy_loss / total_batch
        avg_value_loss = total_value_loss / total_batch
        print(mdp)
        print('    policy_loss: %.5f, value_loss: %.5f' %
              (avg_policy_loss, avg_value_loss))

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            # if ret is not None:
            #     self.logger.add_scalar(
            #         'episodic_return_train', ret, self.total_steps + offset)
            #     self.logger.info('steps %d, episodic_return_train %s' %
            #                      (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def step(self):
        config = self.config
        states = self.states
        t0 = time.time()

        total_reward = 0.0

        for _ in range(config.rollout_length):
            prediction = self.network(states)
            pi_hat = self.compute_pi_hat(
                prediction, self.prev_options, self.is_initial_states)
            dist = torch.distributions.Categorical(probs=pi_hat)
            options = dist.sample()

            # self.logger.add_scalar('beta', prediction['beta'][self.worker_index, self.prev_options], log_level=5)
            # self.logger.add_scalar('option', options[0], log_level=5)
            # self.logger.add_scalar('pi_hat_ent', dist.entropy(), log_level=5)
            # self.logger.add_scalar('pi_hat_o', dist.log_prob(options).exp(), log_level=5)

            mean = prediction['mean'][self.worker_index, options]
            std = prediction['std'][self.worker_index, options]
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()

            pi_bar = self.compute_pi_bar(options.unsqueeze(-1), actions,
                                         prediction['mean'], prediction['std'])

            v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
            v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

            next_states, rewards, terminals, truncated, info = self.env.step(
                to_np(actions))
            # self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            self.storage.add(prediction)

            self.storage.add({'r': tensor(rewards).unsqueeze(-1),
                              'm': tensor(1 - terminals).unsqueeze(-1),
                              'a': actions,
                              'o': options.unsqueeze(-1),
                              'prev_o': self.prev_options.unsqueeze(-1),
                              's': tensor(states),
                              'init': self.is_initial_states.unsqueeze(-1),
                              'pi_hat': pi_hat,
                              'log_pi_hat': pi_hat[self.worker_index, options].add(1e-5).log().unsqueeze(-1),
                              'log_pi_bar': pi_bar.add(1e-5).log(),
                              'v_bar': v_bar,
                              'v_hat': v_hat})

            self.is_initial_states = tensor(terminals).byte()
            self.prev_options = options

            states = next_states
            self.total_steps += config.num_workers

            total_reward += rewards[0]
            if terminals[0]:
                print(f"return: {total_reward}")
                total_reward = 0.0

            # print('steps %d, %.2f steps/s' % (self.total_steps,
            #       config.num_workers / (time.time() - t0)))

        print(f"return: {total_reward}")

        self.states = states
        prediction = self.network(states)
        pi_hat = self.compute_pi_hat(
            prediction, self.prev_options, self.is_initial_states)
        dist = torch.distributions.Categorical(pi_hat)
        options = dist.sample()
        v_bar = prediction['q_o'].gather(1, options.unsqueeze(-1))
        v_hat = (prediction['q_o'] * pi_hat).sum(-1).unsqueeze(-1)

        self.storage.add(prediction)
        self.storage.add({
            'v_bar': v_bar,
            'v_hat': v_hat,
        })
        self.storage.placeholder()

        # [o] = storage.cat(['o'])
        # for i in range(config["num_o"]):
        #     self.logger.add_scalar('option_%d' % (
        #         i), (o == i).float().mean(), log_level=1)

        self.compute_adv(self.storage, 'bar')
        self.compute_adv(self.storage, 'hat')

        if config.learning == 'all':
            mdps = ['hat', 'bar']
            np.random.shuffle(mdps)
            self.learn(self.storage, mdps[0])
            self.learn(self.storage, mdps[1])
        elif config.learning == 'alt':
            if self.count % 2:
                self.learn(self.storage, 'hat')
            else:
                self.learn(self.storage, 'bar')
            self.count += 1

    def close(self):
        self.env.close()
        self.eval_env.close()

    def eval_episode(self, epi):
        config = self.config
        env = self.config.eval_env

        total_epi_return = 0.0

        config.state_normalizer.set_read_only()
        config.reward_normalizer.set_read_only()

        for _ in range(epi):
            state, _ = env.reset()
            state = np.expand_dims(state, 0)

            epi_return = 0.0
            while True:
                prediction = self.network(state)
                pi_hat = self.compute_pi_hat(
                    prediction, self.prev_options, self.is_initial_states)
                dist = torch.distributions.Categorical(probs=pi_hat)
                options = dist.sample()

                mean = prediction['mean'][self.worker_index, options]
                std = prediction['std'][self.worker_index, options]
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                action = to_np(actions)[0]

                next_state, reward, terminal, truncated, info = env.step(
                    action)
                epi_return += float(reward)
                reward = config.reward_normalizer(reward)
                state = config.state_normalizer(next_state)
                # next_state = config.state_normalizer(np.expand_dims(next_state, 0))
                # state = np.expand_dims(next_state, 0)

                self.is_initial_states = tensor([terminal]).byte()
                self.prev_options = options

                if terminal or truncated:
                    break
                # break
#
                # action = self.eval_step(state)
                # state, reward, done, info = env.step(action)
                # ret = info[0]['episodic_return']
                # if ret is not None:
                #     break
            # return ret

            total_epi_return += epi_return

        self.is_initial_states = tensor(np.ones((config.num_workers))).byte()
        self.prev_options = tensor(np.zeros(config.num_workers)).long()

        print('eval episodic_return: %.5f' % (total_epi_return / epi))

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename,
                                map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))


def eval(agent: ASquaredCPPOAgent):
    config = agent.config
    agent.load("data/DAC-test-0")
    agent.eval_episode(5)


def run_steps(agent: ASquaredCPPOAgent):
    config = agent.config

    while True:
        # if config["save_interval"] and not agent.total_steps % config["save_interval"]:
        #     agent.save('data/%s-%s-%d' % (agent_name, config["tag"], agent.total_steps))
        # if config["log_interval"] and not agent.total_steps % config["log_interval"]:
        #     agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config["log_interval"] / (time.time() - t0)))
        #     t0 = time.time()
        # agent.save('data/%s-%s-%d' % ("DAC", "test", agent.total_steps))
        # if agent.total_steps > 0 and not agent.total_steps % config.eval_interval:
        #     agent.eval_episode()
        if agent.total_steps >= config.max_steps:
            agent.close()
            break

        agent.step()

        # print('steps %d, %.2f steps/s' %
        #       (agent.total_steps, steps / (time.time() - t0)))
    agent.save('data/%s-%s-%d' % ("DAC", "test", 0))
