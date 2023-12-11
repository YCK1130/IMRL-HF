#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
import argparse
import torch


DEVICE = torch.device('mps')

config = {}

config["parser"] = argparse.ArgumentParser()
config["task_fn"] = None
config["optimizer_fn"] = None
config["actor_optimizer_fn"] = None
config["critic_optimizer_fn"] = None
config["network_fn"] = None
config["actor_network_fn"] = None
config["critic_network_fn"] = None
config["replay_fn"] = None
config["random_process_fn"] = None
config["discount"] = None
config["target_network_update_freq"] = None
config["exploration_steps"] = None
config["log_level"] = 0
config["history_length"] = None
config["double_q"] = False
config["tag"] = 'vanilla'
config["num_workers"] = 1
config["gradient_clip"] = None
config["entropy_weight"] = 0
config["use_gae"] = False
config["gae_tau"] = 1.0
config["target_network_mix"] = 0.001
config["state_normalizer"] = RescaleNormalizer()
config["reward_normalizer"] = RescaleNormalizer()
config["min_memory_size"] = None
config["max_steps"] = 0
config["rollout_length"] = None
config["value_loss_weight"] = 1.0
config["iteration_log_interval"] = 30
config["categorical_v_min"] = None
config["categorical_v_max"] = None
config["categorical_n_atoms"] = 51
config["num_quantiles"] = None
config["optimization_epochs"] = 4
config["mini_batch_size"] = 64
config["termination_regularizer"] = 0
config["sgd_update_frequency"] = None
config["random_action_prob"] = None
config["__eval_env"] = None
config["log_interval"] = int(1e3)
config["save_interval"] = 0
config["eval_interval"] = 0
config["eval_episodes"] = 10
config["async_actor"] = True
config["tasks"] = False

    # @property
    # def eval_env(self):
    #     return self.__eval_env
    #
    # @eval_env.setter
    # def eval_env(self, env):
    #     self.__eval_env = env
    #     self.state_dim = env.state_dim
    #     self.action_dim = env.action_dim
    #     self.task_name = env.name
    #
    # def add_argument(self, *args, **kwargs):
    #     self.parser.add_argument(*args, **kwargs)
    #
    # def merge(self, config_dict=None):
    #     if config_dict is None:
    #         args = self.parser.parse_args()
    #         config_dict = args.__dict__
    #     for key in config_dict.keys():
    #         setattr(self, key, config_dict[key])
