__credits__ = ["Kallinteris-Andreas"]

from typing import Any, Dict
import math
import numpy as np
import os
from stable_baselines3 import SAC, TD3, A2C, PPO
import wandb
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding
from collections import deque

from deep_rl.agent.ASquaredC_PPO_agent import ASquaredCPPOAgent
from deep_rl.component.envs import TaskSB3
from deep_rl.utils import Config, MeanStdNormalizer, generate_tag, run_steps, tensor
from deep_rl.network import FCBody, OptionGaussianActorCriticNet
import torch
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}
COLORMAP = {
    "Gray": np.array([0.2, 0.2, 0.2, 1]),
    "Green": np.array([0, 1, 0, 1]),
    "Yellow": np.array([1, 1, 0, 1]),
    "Blue": np.array([0, 0, 1, 1]),
    "Red": np.array([1, 0, 0, 1]),
    "Orange": np.array([1, 144 / 255, 101 / 255, 1]),
}
EPISODE_LOG = False


def vec_hat(vec):
    return vec / np.linalg.norm(vec)
def dacConfigSetup(first_state_step, alter_state_step, save_model_dir, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('game', 'Fencer')
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', torch.nn.ReLU())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 1e5)
    kwargs.setdefault('beta_weight', 0)
    kwargs.setdefault('real_max_steps', 2e6)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)
    my_config = {
        'first_stage_steps':first_state_step, 
        'second_stage_alternating_steps':alter_state_step, 
        'save_path': save_model_dir,
    }
    config.task_fn = lambda: TaskSB3(config.game, my_config=my_config, method=ASquaredCPPOAgent)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    return config

class GameStatus():
    IDLE = 0
    WIN = 1
    LOSE = 2
    DRAW = 3
    FOUL = 4
    TIMEOUT = 5

    def __init__(self, name, values):
        self.name = name
        assert len(values) == 6
        self.values = values
        self.status = self.IDLE
        self.foul_list = [False, False]

    def agent_win(self):
        if self.status in [self.LOSE, self.DRAW]:
            self.status = self.DRAW
        elif self.status == self.FOUL:
            pass
        else:
            self.status = self.WIN
        return self.status

    def oppent_win(self):
        if self.status in [self.WIN, self.DRAW]:
            self.status = self.DRAW
        elif self.status == self.FOUL:
            pass
        else:
            self.status = self.LOSE
        return self.status

    def foul(self, agent=0):
        assert agent in [0, 1]
        self.status = self.FOUL
        self.foul_list[agent] = True
        return self.status

    def timeout(self):
        self.status = self.TIMEOUT
        return self.status

    def reset(self):
        self.status = self.IDLE
        self.foul_list = [False, False]
        return self.status

    def __eq__(self, __value: object) -> bool:
        return self.status == __value

    def __ne__(self, __value: object) -> bool:
        return self.status != __value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.status, self.values[self.status]


class FencerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "fencer.xml",
        save_model_dir: str = "models",
        method=PPO,
        device: str = "cuda",
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 0.1,
        first_state_step: int = 5e5,
        second_state_method: str = "alter",
        second_state_model: str = None,
        alter_state_step: int = 5e4,
        wandb_log: bool = False,
        enable_random: bool = False,
        version: str = "v2",
        opponent_version: str = "v2",
        time_limit: int = 1900,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_near_weight,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )
        print("-------------------------------------")
        print("ENV: ", self.__class__.__name__)
        print("VERSION: ", version)
        print("opponent VERSION: ", opponent_version)
        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        obs_state_num = 59
        if version == "v1":
            """no time state"""
            obs_state_num = 59
        elif version == "v2":
            """add time state"""
            obs_state_num = 60
        self.version = version
        self.opponent_version = opponent_version
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_state_num,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self.wandb_log = wandb_log
        if self.wandb_log:
            self.log_info = [
                "reward",
                "reward_ctrl",
                "reward_near",
                "penalty_oppent_near",
                "eps_stepcnt",
                "win",
                "lose",
                "draw",
                "foul",
                "timeout",
            ]
            self.eps_info = np.array([0] * len(self.log_info), dtype=np.float32)
            self.eps_infos = deque(maxlen=100)
        """ 
        change action space to be half of the original
        action space, since we are only controlling one arm
        """
        ############################################
        # check action space
        ############################################
        assert self.action_space.shape and len(self.action_space.shape) >= 1
        env_action_space_shape = self.action_space.shape[0]
        self.env_action_space_shape = env_action_space_shape
        self.env_action_space = self.action_space
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[: env_action_space_shape // 2], high[: env_action_space_shape // 2]
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(" real action space:", self.env_action_space)
        print("agent action space:", self.action_space)
        # self.init_direction = np.array([1, 0, 0])
        ############################################
        # match related properties
        ############################################
        self.target_point = ["target1", "target2", "target0"]
        self.velocity_point = ["upper_arm", "elbow_flex", "forearm_roll", "forearm", "wrist_flex"]
        self.last_obs = np.zeros(2 * 3 * len(self.target_point), dtype=np.float32)
        self.last_obs_opponent = np.zeros(2 * 3 * len(self.target_point), dtype=np.float32)
        # self.target_weight = [1, 1, 1]
        self.attact_point = "sword_tip"
        self.center_point = "shoulder_pan"
        self.match_reward = {
            "win": 10,
            "lose": -10,
            "draw": 0,
            "foul": -5,
            "timeout": -5,
        }
        self.GAME_STATUS = GameStatus("Rules", ["IDLE", "WIN", "LOSE", "DRAW", "FOUL", "TIMEOUT"])
        self.match_color = {
            "IDLE": [COLORMAP["Gray"], COLORMAP["Gray"]],
            "WIN": [COLORMAP["Blue"], COLORMAP["Gray"]],
            "LOSE": [COLORMAP["Gray"], COLORMAP["Yellow"]],
            "DRAW": [COLORMAP["Green"], COLORMAP["Green"]],
            "FOUL": [COLORMAP["Red"], COLORMAP["Red"]],
            "TIMEOUT": [COLORMAP["Orange"], COLORMAP["Orange"]],
        }
        ############################################
        # target area related properties
        # agent: 0, opponent: 1
        ############################################
        self.target_geom = ["shoulder_pan", "shoulder_lift"]
        self.target_z_constraint_reference_point = ["target0", "target1"]  # [lower, upper]
        self.collide_dist_threshold = 2  # min should be 0.6
        self.z_nearness_threshold = 0.5  # min should be 0.3
        print("0 attact_geom_id: ", self.get_geom_id(f"{0}_{self.attact_point}"))
        print("0 target_geom_id: ", self.get_geom_id(f"{1}_{self.target_geom[0]}"))
        # print(self.data.geom(f"{0}_{self.attact_point}"))
        self.agent_nearness_threshold = 0.1  # can't be too large
        self.agent_nearness_reward_slope = self._reward_near_weight
        self.agent_nearness_reward_offset = 0.2
        # means when the opponent is @ 0.5, the penalty is penalty_threshold ## can't be too large
        self.oppent_nearness_threshold = 0.05
        self.oppent_nearness_penalty_threshold = -1.5
        self.oppent_nearness_exponential_coeff = -5
        ############################################
        # learning related properties
        ############################################
        self.step_count = 0
        self.first_state_step = int(first_state_step)
        self.last_model_update_step = 0
        self.save_model_dir = save_model_dir
        self.most_recent_file = None
        self.oppent_model = None
        self.method = method
        self.device = device
        if second_state_method not in ["alter", "manual"]:
            raise ValueError("second_state_method must be 'alter' or 'manual'")
        if second_state_method == "manual" and second_state_model is None:
            raise ValueError("You must specify the second_state_model when second_state_method is 'manual'")
        self.second_state_method = second_state_method
        self.alter_state_step = int(alter_state_step)
        if self.second_state_method == "manual":
            try:
                self.second_state_model_path = second_state_model
                self.second_state_model = self.method.load(second_state_model)
                self.second_state_model.set_env(self)
            except Exception as e:
                self.second_state_model_path = second_state_model
                dac_config = dacConfigSetup(game='Fencer', first_state_step=self.first_state_step, alter_state_step=self.alter_state_step, save_model_dir=self.save_model_dir)
                load_agent = ASquaredCPPOAgent(dac_config)
                load_agent.load(filename=second_state_model)
                self.second_state_model = load_agent
                # self.second_state_model.set_env(self)
            except Exception as e:
                print(e)
                raise ValueError(f"{second_state_model} not found, please check the path")
        # not used, override by the env wrapper
        self.truncated_step = int(time_limit)
        print("first_state_step: ", self.first_state_step)
        if second_state_method == "alter":
            print("alter_state_step: ", self.alter_state_step)
        print("truncated_step: ", self.truncated_step)
        ############################################
        # episode related properties
        ############################################
        self.enable_random = enable_random
        self.eps_stepcnt = 0
        self.eps_reward = 0
        # self.agent0_attacked = False
        # self.agent1_attacked = False
        self.init_extra_step_after_done = 30
        self.extra_step_after_done = self.init_extra_step_after_done

        print("extra_step_after_done: ", self.extra_step_after_done)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.fps = self.metadata["render_fps"]
        print("fps: ", self.fps)
        print("oppoent method: ", self.second_state_method)
        print("-------------------------------------")

    def get_geom_com(self, geom_name):
        return self.data.geom(geom_name).xpos

    def get_geom_id(self, geom_name):
        return self.data.geom(geom_name).id

    def get_target_z_constraint(self, agent=0):
        return [self.get_geom_com(f"{agent}_{name}") for name in self.target_z_constraint_reference_point]

    def control_reward(self, action):
        return -np.square(action).sum() * self._reward_control_weight

    def collide2target(self, agent=0):
        if self.data.contact is None:
            return False
        collision = False
        attactor = agent
        opponent = 1 - attactor
        z_constraint = self.get_target_z_constraint(opponent)
        attact_geom_id = self.get_geom_id(f"{attactor}_{self.attact_point}")
        target_geom_id = [self.get_geom_id(f"{opponent}_{name}") for name in self.target_geom]
        for contact in self.data.contact:
            geom1 = contact.geom1
            geom2 = contact.geom2
            # check if tip collide with target
            collision = geom1 == attact_geom_id and geom2 in target_geom_id
            collision = collision or (geom2 == attact_geom_id and geom1 in target_geom_id)
            # if collision:
            #     print("collide ",contact.dist)
            # check if in target area
            collision = collision and contact.pos[-1] > z_constraint[0][-1] and contact.pos[-1] < z_constraint[1][-1]
            if collision:
                if EPISODE_LOG:
                    print(f"agent{agent} collide with target @step={self.eps_stepcnt}")
                return True
        return False

    def calculate_nearness(self, agent=0):
        """
        return the nearness of the agent to the target

        @return:
        scalar_parameter, vec_parameter, center_vec

        @param:
        agent: 0 or 1
        @return:
        scalar_parameter: the sum of the distance from the agent to the target reference point
        vec_parameter: the sum of the vector from the agent to the target reference point
        center_vec: the vector from the agent to the center of the target (z truncated)
        """
        vec_parameter = np.array([0, 0, 0], dtype=np.float32)
        scalar_parameter = 0.0
        opponent = 1 - agent
        n = len(self.target_point)
        for point in self.target_point:
            rel_vec = self._get_rel_pos(
                self.get_geom_com(f"{agent}_{self.attact_point}"),
                self.get_geom_com(f"{opponent}_{point}"),
                agent,
            )
            vec_parameter += rel_vec / n
            scalar_parameter += np.linalg.norm(rel_vec) / n

        # center vec, z truncated
        center_vec = self.get_geom_com(f"{agent}_{self.center_point}") - self.get_geom_com(f"{opponent}_{self.attact_point}")
        center_vec[-1] = np.sign(center_vec[-1]) * max(0.0, abs(center_vec[-1]) - self.z_nearness_threshold)
        return scalar_parameter, vec_parameter, center_vec

    def calculate_velocity(self, agent=0):
        # not used
        vec_parameter = np.array([0, 0, 0], dtype=np.float32)
        scalar_parameter = 0.0
        opponent = 1 - agent
        n = len(self.target_point)
        for point in self.target_point:
            rel_vec = self._get_rel_pos(
                self.get_geom_com(f"{agent}_{self.attact_point}"),
                self.get_geom_com(f"{opponent}_{point}"),
                agent,
            )
            vec_parameter += rel_vec / n
            scalar_parameter += np.linalg.norm(rel_vec) / n

        # center vec, z truncated
        center_vec = self.get_geom_com(f"{agent}_{self.center_point}") - self.get_geom_com(f"{opponent}_{self.attact_point}")
        center_vec[-1] = np.sign(center_vec[-1]) * max(0.0, abs(center_vec[-1]) - self.z_nearness_threshold)
        return scalar_parameter, vec_parameter, center_vec

    def agent_nearness_reward(self, nearness):
        """
        return the nearness reward of the agent to the opponent

        @return:
        -max(0, slope * (nearness - threshold)) + offset
        """
        return -max(0, self.agent_nearness_reward_slope * (nearness - self.agent_nearness_threshold)) + self.agent_nearness_reward_offset

    def oppent_nearness_penalty(self, nearness, truncated=True):
        """
        return the penalty of the nearness of the opponent to the agent

        @return:
        max: np.exp(exp_coeff * (-nearness_threshold))
        max(truncated): threshold
        """
        threshold = self.oppent_nearness_penalty_threshold
        nearness_threshold = self.oppent_nearness_threshold
        exp_coeff = self.oppent_nearness_exponential_coeff
        if truncated:
            return threshold * min(1, np.exp(max(-5, exp_coeff * (nearness - nearness_threshold))))
        return threshold * np.exp(max(-5, exp_coeff * (nearness - nearness_threshold)))

    def outOfArena(self, agent=0):
        border0_vec = self.get_geom_com(f"{agent}_{self.center_point}") - self.get_geom_com(f"0_indicator")
        border1_vec = self.get_geom_com(f"{agent}_{self.center_point}") - self.get_geom_com(f"1_indicator")
        # if agent is at the same side of borders, then the dot product of the two vectors should be positive
        return border0_vec[0] * border1_vec[0] >= -0.2

    def step(self, action):
        self.eps_stepcnt += 1
        self.step_count += 1

        agent = 0
        opponent = 1 - agent
        reward_ctrl = self.control_reward(action) / 5
        # calculate nearness
        nearness_scalar_0, nearness_vec_0, center_vec_0 = self.calculate_nearness(agent)
        nearness_scalar_1, nearness_vec_1, center_vec_1 = self.calculate_nearness(opponent)
        # calculate handcraft nearness reward(&penalty)
        old_reward_near = self.agent_nearness_reward(nearness_scalar_0)
        old_penalty_oppent_near = self.oppent_nearness_penalty(nearness_scalar_1)  # reward dodge

        ############################################
        ##### DON'T MODIFY THE CODE BELOW HERE #####
        ############################################
        # change action space back to original, for the mujoco env
        temp_action_space = self.action_space
        self.action_space = self.env_action_space
        total_action = np.concatenate([action, self._get_opponent_action()])
        # print("total_action",total_action)
        self.do_simulation(total_action, self.frame_skip)
        # change action space back to model env (one model)
        self.action_space = temp_action_space
        ############################################
        # after taking action
        observation = self._get_obs(version=self.version)
        # calculate nearness
        nearness_scalar_0, nearness_vec_0, center_vec_0 = self.calculate_nearness(agent)
        nearness_scalar_1, nearness_vec_1, center_vec_1 = self.calculate_nearness(opponent)

        # test = self._get_rel_pos(self.get_geom_com(f"0_{self.attact_point}"),self.get_geom_com( f"1_{self.target_geom[0]}"), 0)
        # print(np.dot(vec_hat(nearness_vec_0),vec_hat(test)) )

        # calculate handcraft nearness reward(&penalty)
        reward_near = self.agent_nearness_reward(nearness_scalar_0)
        penalty_oppent_near = self.oppent_nearness_penalty(nearness_scalar_1)  # reward dodge

        # calculate actual reward based on the change of nearness
        reward_near = ((reward_near - old_reward_near) <= 0) * reward_near
        penalty_oppent_near = ((penalty_oppent_near - old_penalty_oppent_near) <= 0) * penalty_oppent_near

        if np.linalg.norm(center_vec_0) < self.collide_dist_threshold:
            if self.collide2target(0):  # attack success
                self.GAME_STATUS.agent_win()
        if np.linalg.norm(center_vec_1) < self.collide_dist_threshold:
            if self.collide2target(1):  # be attacked
                self.GAME_STATUS.oppent_win()
        if self.outOfArena(0):
            self.GAME_STATUS.foul(0)
        if self.outOfArena(1):
            self.GAME_STATUS.foul(1)
        if self.eps_stepcnt > self.truncated_step:
            self.GAME_STATUS.timeout()
        reward = reward_ctrl + reward_near + penalty_oppent_near
        # reward = 0
        done = False
        # print("COLOR:", self.model.geom_rgba[self.get_geom_id(f"{agent}_{self.attact_point}")])
        self.eps_reward += reward
        if self.GAME_STATUS != self.GAME_STATUS.IDLE:
            self.extra_step_after_done -= 1
            # print("COLOR:", self.model.geom_rgba[self.get_geom_id(f"{agent}_{self.attact_point}")])
            if self.extra_step_after_done < 0:
                done = True
                match_reward = 0
                if self.GAME_STATUS == self.GAME_STATUS.DRAW:
                    match_reward = self.match_reward["draw"]
                elif self.GAME_STATUS == self.GAME_STATUS.WIN:
                    match_reward = self.match_reward["win"]
                elif self.GAME_STATUS == self.GAME_STATUS.LOSE:
                    match_reward = self.match_reward["lose"]
                elif self.GAME_STATUS == self.GAME_STATUS.FOUL:
                    if self.GAME_STATUS.foul_list[0]:
                        match_reward = self.match_reward["foul"]
                    else:
                        match_reward = -self.match_reward["foul"]
                elif self.GAME_STATUS == self.GAME_STATUS.TIMEOUT:
                    match_reward = self.match_reward["timeout"]
                reward += match_reward
        if self.render_mode == "human":
            self.render()
            self.game_status_indicator(agent)
        info = {}
        # print(self.wandb_log)
        # print(self.step_count, len(self.eps_infos), self.eps_infos.maxlen)
        if self.wandb_log:
            self.eps_info += np.array(
                [
                    reward,
                    reward_ctrl,
                    reward_near,
                    penalty_oppent_near,
                    self.eps_stepcnt,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=np.float32,
            )
            if done:
                self.eps_info += np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        self.GAME_STATUS == self.GAME_STATUS.WIN,
                        self.GAME_STATUS == self.GAME_STATUS.LOSE,
                        self.GAME_STATUS == self.GAME_STATUS.DRAW,
                        self.GAME_STATUS == self.GAME_STATUS.FOUL,
                        self.GAME_STATUS == self.GAME_STATUS.TIMEOUT,
                    ],
                    dtype=np.float32,
                )
                self.eps_infos.append(self.eps_info)
                for i, key in enumerate(self.log_info):
                    info[key] = self.eps_info[i]
                self.eps_info = np.array([0] * len(self.eps_info), dtype=np.float32)
            if self.step_count % 1000 == 0 and len(self.eps_infos) == self.eps_infos.maxlen:
                avg_info = np.average(np.array(self.eps_infos), axis=0)
                # print("avg_info",avg_info)
                # print(self.eps_infos)
                temp_info = {}
                for i, key in enumerate(self.log_info):
                    temp_info[key] = avg_info[i]
                temp_info["step_count"] = self.step_count
                wandb.log(temp_info)

        return observation, reward, done, False, info

    def reset_model(self):
        # self.step_count = 0
        if EPISODE_LOG:
            print("reset model@ ", self.eps_stepcnt)
        qpos = self.init_qpos
        qvel = self.init_qvel
        if self.enable_random and self.step_count > self.first_state_step + self.alter_state_step:
            qpos += self.np_random.uniform(low=-0.5, high=0.5, size=len(self.init_qpos))
            qvel += self.np_random.uniform(low=-0.1, high=0.1, size=len(self.init_qvel))
        self.GAME_STATUS.reset()
        self.eps_reward = 0
        self.eps_stepcnt = 0
        self.extra_step_after_done = self.init_extra_step_after_done
        self.last_obs = np.zeros(2 * 3 * len(self.target_point), dtype=np.float32)
        self.last_obs_opponent = np.zeros(2 * 3 * len(self.target_point), dtype=np.float32)
        self.set_state(qpos, qvel)
        return self._get_obs(version=self.version)

    def _get_obs(self, agent=0, version="v1"):
        assert len(self.data.qpos) % 2 == 0
        if agent == 0:
            actuator_state = (0, len(self.data.qpos) // 2)
        else:
            actuator_state = (len(self.data.qpos) // 2, len(self.data.qpos))
        """ joint state """
        obs = np.concatenate(
            [
                self.data.qpos.flat[actuator_state[0] : actuator_state[1]],
                self.data.qvel.flat[actuator_state[0] : actuator_state[1]],
            ]
        )
        """add time state"""
        if version == "v2":
            obs = np.concatenate([obs, [self.eps_stepcnt]])
        """ tip state """
        obs = np.concatenate(
            [
                obs,
                self._get_rel_pos(
                    self.get_geom_com(f"{agent}_{self.center_point}"),
                    self.get_geom_com(f"{agent}_{self.attact_point}"),
                    agent,
                ),
            ]
        )
        """ opponent state relative to agent tip """
        opponent = 1 - agent
        for point in self.target_point:
            obs = np.concatenate(
                [
                    obs,
                    self._get_rel_pos(
                        self.get_geom_com(f"{agent}_{self.attact_point}"),
                        self.get_geom_com(f"{opponent}_{point}"),
                        agent,
                    ),
                ]
            )
        for point in self.target_point:
            obs = np.concatenate(
                [
                    obs,
                    self._get_rel_pos(
                        self.get_geom_com(f"{opponent}_{self.attact_point}"),
                        self.get_geom_com(f"{agent}_{point}"),
                        agent=agent,
                    ),
                ]
            )
        """ calculate the relative velocity of both agent and opponent """
        assert len(obs) > 2 * 3 * len(self.target_point)
        this_obs = obs[-2 * 3 * len(self.target_point) :]
        if agent == 1:
            obs = np.concatenate([obs, (this_obs - self.last_obs_opponent) * self.fps])
            self.last_obs_opponent = this_obs.copy()
        else:
            obs = np.concatenate([obs, (this_obs - self.last_obs) * self.fps])
            self.last_obs = this_obs.copy()
        # print("obs.shape",obs.shape) # (59,)
        return obs.astype(np.float32)

    def _get_obs_agent1(self):
        return self._get_obs(agent=1, version=self.opponent_version)

    def _get_rel_pos(self, base_pos, rel_pos, agent=0):
        rel_vector = rel_pos - base_pos
        if agent == 0:
            return rel_vector
        else:
            rel_vector[0] = -rel_vector[0]
            rel_vector[1] = -rel_vector[1]
            return rel_vector

    def _get_opponent_action(self):
        if self.second_state_method == "manual" and self.oppent_model is None:
            print("second_state_method loading: ", self.second_state_model_path)
            self.oppent_model = self.second_state_model
            if(self.method == ASquaredCPPOAgent):
                opp_action = self.oppent_model.record_step(self._get_obs_agent1())
            else:
                opp_action, _ = self.oppent_model.predict(
                self._get_obs_agent1(), deterministic=True)
            return opp_action
        if self.step_count < self.first_state_step:
            return np.zeros(self.env_action_space_shape // 2)
        elif self.step_count > self.last_model_update_step + self.alter_state_step:
            self.last_model_update_step = self.step_count
            if self.second_state_method == "alter":
                self.oppent_model = self.find_last_model()
            if self.oppent_model is None:
                return np.zeros(self.env_action_space_shape//2)
            if(self.method == ASquaredCPPOAgent):
                opp_action = self.oppent_model.record_step(self._get_obs_agent1())
            else:
                opp_action, _ = self.oppent_model.predict(
                self._get_obs_agent1(), deterministic=True)
            # print(opp_action)
            return opp_action
        elif self.oppent_model is not None:
            if(self.method == ASquaredCPPOAgent):
                opp_action = self.oppent_model.record_step(self._get_obs_agent1())
            else:
                opp_action, _ = self.oppent_model.predict(
                self._get_obs_agent1(), deterministic=True)
            # print(opp_action)
            return opp_action
        else:
            return np.zeros(self.env_action_space_shape // 2)

    def find_last_model(self):
        most_recent_file = None
        most_recent_time = 0
        # return None
        # iterate over the files in the directory using os.scandir
        for entry in os.scandir(self.save_model_dir):
            if entry.is_file():
                # get the modification time of the file using entry.stat().st_mtime_ns
                mod_time = entry.stat().st_mtime_ns
                if mod_time > most_recent_time:
                    # update the most recent file and its modification time
                    most_recent_file = entry.name
                    most_recent_time = mod_time
        if most_recent_file == self.most_recent_file:
            return self.oppent_model
        print("most_recent_file", most_recent_file)
        print(f"restore from {self.save_model_dir}/{most_recent_file}")
        self.most_recent_file = most_recent_file
        print(self.method)
        if(self.method == ASquaredCPPOAgent):
            dac_config = dacConfigSetup(game='Fencer', first_state_step=self.first_state_step, alter_state_step=self.alter_state_step, save_model_dir=self.save_model_dir)
            load_agent = ASquaredCPPOAgent(dac_config)
            filename = f"{self.save_model_dir}/{most_recent_file}"
            if "model" in filename:
                filename = filename[:-6]
            elif "stat" in filename:
                filename = filename[:-5]
            load_agent.load(filename)
            return load_agent
        return self.method.load(f"{self.save_model_dir}/{most_recent_file}", device=self.device, verbose=0)

    def game_status_indicator(self, agent=0):
        status, statusName = self.GAME_STATUS()
        oppent = 1 - agent
        agent_indicator_id = self.get_geom_id(f"{agent}_indicator")
        oppent_indicator_id = self.get_geom_id(f"{oppent}_indicator")
        self.model.geom_rgba[agent_indicator_id] = self.match_color[statusName][0]
        self.model.geom_rgba[oppent_indicator_id] = self.match_color[statusName][1]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
