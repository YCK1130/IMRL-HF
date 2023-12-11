__credits__ = ["Kallinteris-Andreas"]

from typing import Dict
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
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

EPISODE_LOG = False

def vec_hat(vec):
    return vec / np.linalg.norm(vec)
class GameStatus():
    IDLE = 0
    WIN = 1
    LOSE = 2
    DRAW = 3
    FOUL = 4
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.status = self.IDLE
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
    def foul(self):
        self.status = self.FOUL
        return self.status
    def reset(self):
        self.status = self.IDLE
        return self.status
    def __eq__(self, __value: object) -> bool:
        return self.status == __value
    def __ne__(self, __value: object) -> bool:
        return self.status != __value

class FencerEnv(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    "Pusher" is a multi-jointed robot arm which is very similar to that of a human.
    The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
    The robot consists of shoulder, elbow, forearm, and wrist joints.

    Gymnasium includes the following versions of the environment:

    | Environment               | Binding         | Notes                                       |
    | ------------------------- | --------------- | ------------------------------------------- |
    | Pusher-v5                 | `mujoco=>2.3.3` | Recommended (most features, the least bugs) |
    | Pusher-v4                 | `mujoco=>2.1.3` | Maintained for reproducibility              |
    | Pusher-v2                 | `mujoco-py`     | Maintained for reproducibility              |

    For more information see section "Version History".


    ## Action Space
    The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Rotation of the panning the shoulder                               | -2          | 2           | r_shoulder_pan_joint             | hinge | torque (N m) |
    | 1   | Rotation of the shoulder lifting joint                             | -2          | 2           | r_shoulder_lift_joint            | hinge | torque (N m) |
    | 2   | Rotation of the shoulder rolling joint                             | -2          | 2           | r_upper_arm_roll_joint           | hinge | torque (N m) |
    | 3   | Rotation of hinge joint that flexed the elbow                      | -2          | 2           | r_elbow_flex_joint               | hinge | torque (N m) |
    | 4   | Rotation of hinge that rolls the forearm                           | -2          | 2           | r_forearm_roll_joint             | hinge | torque (N m) |
    | 5   | Rotation of flexing the wrist                                      | -2          | 2           | r_wrist_flex_joint               | hinge | torque (N m) |
    | 6   | Rotation of rolling the wrist                                      | -2          | 2           | r_wrist_roll_joint               | hinge | torque (N m) |


    ## Observation Space
    Observations consist of

    - Angle of rotational joints on the pusher
    - Angular velocities of rotational joints on the pusher
    - The coordinates of the fingertip of the pusher
    - The coordinates of the object to be moved
    - The coordinates of the goal position

    The observation is a `Box(-Inf, Inf, (23,), float64)` where the elements correspond to the table below.
    An analogy can be drawn to a human arm in order to help understand the state space, with the words flex and roll meaning the
    same as human joints.

    | Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Type (Unit)              |
    | --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
    | 0   | Rotation of the panning the shoulder                     | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angle (rad)              |
    | 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angle (rad)              |
    | 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angle (rad)              |
    | 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | r_elbow_flex_joint               | hinge    | angle (rad)              |
    | 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | r_forearm_roll_joint             | hinge    | angle (rad)              |
    | 5   | Rotation of flexing the wrist                            | -Inf | Inf | r_wrist_flex_joint               | hinge    | angle (rad)              |
    | 6   | Rotation of rolling the wrist                            | -Inf | Inf | r_wrist_roll_joint               | hinge    | angle (rad)              |
    | 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | r_shoulder_pan_joint             | hinge    | angular velocity (rad/s) |
    | 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | r_shoulder_lift_joint            | hinge    | angular velocity (rad/s) |
    | 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | r_upper_arm_roll_joint           | hinge    | angular velocity (rad/s) |
    | 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | r_elbow_flex_joint               | hinge    | angular velocity (rad/s) |
    | 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | r_forearm_roll_joint             | hinge    | angular velocity (rad/s) |
    | 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | r_wrist_flex_joint               | hinge    | angular velocity (rad/s) |
    | 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | r_wrist_roll_joint               | hinge    | angular velocity (rad/s) |
    | 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | tips_arm                         | slide    | position (m)             |
    | 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidex)              | slide    | position (m)             |
    | 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (obj_slidey)              | slide    | position (m)             |
    | 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                           | cylinder | position (m)             |
    | 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidex)               | slide    | position (m)             |
    | 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (goal_slidey)               | slide    | position (m)             |
    | 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                             | sphere   | position (m)             |


    ## Rewards
    The total reward is: ***reward*** *=* *reward_dist + reward_ctrl + reward_near*.

    - *reward_near*:
    This reward is a measure of how far the *fingertip* of the pusher (the unattached end) is from the object,
    with a more negative value assigned for when the pusher's *fingertip* is further away from the target.
    It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
    where $w_{near}$ is the `reward_near_weight`.
    - *reward_dist*:
    This reward is a measure of how far the object is from the target goal position,
    with a more negative value assigned if the object that is further away from the target.
    It is $-w_{dist} \|(P_{object} - P_{target})\|_2$.
    where $w_{dist}$ is the `reward_dist_weight`.
    - *reward_control*:
    A negative reward to penalize the pusher for taking actions that are too large.
    It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
    where $w_{control}$ is the `reward_control_weight`.

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state of the Pusher arm is $0_{6}$.
    The initial position state of the object is $\mathcal{U}_{[[-0.3, -0.2], [0, 0.2]]}$.
    The position state of the goal is (permanently) $[0.45, -0.05, -0.323]$.
    The initial velocity state of the Pusher arm is $\mathcal{U}_{[-0.005 \times 1_{6}, 0.005 \times 1_{6}]}$.
    The initial velocity state of the object is $0_2$.
    The velocity state of the goal is (permanently) $0_3$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution.

    Note that the initial position state of the object is sampled until it's distance to the goal is $ > 0.17 m$.

    The default frame rate is 5, with each frame lasting for 0.01, so *dt = 5 * 0.01 = 0.05*.


    ## Episode End
    #### Termination
    The Pusher never terminates.

    #### Truncation
    The default duration of an episode is 100 timesteps


    ## Arguments
    Pusher provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Pusher-v5', xml_file=...)
    ```

    | Parameter               | Type       | Default      |Description                                               |
    |-------------------------|------------|--------------|----------------------------------------------------------|
    | `xml_file`              | **str**    |`"pusher.xml"`| Path to a MuJoCo model                                   |
    | `reward_near_weight`    | **float**  | `0.5`        | Weight for *reward_near* term (see section on reward)    |
    | `reward_dist_weight`    | **float**  | `1`          | Weight for *reward_dist* term (see section on reward)    |
    | `reward_control_weight` | **float**  | `0.1`        | Weight for *reward_control* term (see section on reward) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Added `xml_file` argument.
        - Added `reward_near_weight`, `reward_dist_weight`, `reward_control_weight` arguments, to configure the reward function (defaults are effectively the same as in `v4`).
        - Fixed `info["reward_ctrl"]` being not being multiplied by the reward weight.
        - Added `info["reward_near"]` which is equal to the reward term `reward_near`.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks (not including pusher, which has a max_time_steps of 100). Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0).
    """

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
        device: str = 'cuda',
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 0.1,
        first_state_step: int = 5e5,
        alter_state_step: int = 5e4,
        wandb_log: bool = False,
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
        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf,
                                shape=(30,), dtype=np.float32)

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
            self.log_info = ["reward", "reward_ctrl", "reward_near", "penalty_oppent_near", "eps_stepcnt","win","lose","draw","foul"]
            self.eps_info = np.array([0]*len(self.log_info),dtype=np.float32)
            self.eps_infos = deque(maxlen=100)
        ''' 
        change action space to be half of the original
        action space, since we are only controlling one arm
        '''
        ############################################
        ### check action space
        ############################################
        assert self.action_space.shape and len(self.action_space.shape) >= 1
        env_action_space_shape = self.action_space.shape[0]
        self.env_action_space_shape = env_action_space_shape
        self.env_action_space = self.action_space
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:env_action_space_shape //
                        2], high[:env_action_space_shape//2]
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(" real action space:",self.env_action_space)
        print("agent action space:",self.action_space)
        # self.init_direction = np.array([1, 0, 0])
        ############################################
        ### match related properties
        ############################################
        self.target_point = ["target1", "target2", "target0"]
        # self.target_weight = [1, 1, 1]
        self.attact_point = "sword_tip"
        self.center_point = "shoulder_pan"
        self.match_reward = {
            "win": 2,
            "lose": -2,
            "draw": 0,
            "foul": -5,
        }
        self.GAME_STATUS = GameStatus('Rules', ['WIN', 'LOSE', 'DRAW', 'FOUL'])
        ############################################
        ### target area related properties
        ### agent: 0, opponent: 1
        ############################################
        self.target_geom = ["shoulder_pan", "shoulder_lift"]
        self.target_z_constraint_reference_point = ["target0", "target1"] # [lower, upper]
        self.collide_dist_threshold = 2 # min should be 0.6
        self.z_nearness_threshold = 0.5 # min should be 0.3
        print("0 attact_geom_id: ",self.get_geom_id(f"{0}_{self.attact_point}"))
        print("0 target_geom_id: ",self.get_geom_id(f"{1}_{self.target_geom[0]}"))
        # print(self.data.geom(f"{0}_{self.attact_point}"))
        self.agent_nearness_threshold = 0.1 ## can't be too large
        self.agent_nearness_reward_slope = self._reward_near_weight
        self.agent_nearness_reward_offset = 0.2
        self.oppent_nearness_threshold = 0.1 # means when the opponent is @ 0.5, the penalty is penalty_threshold ## can't be too large
        self.oppent_nearness_penalty_threshold = -2
        self.oppent_nearness_exponential_coeff = -5
        ############################################
        ### learning related properties
        ############################################
        self.step_count = 0
        self.first_state_step = int(first_state_step)
        self.alter_state_step = int(alter_state_step)
        self.truncated_step = int(2500) ## not used, override by the env wrapper
        print("first_state_step: ", self.first_state_step)
        print("alter_state_step: ", self.alter_state_step)
        print("truncated_step: ", self.truncated_step)
        self.last_model_update_step = 0
        self.save_model_dir = save_model_dir
        self.most_recent_file = None
        self.oppent_model = None
        self.method = method
        self.device = device
        ############################################
        ### episode related properties
        ############################################
        self.eps_stepcnt = 0
        self.eps_reward = 0
        # self.agent0_attacked = False
        # self.agent1_attacked = False
        self.init_extra_step_after_done = 30
        self.extra_step_after_done = self.init_extra_step_after_done
        print("extra_step_after_done: ",self.extra_step_after_done)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def get_geom_com(self, geom_name):
        return self.data.geom(geom_name).xpos
    def get_geom_id(self, geom_name):
        return self.data.geom(geom_name).id
    def get_target_z_constraint(self, agent=0):
        return [self.get_geom_com(f"{agent}_{name}") for name in self.target_z_constraint_reference_point]
    def control_reward(self, action):
        return -np.square(action).sum() * self._reward_control_weight
    def collide2target(self, agent=0):
        if(self.data.contact is None): return False
        collision = False
        attactor = agent
        opponent = 1 - attactor
        z_constraint = self.get_target_z_constraint(opponent)
        attact_geom_id = self.get_geom_id(f"{attactor}_{self.attact_point}")
        target_geom_id = [self.get_geom_id(f"{opponent}_{name}") for name in self.target_geom]
        for contact in self.data.contact:
            geom1 = contact.geom1
            geom2 = contact.geom2
            ### check if tip collide with target
            collision =  (geom1 == attact_geom_id and geom2 in target_geom_id)
            collision = collision or (geom2 == attact_geom_id and geom1 in target_geom_id)
            # if collision:
            #     print("collide ",contact.dist)
            ### check if in target area
            collision = collision and contact.pos[-1] > z_constraint[0][-1] and contact.pos[-1] < z_constraint[1][-1]
            if collision:
                if EPISODE_LOG: print(f"agent{agent} collide with target @step={self.eps_stepcnt}")
                return True
        return False
    def calculate_nearness(self, agent=0):
        '''
        return the nearness of the agent to the target

        @return:
        scalar_parameter, vec_parameter, center_vec
        
        @param:
        agent: 0 or 1
        @return:
        scalar_parameter: the sum of the distance from the agent to the target reference point
        vec_parameter: the sum of the vector from the agent to the target reference point
        center_vec: the vector from the agent to the center of the target (z truncated)
        '''
        vec_parameter = np.array([0,0,0],dtype=np.float32)
        scalar_parameter = 0.0
        opponent = 1 - agent
        n = len(self.target_point)
        for point in self.target_point:
            rel_vec = self._get_rel_pos(
                self.get_geom_com(f"{agent}_{self.attact_point}"),
                self.get_geom_com(f"{opponent}_{point}"),
                agent
            )
            vec_parameter += rel_vec / n
            scalar_parameter += np.linalg.norm(rel_vec) / n
        
        ### center vec, z truncated
        center_vec = self.get_geom_com(
            f"{agent}_{self.center_point}") - self.get_geom_com(f"{opponent}_{self.attact_point}")
        center_vec[-1] = np.sign(center_vec[-1]) * max(0.0, abs(center_vec[-1]) - self.z_nearness_threshold)
        return scalar_parameter, vec_parameter, center_vec
    def agent_nearness_reward(self, nearness):
        '''
        return the nearness reward of the agent to the opponent

        @return:
        -max(0, slope * (nearness - threshold)) + offset
        '''
        return -max(0, self.agent_nearness_reward_slope * (nearness - self.agent_nearness_threshold)) + self.agent_nearness_reward_offset
    def oppent_nearness_penalty(self, nearness, truncated=True):
        '''
        return the penalty of the nearness of the opponent to the agent

        @return:
        max: np.exp(exp_coeff * (-nearness_threshold))
        max(truncated): threshold
        '''
        threshold = self.oppent_nearness_penalty_threshold
        nearness_threshold = self.oppent_nearness_threshold
        exp_coeff = self.oppent_nearness_exponential_coeff
        if truncated:
            return threshold*min(1,np.exp(max(-5,exp_coeff * (nearness - nearness_threshold))))
        return threshold*np.exp(max(-5,exp_coeff * (nearness - nearness_threshold)))
    def step(self, action):
        self.eps_stepcnt += 1
        self.step_count += 1

        agent = 0
        opponent = 1 - agent
        reward_ctrl = self.control_reward(action) / 5
        ### calculate nearness
        nearness_scalar_0, nearness_vec_0, center_vec_0 = self.calculate_nearness(agent)
        nearness_scalar_1, nearness_vec_1, center_vec_1 = self.calculate_nearness(opponent)
        ### calculate handcraft nearness reward(&penalty)
        old_reward_near = self.agent_nearness_reward(nearness_scalar_0)
        old_penalty_oppent_near = self.oppent_nearness_penalty(nearness_scalar_1) # reward dodge
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
        ### after taking action
        observation = self._get_obs()

        ### calculate nearness
        nearness_scalar_0, nearness_vec_0, center_vec_0 = self.calculate_nearness(agent)
        nearness_scalar_1, nearness_vec_1, center_vec_1 = self.calculate_nearness(opponent)

        # test = self._get_rel_pos(self.get_geom_com(f"0_{self.attact_point}"),self.get_geom_com( f"1_{self.target_geom[0]}"), 0)
        # print(np.dot(vec_hat(nearness_vec_0),vec_hat(test)) )

        ### calculate handcraft nearness reward(&penalty)
        reward_near = self.agent_nearness_reward(nearness_scalar_0)
        penalty_oppent_near = self.oppent_nearness_penalty(nearness_scalar_1) # reward dodge

        ### calculate actual reward based on the change of nearness
        reward_near = ((reward_near - old_reward_near) <= 0)*reward_near
        penalty_oppent_near = ((penalty_oppent_near - old_penalty_oppent_near) <= 0)*penalty_oppent_near

        if np.linalg.norm(center_vec_0) < self.collide_dist_threshold:
            if self.collide2target(0):  # attack success
                self.GAME_STATUS.agent_win()
        if np.linalg.norm(center_vec_1) < self.collide_dist_threshold:
            if self.collide2target(1):  # be attacked
                self.GAME_STATUS.oppent_win()

        reward = reward_ctrl + reward_near + penalty_oppent_near
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
                    match_reward = self.match_reward["foul"]
                reward += match_reward
        if self.render_mode == "human":
            self.render()
            self.game_status_indicator(agent)
        info = {}
        if self.wandb_log:
            self.eps_info += np.array([reward,
                                        reward_ctrl,
                                        reward_near, 
                                        penalty_oppent_near, 
                                        self.eps_stepcnt,
                                        0, 0, 0, 0
                                        ],dtype=np.float32)
            if done:
                if self.step_count > self.first_state_step - 10*1000:
                    self.eps_info += np.array([0, 0, 0, 0, 0,
                                            self.GAME_STATUS == self.GAME_STATUS.WIN,
                                            self.GAME_STATUS == self.GAME_STATUS.LOSE,
                                            self.GAME_STATUS == self.GAME_STATUS.DRAW,
                                            self.GAME_STATUS == self.GAME_STATUS.FOUL
                                            ],dtype=np.float32)
                self.eps_infos.append(self.eps_info)
                for i, key in enumerate(self.log_info):
                    info[key] = self.eps_info[i]
                self.eps_info = np.array([0]*len(self.eps_info),dtype=np.float32)
            if self.step_count%1000==0 and len(self.eps_infos) == self.eps_infos.maxlen: 
                avg_info = np.average(np.array(self.eps_infos),axis=0)
                # print("avg_info",avg_info)
                # print(self.eps_infos)
                temp_info = {}
                for i, key in enumerate(self.log_info):
                    temp_info[key] = avg_info[i]
                temp_info['step_count'] = self.step_count
                wandb.log(temp_info)

        return observation, reward, done, self.eps_stepcnt > self.truncated_step, info

    def reset_model(self):
        # self.step_count = 0
        if EPISODE_LOG: print("reset model@ ",self.eps_stepcnt)
        qpos = self.init_qpos
        qvel = self.init_qvel
        # if self.step_count > self.first_state_step + self.alter_state_step:
        #     qpos += self.np_random.uniform(low=-0.1, high=0.1,size=len(self.init_qpos))
        #     qvel += self.np_random.uniform(low=-0.1, high=0.1,size=len(self.init_qvel))
        self.GAME_STATUS.reset()
        self.eps_reward = 0
        self.eps_stepcnt = 0
        self.extra_step_after_done = self.init_extra_step_after_done
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self, agent=0):

        assert len(self.data.qpos) % 2 == 0
        if agent == 0:
            actuator_state = (0, len(self.data.qpos)//2)
        else:
            actuator_state = (len(self.data.qpos)//2, len(self.data.qpos))
        ''' joint state '''
        obs = np.concatenate([self.data.qpos.flat[actuator_state[0]:actuator_state[1]],
                              self.data.qvel.flat[actuator_state[0]:actuator_state[1]],])
        ''' tip state '''
        obs = np.concatenate([obs, self._get_rel_pos(
            self.get_geom_com(f"{agent}_{self.center_point}"),
            self.get_geom_com(f"{agent}_{self.attact_point}"),
            agent
        )])
        ''' opponent state relative to agent tip '''
        opponent = 1-agent
        for point in self.target_point:
            obs = np.concatenate([obs, self._get_rel_pos(
                self.get_geom_com(f"{agent}_{self.attact_point}"),
                self.get_geom_com(f"{opponent}_{point}"),
                agent
            )])
        # print("obs.shape",obs.shape) # (32,)
        return obs.astype(np.float32)

    def _get_obs_agent1(self):
        return self._get_obs(agent=1)

    def _get_rel_pos(self, base_pos, rel_pos, agent=0):
        rel_vector = rel_pos - base_pos
        if agent == 0:
            return rel_vector
        else:
            rel_vector[0] = -rel_vector[0]
            rel_vector[1] = -rel_vector[1]
            return rel_vector

    def _get_opponent_action(self):
        if self.step_count < self.first_state_step:
            return np.zeros(self.env_action_space_shape//2)
        elif self.oppent_model is None or self.step_count > self.last_model_update_step + self.alter_state_step:
            self.last_model_update_step = self.step_count
            self.oppent_model = self.find_last_model()
            if self.oppent_model is None:
                return np.zeros(self.env_action_space_shape//2)
            opp_action, _ = self.oppent_model.predict(
                self._get_obs_agent1(), deterministic=True)
            # print(opp_action)
            return opp_action
        elif self.oppent_model is not None:
            opp_action, _ = self.oppent_model.predict(
                self._get_obs_agent1(), deterministic=True)
            # print(opp_action)
            return opp_action
        else:
            return np.zeros(self.env_action_space_shape//2)

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
        return self.method.load(f"{self.save_model_dir}/{most_recent_file}", device=self.device, verbose=0)
    def game_status_indicator(self, agent=0):
        status = self.GAME_STATUS.status
        oppent = 1 - agent
        agent_indicator_id = self.get_geom_id(f"{agent}_indicator")
        oppent_indicator_id = self.get_geom_id(f"{oppent}_indicator")
        if status == self.GAME_STATUS.IDLE:
            self.model.geom_rgba[agent_indicator_id] = np.array([0.2, 0.2, 0.2, 1])
            self.model.geom_rgba[oppent_indicator_id] = np.array([0.2, 0.2, 0.2, 1])
        elif status == self.GAME_STATUS.WIN:
            self.model.geom_rgba[agent_indicator_id] = np.array([0, 1, 0, 1]) # green
        elif status == self.GAME_STATUS.LOSE:
            self.model.geom_rgba[oppent_indicator_id] = np.array([1, 1, 0, 1]) # yellow
        elif status == self.GAME_STATUS.DRAW:
            self.model.geom_rgba[agent_indicator_id] = np.array([0, 0, 1, 1]) # blue
            self.model.geom_rgba[oppent_indicator_id] = np.array([0, 0, 1, 1]) # blue
        elif status == self.GAME_STATUS.FOUL:
            self.model.geom_rgba[agent_indicator_id] = np.array([1, 0, 0, 1]) # red
            self.model.geom_rgba[oppent_indicator_id] = np.array([1, 0, 0, 1]) # red
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]