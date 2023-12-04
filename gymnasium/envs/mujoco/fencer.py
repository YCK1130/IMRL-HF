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


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


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
        first_state_step: int = 1e3,
        alter_state_step: int = 5e2,
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

        ''' 
        change action space to be half of the original
        action space, since we are only controlling one arm
        '''
        # check action space
        assert self.action_space.shape and len(self.action_space.shape) >= 1
        env_action_space_shape = self.action_space.shape[0]
        self.env_action_space_shape = env_action_space_shape
        self.env_action_space = self.action_space
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:env_action_space_shape //
                        2], high[:env_action_space_shape//2]
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        print(self.env_action_space)
        print(self.action_space)
        # print(self.observation_space)

        # print(len(self.data.qpos))
        # print(self.data)

        # self.init_direction = np.array([1, 0, 0])
        self.target_point = ["target1", "target2", "target0"]
        self.target_weight = [1, 1, 1]
        self.attact_point = "sword_tip"
        self.center_point = "shoulder_pan"

        self.step_count = 0
        self.first_state_step = int(first_state_step)
        self.alter_state_step = int(alter_state_step)
        print("first_state_step: ", self.first_state_step)
        print("alter_state_step: ", self.alter_state_step)
        self.last_model_update_step = 0
        self.save_model_dir = save_model_dir
        self.most_recent_file = None
        self.oppent_model = None
        self.method = method
        self.device = device
        self.eps_stepcnt = 0
        self.eps_reward = 0
        self.agent0_attacked = False
        self.agent1_attacked = False
        self.init_extra_step_after_done = 60
        self.extra_step_after_done = self.init_extra_step_after_done
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

    def step(self, action):
        attacked = False
        self.eps_stepcnt += 1
        self.step_count += 1
        vecs_1 = []
        vecs_2 = []
        agent = 0
        opponent = 1 - agent
        for i in range(len(self.target_point)):
            point = self.target_point[i]
            weight = 1
            if self.eps_stepcnt > 200:
                weight = 2
            vecs_1 += [weight*self._get_rel_pos(
                self.get_geom_com(f"{agent}_{self.attact_point}"),
                self.get_geom_com(f"{opponent}_{point}"),
                agent
            )]
            vecs_2 += [weight*self._get_rel_pos(
                self.get_geom_com(f"{opponent}_{self.attact_point}"),
                self.get_geom_com(f"{agent}_{point}"),
                opponent
            )]
        penalty_1 = self.get_geom_com(
            f"{agent}_{self.center_point}") - self.get_geom_com(f"{opponent}_{self.attact_point}")
        penalty_2 = self.get_geom_com(
            f"{opponent}_{self.center_point}") - self.get_geom_com(f"{agent}_{self.attact_point}")
        penalty_1[-1] = max(0.0, abs(penalty_1[-1])-0.3)
        penalty_2[-1] = max(0.0, abs(penalty_2[-1])-0.3)
        # vec_9 = self.get_geom_com("1_sword_tip")-self.get_body_com()
        reward_match = 0

        reward_near_mirror = 0
        reward_near = 0
        for i in range(3):
            reward_near_mirror += - \
                np.linalg.norm(vecs_1[i]) * self._reward_near_weight
            reward_near += - \
                np.linalg.norm(vecs_2[i]) * self._reward_near_weight
        reward_near /= 3
        reward_near_mirror /= 3

        reward_ctrl = -np.square(action).sum() * \
            self._reward_control_weight*0.5
        penalty_far_mirror = - \
            np.linalg.norm(penalty_1) * self._reward_dist_weight
        penalty_far = - np.linalg.norm(penalty_2) * self._reward_dist_weight
        # print(penalty_far)
        if penalty_far > -0.11:
            penalty_far = 0
            self.agent1_attacked = True  # attack success
            # print(f"agent {opponent} ATTACKED by {agent}")
        elif penalty_far > -1.2:
            penalty_far = 0
        if penalty_far_mirror > -0.11:
            self.agent0_attacked = True  # attacked

            # print(f"agent {agent} ATTACKED by {opponent}")
        penalty_far_mirror = 0

        # change action space back to original, for the mujoco env
        temp_action_space = self.action_space
        self.action_space = self.env_action_space
        total_action = np.concatenate([action, self._get_opponent_action()])
        # print("total_action",total_action)
        self.do_simulation(total_action, self.frame_skip)
        # change action space back to model env (one model)
        self.action_space = temp_action_space

        observation = self._get_obs()
        reward = reward_ctrl + reward_near + penalty_far_mirror + penalty_far
        info = {
            # "reward_near_mirror": reward_near_mirror,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
            "penalty_far_mirror": penalty_far_mirror,
            "penaly_far": penalty_far
        }
        done = False
        self.eps_reward += reward
        if self.agent0_attacked or self.agent1_attacked:
            self.extra_step_after_done -= 1
            if self.extra_step_after_done < 0:
                self.extra_step_after_done = self.init_extra_step_after_done
                done = True
                wandb.log({"eps_reward": self.eps_reward/self.eps_stepcnt})
                self.eps_reward = 0
                self.eps_stepcnt = 0

                reward += int(self.agent1_attacked) - \
                    5*int(self.agent0_attacked)
        if self.render_mode == "human":
            self.render()
        wandb.log({"reward": reward})
        return observation, reward, done, False, info

    def reset_model(self):
        # self.step_count = 0
        # print("reset model@ ",self.step_count)
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.agent0_attacked = False
        self.agent1_attacked = False
        self.init_extra_step_after_done = 60
        self.extra_step_after_done = self.init_extra_step_after_done
        # self.goal_pos = np.asarray([0, 0])
        # while True:
        #     self.cylinder_pos = np.concatenate(
        #         [
        #             self.np_random.uniform(low=-0.3, high=0, size=1),
        #             self.np_random.uniform(low=-0.2, high=0.2, size=1),
        #         ]
        #     )
        #     if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
        #         break

        # qpos[-4:-2] = self.cylinder_pos
        # qpos[-2:] = self.goal_pos
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=-0.005, high=0.005, size=self.model.nv
        # )
        # qvel[-4:] = 0
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
            return - rel_vector

    def _get_opponent_action(self):
        if self.step_count < self.first_state_step:
            return np.zeros(self.env_action_space_shape//2)
        elif self.step_count > self.last_model_update_step + self.alter_state_step and self.oppent_model is not None:
            # print("update oppent model")
            self.last_model_update_step = self.step_count
            self.oppent_model = self.find_last_model()
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
        print(f"{self.save_model_dir}/{most_recent_file}")
        self.most_recent_file = most_recent_file
        return self.method.load(f"{self.save_model_dir}/{most_recent_file}", device=self.device, verbose=0)
