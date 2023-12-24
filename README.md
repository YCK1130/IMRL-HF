# Hierarchy Fencer

This repository is the source code for _A Two-Step Approach for Physically Competitive Sports: A Case Study on Fencing_. It is built on top of [Gymnasium] (https://github.com/Farama-Foundation/Gymnasium) by the Farama Foundation and branch DAC of [DeepRL] (https://github.com/ShangtongZhang/DeepRL/tree/DAC) by Shangtong Zhang.

## Installation

After cloning the repository, go into the directory and use the command below to install neccessary libraries:

```
pip install -r requirements.txt
pip install -e .
```

## Training

This repository currently supports training methods of SAC, TD3, A2C, PPO, and DAC ([Double Actor-Critic] (https://arxiv.org/abs/1904.12691) by S. Zhang). All algorithms except for DAC are also supported by StableBaselines3.

```
cd gymnasium/

# Step 1

# Algorithms in StableBaselines3
python sb3.py AlgorithmName -t
# 'AlgorithmName' can be SAC, TD3, PPO, or A2C

# Double Actor-Critics
python DAC.py DAC -t

# Step 2 (The two models should be trained by same method)

# Algorithms in StableBaselines3
python sb3.py AlgorithmName -t -s2 /path/to/second/model

# Double Actor-Critics
python DAC.py DAC -t -s2 /path/to/second/model
```

In the last command, the file extension of the path must be removed. For example, models/0000_test/DAC_100000 is used instead of models/0000_test/DAC_100000.model .

The model for the command above will be stored in gymnasium/models/\[run_id\]/\[algorithm_name\]\_\[steps\].zip or gymnasium/models/\[run_id\]/\[algorithm_name\]\_\[steps\].model .

## Testing

```
# Play with itself (with first stage stationary)

# Algorithms in StableBaselines3
python sb3.py AlgorithmName -s /path/to/model

# Double Actor-Critics
python DAC.py DAC -s /path/to/model

# Play with another model (The two models should be trained by same method)

# Algorithms in StableBaselines3
python sb3.py AlgorithmName -s /path/to/model -s2 /path/to/second/model

# Double Actor-Critics
python DAC.py DAC -s /path/to/model -s2 /path/to/second/model
```

In the last command, the file extension of the path must also be removed.
