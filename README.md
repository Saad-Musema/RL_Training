# CartPole Reinforcement Learning with DQN
## Overview

This project trains a Deep Q-Network (DQN) agent to balance a pole on a cart in the CartPole-v1 environment from OpenAI Gym. The agent learns through reinforcement learning and is evaluated after training.
Features

    Uses Stable Baselines3 to train a DQN model.
    Trains the model for 100,000 timesteps.
    Saves and loads the trained model.
    Tests the trained model with real-time visualization.
    Displays the final score after testing.

## Installation

Ensure you have Python installed, then install the required dependencies:

pip install gym stable-baselines3

Usage

Run the script to train and test the agent:

python new_with_train.py

## Training

The script:

    Initializes the CartPole-v1 environment.
    Trains a DQN agent on the environment for 100,000 timesteps.
    Saves the trained model as cartpole_agent.

## Testing

After training, the script:

    Loads the saved model.
    Tests the model in a new environment instance with rendering enabled.
    Displays the final score (sum of rewards).

Expected Output

    During training, the script will log progress.
    During testing, the CartPole environment will be displayed.
    At the end, the final score is printed:

    🎯 Final Score: XX.XX  

Customization

    Adjust total_timesteps=100000 to train longer.
    Modify time.sleep(0.05) to change visualization speed during testing.

# Here's a video of the model playing the game


https://drive.google.com/file/d/17D-blWcj6pgnvqK0DpJ719hn3BDxAzDy/view?usp=sharing

# MineRL RL Model - Tree Chopping Task

This project trains a reinforcement learning (RL) agent using Stable-Baselines3's PPO algorithm in the MineRLTreechop-v0 environment. The goal is to teach the agent to efficiently chop trees within Minecraft.
Features

    Uses Proximal Policy Optimization (PPO) with a CNN-based policy.
    Trains the model for 100,000 steps to improve performance.
    Saves and loads the trained model for testing and evaluation.
    Evaluates model performance over multiple episodes.

Setup Instructions

    Install dependencies:

pip install minerl gym stable-baselines3

Run the training script:

    python train_minerl_rl.py

    Load and test the trained model.


License

This project is open-source and provided for learning purposes.
