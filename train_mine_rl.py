import minerl
import gym
import torch
from stable_baselines3 import PPO

# Initialize the MineRL environment
env = gym.make("MineRLTreechop-v0")

# Define PPO model
model = PPO(
    "CnnPolicy",  # CNN policy to handle image observations
    env,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_steps=512,
    batch_size=32,
    gae_lambda=0.95,
    gamma=0.99,
    learning_rate=3e-4
)

# Train the model for 100,000 steps
print("Training the RL model...")

model.learn(total_timesteps=100000)

print("Training completed!")

# Save the trained model
model.save("ppo_minerl_treechop")
print("Model saved as ppo_minerl_treechop.zip")

# Load the trained model
model = PPO.load("ppo_minerl_treechop")

# Test the trained model
def test_model(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()  # Render the environment
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

test_model(env, model)

# Evaluate model performance
def evaluate_model(env, model, episodes=10):
    episode_rewards = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)
    avg_reward = sum(episode_rewards) / episodes
    print("Average reward over 10 episodes:", avg_reward)

evaluate_model(env, model)
