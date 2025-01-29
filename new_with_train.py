import gym
from stable_baselines3 import DQN
import time  # Import time module to control the delay

# Load the CartPole environment
env = gym.make("CartPole-v1")

# Create the DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model for 100,000 steps
model.learn(total_timesteps=100000)

model.save("cartpole_agent")
print("✅ Model trained and saved successfully!")

# Load the trained model
model = DQN.load("cartpole_agent")

# Test the trained model and measure the score with rendering enabled only during testing
test_env = gym.make("CartPole-v1", render_mode="human")  # Only enable rendering during testing

obs, _ = test_env.reset()
done = False
score = 0  # Track total rewards (score)

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)
    score += reward  # Update score
    test_env.render()  # Show the game during testing
    
    # Add a small delay to slow down the rendering
    time.sleep(0.05)  # Adjust the value (in seconds) to control the speed of the game

test_env.close()

print(f"🎯 Final Score: {score:.2f}")  # Display the score after the game ends
