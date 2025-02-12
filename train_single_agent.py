import gym
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import time
# Import your custom environment
from gym_macro_overcooked.overcooked_V1 import Overcooked_V1
import torch
import random
random.seed(42)



class SingleAgentWrapper(gym.Wrapper):
    """
    A wrapper to treat a multi-agent environment as a single-agent environment for training.
    It assumes the environment returns lists for observations and rewards, and it will take the first agent's view.
    """
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        return obs[0] if isinstance(obs, list) else obs

    def step(self, action):
        actions = [action, 4]  # Adjust according to your needs
        obs, rewards, dones, info = self.env.step(actions)
        return obs[0], rewards[0], dones, info

# Define your environment parameters
rewardList = {
    # "minitask finished": 0,
    # "minitask failed": 0,
    # "metatask finished": 0,
    "metatask failed": 0,
    "goodtask finished": 5,
    # "goodtask failed": 0,
    "subtask finished": 10,
    # "subtask failed": 0,
    "correct delivery": 200,
    "wrong delivery": -50,
    # "wrong delivery": -5, # 鼓励多去deliver，多试错
    "step penalty": -0.1
    # "step penalty": -1
}


# env_id = 'Overcooked-v1'
env_id = 'Overcooked-shuai-v0'


env_params = {
    'grid_dim': [5, 5],
    'task': "tomato salad",
    'rewardList': rewardList,
    'map_type': "A",
    'n_agent': 2,
    'obs_radius': 0,
    'mode': "vector",
    'debug': True
}

# Create and wrap the environment
env = gym.make(env_id, **env_params)
env = SingleAgentWrapper(env)




class CumulativeRewardCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CumulativeRewardCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.cumulative_rewards = []
        self.cumulative_reward = 0.0
        self.step_counter = 0  # Initialize a step counter

    def _on_step(self) -> bool:
        # Increment the step counter
        self.step_counter += 1

        # Print the current step every 1,000 steps
        if self.step_counter % 1000 == 0:
            print(f"Current step: {self.step_counter}")

        # Save the model every save_freq steps
        if self.step_counter % self.save_freq == 0:
            model_path = f"{self.save_path}_step_{self.step_counter}"
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

        # Accumulate rewards
        self.cumulative_reward += self.locals['rewards'][0]  # Access the reward correctly
        self.cumulative_rewards.append(self.cumulative_reward)
        return True

    def get_cumulative_rewards(self):
        return self.cumulative_rewards

# Create the callback with the desired save frequency and path
reward_callback = CumulativeRewardCallback(save_freq=50000, save_path='trust_B_55_lowaction')



new_logger = configure('./logs/', ["csv", "tensorboard"])  # Remove "stdout" to prevent console logging


# Define the model
ppo_params = {
    'learning_rate': 1e-3,    # Starting learning rate
    'n_steps': 128,           # 多少步更新一次策略
    'batch_size': 64,         # 每次更新策略的时候使用多少步的数据
    'n_epochs': 10,            # 表示在每次策略更新时，算法将对收集到的经验数据进行多少次遍历和优化。也就是说，每当从环境中收集了一批数据（通常是 n_steps 个时间步的数据）后，算法会使用这些数据进行 n_epochs 次优化。
    'gamma': 0.99,            # Discount factor
    'gae_lambda': 0.98,       # Lambda for GAE
    'clip_range': 0.05,       # PPO clipping factor
    'ent_coef': 0.1,          # Entropy coefficient
    'vf_coef': 0.1,           # Value function coefficient
    'max_grad_norm': 0.1,     # Max gradient norm
    'verbose': 1,             # Verbosity level
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    'verbose': 0,             # Set verbosity level to 0 to minimize output during training
}


# Initialize the PPO model with these parameters
model = PPO("MlpPolicy", env, **ppo_params)



# model.set_logger(new_logger)

# # Create the callback
# # reward_callback = CumulativeRewardCallback()

# # Train the model with the custom callback
# model.learn(total_timesteps=5000000, callback=reward_callback)

# model.save("A_55_lowaction_five_million_step")

# # Retrieve cumulative rewards
# cumulative_rewards = reward_callback.get_cumulative_rewards()

# # Ensure there's data to plot
# if cumulative_rewards:
#     # Plot the cumulative reward curve
#     plt.plot(cumulative_rewards)
#     plt.xlabel('Step')
#     plt.ylabel('Cumulative Reward')
#     plt.title('Cumulative Reward Curve')
#     plt.show()
# else:
#     print("No cumulative rewards were logged. Please check the environment and reward logging.")


model.load("A_55_lowaction_five_million_step")
# Test the trained model
obs = env.reset()
for step in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()
    time.sleep(0.1)


# Close the environment
env.close()