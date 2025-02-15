import argparse
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import torch
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from gym_macro_overcooked.Overcooked import Overcooked_multi

random.seed(42)
torch.manual_seed(42)

class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleAgentWrapper, self).__init__(env)
        self.observation_space = env.observation_spaces['ai']
        self.action_space = env.action_spaces['ai']

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        return obs['ai'], {}

    def step(self, action):
        actions = {"human": action, "ai": 4}  # Adjust if needed
        obs, rewards, dones, truncated, info = self.env.step(actions)
        return obs['ai'], rewards['ai'], dones['__all__'], truncated['__all__'], info['ai']


class CumulativeRewardCallback(BaseCallback):
    """Tracks cumulative rewards and saves the model periodically."""

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.cumulative_rewards = []
        self.cumulative_reward = 0.0
        self.step_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1

        # Log step progress
        if self.step_counter % 1000 == 0:
            print(f"Step: {self.step_counter}")

        # Save model periodically
        if self.step_counter % self.save_freq == 0:
            model_path = f"{self.save_path}/step_{self.step_counter}.zip"
            self.model.save(model_path)
            print(f"Model saved at {model_path}")

        # Accumulate rewards
        if 'rewards' in self.locals:
            self.cumulative_reward += self.locals['rewards'][0]
            self.cumulative_rewards.append(self.cumulative_reward)

        return True

    def get_cumulative_rewards(self):
        return self.cumulative_rewards

def create_env():
    """Creates and wraps the environment."""
    reward_config = {
        "metatask failed": 0,
        "goodtask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1.,
    }

    env_params = {
        "grid_dim": [5, 5],
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "obs_radius": 0, # full observability.
        "mode": "vector",
        "debug": False,
    }
    env = Overcooked_multi(**env_params)
    return SingleAgentWrapper(env)


def train_model(env, exp_name):
    """Trains the PPO model on the environment."""
    exp_dir = f"./experiments/{exp_name}"
    log_dir = f"./runs/sb3_run/"
    save_path = f"{exp_dir}/model"

    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    new_logger = configure(log_dir, ["csv", "tensorboard"])

    ppo_params = {
        "learning_rate": 1e-3,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "clip_range": 0.05,
        "ent_coef": 0.1,
        "vf_coef": 0.1,
        "max_grad_norm": 0.1,
        "verbose": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    model = PPO("MlpPolicy", env, **ppo_params, tensorboard_log = log_dir)
    model.set_logger(new_logger)

    reward_callback = CumulativeRewardCallback(save_freq=50_000, save_path=save_path, verbose=1)  # TODO: save_freq=50000

    model.learn(total_timesteps=5_000_000, callback=reward_callback)  # TODO: total_timesteps=5_000_000
    model.save(f"{save_path}/final_model.zip")

    return model, reward_callback


def plot_rewards(reward_callback):
    """Plots cumulative rewards after training."""
    cumulative_rewards = reward_callback.get_cumulative_rewards()

    if cumulative_rewards:
        plt.plot(cumulative_rewards)
        plt.xlabel("Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Curve")
        plt.show()
    else:
        print("No cumulative rewards logged. Check environment and reward tracking.")


def test_model(env, model_path):
    """Loads and tests the trained model."""
    try:
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return

    obs = env.reset()
    env.game.on_init()
    for step in range(100):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if done:
            break

        env.render()
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PPO model on Overcooked environment.")
    parser.add_argument("--exp_name", type=str, default="reproduce",
                        help="Experiment name (used for saving logs and models)")
    parser.add_argument("--train", action="store_true", help="Enable training mode", default=True)
    parser.add_argument("--test", action="store_true", help="Enable testing mode", default=True)

    args = parser.parse_args()

    env = create_env()
    if args.train:
        print(f"Starting training: {args.exp_name}")
        model, reward_callback = train_model(env, args.exp_name)
        plot_rewards(reward_callback)

    if args.test:
        model_path = f"./experiments/{args.exp_name}/model/final_model.zip"

        if os.path.exists(model_path):
            print(f"Starting testing using model: {model_path}")
            test_model(env, model_path)
        else:
            print(f"Model file not found: {model_path}. Ensure training has been completed first.")