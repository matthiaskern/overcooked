import argparse
import os
import matplotlib.pyplot as plt
import time
import torch
import random
import numpy as np

from gym_macro_overcooked.overcooked_V1 import Overcooked_V1

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def create_env():
    """Creates and wraps the environment."""
    reward_config = {
        "metatask failed": 0,
        "goodtask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -0.1,
    }
    env_params = {
        "grid_dim": [7, 7],
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "n_agent": 2,
        "obs_radius": 0,
        "mode": "vector",
        "debug": False,
    }
    return Overcooked_V1(**env_params)


def train_model(env, exp_name):
    """Trains the PPO model using RLlib with a heuristic policy."""
    exp_dir = f"./experiments/{exp_name}"
    log_dir = f"{exp_dir}/logs"
    save_path = f"{exp_dir}/model"

    # Ensure directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Register environment
    register_env("OvercookedEnv", lambda _: create_env())

    config = (
        PPOConfig()
        .environment("OvercookedEnv")
        .env_runners(num_env_runners=0)  # No parallelism
        .multi_agent(
            policies={"human", "ai"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "human" if agent_id == 0 else "ai",
            policies_to_train=["ai"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "human": RLModuleSpec(module_class=RandomRLModule),
                    "ai": RLModuleSpec(),
                }
            ),
        )
    )

    algo = config.build_algo()

    print("Starting training...")
    for i in range(1):
        result = algo.train()
        if (i + 1) % 1 == 0:
            save_path = os.path.abspath(f"./experiments/{exp_name}/model")
            algo.save(save_path)
            print(f"Saved model to {save_path}")

    algo.stop()
    return algo


def plot_rewards(log_dir):
    """Plots cumulative rewards after training."""
    log_path = os.path.join(log_dir, "progress.csv")
    if not os.path.exists(log_path):
        print(f"No logs found in {log_path}. Skipping reward plot.")
        return

    import pandas as pd
    df = pd.read_csv(log_path)

    if "episode_reward_mean" in df.columns:
        plt.plot(df["episode_reward_mean"])
        plt.xlabel("Training Iterations")
        plt.ylabel("Mean Episode Reward")
        plt.title("Training Reward Curve")
        plt.show()
    else:
        print("Reward data not found in logs.")



def test_model(env, model_path):
    """Loads and tests the trained model."""
    algo = Algorithm.from_checkpoint(model_path)

    obs = env.reset()
    env.game.on_init()

    for step in range(100):
        actions = {i: algo.compute_actions(obs, policy_id=p) for i, p in enumerate(['human', 'ai'])}
        obs, reward, done, info = env.step(actions)
        if done:
            break

        env.render()
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PPO model on Overcooked environment.")
    parser.add_argument("--exp_name", type=str, default="rllib_experiment", help="Experiment name")
    parser.add_argument("--train", action="store_true", help="Enable training mode", default=True)
    parser.add_argument("--test", action="store_true", help="Enable testing mode", default=True)

    args = parser.parse_args()

    env = create_env()

    if args.train:
        print(f"Starting training: {args.exp_name}")
        algo = train_model(env, args.exp_name)
        plot_rewards(f"./experiments/{args.exp_name}/logs")

    if args.test:
        model_path = os.path.abspath(f"./experiments/{args.exp_name}/model")

        if os.path.exists(model_path):
            print(f"Starting testing using model: {model_path}")
            test_model(env, model_path)
        else:
            print(f"Model file not found: {model_path}. Ensure training has been completed first.")