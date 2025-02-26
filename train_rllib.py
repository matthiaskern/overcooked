import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM, RandomRLM
import os

def main(args):
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

    register_env(
        "Overcooked",
        lambda _: Overcooked_multi(**env_params),
    )

    if args.rl_module == 'stationary':
        rlm = RLModuleSpec(module_class=AlwaysStationaryRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'random':
        rlm = RLModuleSpec(module_class=RandomRLM)
        policies_to_train = ['ai']
    elif args.rl_module == 'learned':
        rlm = RLModuleSpec()
        policies_to_train = ['ai', 'human']
    else:
        raise NotImplementedError(f"{args.rl_module} not a valid human agent")

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Overcooked")
        .env_runners(
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner = 1,
            num_gpus_per_env_runner= 0
        )
        .multi_agent(
            policies={"ai", "human"},
            policy_mapping_fn=lambda aid, *a, **kw: aid,
            policies_to_train=policies_to_train

        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "human": rlm,
                    "ai": RLModuleSpec(),
                }
            ),
        )
        .training(
            lr=1e-3,
            lambda_=0.98,
            gamma=0.99,
            clip_param=0.05,
            entropy_coeff=0.1,
            vf_loss_coeff=0.1,
            grad_clip=0.1,
            num_epochs=10,
            minibatch_size=64,
        )
    )

    ray.init()
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    experiment_name = f"{args.name}_{args.rl_module}_{int(time.time()*1000)}"
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop = {"training_iteration": 200},
            checkpoint_config = CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True, num_to_keep=2),
        )
    )

    tuner.fit()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="stationary", help = "Set the policy of the human, can be stationary, random, or learned")

    args = parser.parse_args()
    ip = main(args)