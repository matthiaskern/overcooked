import ray
from gym_macro_overcooked.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM


if __name__ == "__main__":
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
            policies_to_train=['ai']

        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "human": RLModuleSpec(module_class=AlwaysStationaryRLM),
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

    tuner = tune.Tuner(
        "PPO",
        param_space=config,
    )

    tuner.fit()