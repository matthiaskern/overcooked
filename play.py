import argparse
import copy
from environment.Overcooked import Overcooked_multi
from Agents import *
import pandas as pd
import imageio

TASKLIST = [
    "tomato salad",
    "lettuce salad",
    "onion salad",
    "lettuce-tomato salad",
    "onion-tomato salad",
    "lettuce-onion salad",
    "lettuce-onion-tomato salad",
]


class Player:
    ACTION_MAPPING = {"w": 3, "d": 0, "a": 2, "s": 1, "q": 4}

    REWARD_LIST = {
        "subtask finished": 10,
        "metatask failed": -5,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.1,
    }

    def __init__(
        self,
        grid_dim,
        task,
        map_type,
        mode,
        debug,
        agent="human",
        llm_model=None,
        human="interactive",
        horizon_length=3,
    ):
        self.env_params = {
            "grid_dim": grid_dim,
            "task": TASKLIST[task],
            "rewardList": self.REWARD_LIST,
            "map_type": map_type,
            "mode": mode,
            "debug": debug,
        }
        self.env = Overcooked_multi(**self.env_params)

        if agent == "stationary":
            self.agent = AlwaysStationaryRLM(
                observation_space=self.env.observation_spaces["ai"],
                action_space=self.env.action_spaces["ai"],
                inference_only=True,
            )
        elif agent == "random":
            self.agent = RandomRLM(
                observation_space=self.env.observation_spaces["ai"],
                action_space=self.env.action_spaces["ai"],
                inference_only=True,
            )
        elif agent == "llm":
            self.agent = LLMAgent(
                observation_space=self.env.observation_spaces["ai"],
                action_space=self.env.action_spaces["ai"],
                inference_only=True,
                llm_model=llm_model,
                environment=self.env,
                horizon_length=horizon_length,
            )
        elif agent == "multimodal":
            self.agent = MultiModalAgent(
                observation_space=self.env.observation_spaces["ai"],
                action_space=self.env.action_spaces["ai"],
                inference_only=True,
                llm_model=llm_model,
                environment=self.env,
                horizon_length=horizon_length,
                agent_name="ai",
                agent_idx=1
            )
        elif agent == "human":
            self.agent = "human"
        else:
            raise NotImplementedError(f"{agent} is unknown")

        self.human = human
        if human == "random":
            self.human_agent = RandomRLM(
                observation_space=self.env.observation_spaces["human"],
                action_space=self.env.action_spaces["human"],
                inference_only=True,
            )
            print("Human player replaced with Random Agent")
        elif human == "llm":
            self.human_agent = LLMAgent(
                observation_space=self.env.observation_spaces["human"],
                action_space=self.env.action_spaces["human"],
                inference_only=True,
                llm_model=llm_model,
                environment=self.env,
                horizon_length=horizon_length,
            )
            print("Human player replaced with LLM Agent")
        elif human == "multimodal":
            self.human_agent = MultiModalAgent(
                observation_space=self.env.observation_spaces["human"],
                action_space=self.env.action_spaces["human"],
                inference_only=True,
                llm_model=llm_model,
                environment=self.env,
                horizon_length=horizon_length,
                agent_name="human",
                agent_idx=0
            )
            print("Human player replaced with Multimodal Agent")
        elif human == "stationary":
            self.human_agent = AlwaysStationaryRLM(
                observation_space=self.env.observation_spaces["human"],
                action_space=self.env.action_spaces["human"],
                inference_only=True,
            )
            print("Human player replaced with Stationary Agent")
        else:
            self.human_agent = None

        self.rewards = 0
        self.discount = 1
        self.step = 0
        self.debug = debug

    def print_state_debug(self):
        if not self.debug:
            return
        print("\n===== ENV STATE DEBUG =====")
        for i, a in enumerate(self.env.agent):
            holding = "Nothing"
            if a.holding:
                holding = getattr(a.holding, "rawName", str(a.holding))
            print(f"Agent #{i} at ({a.x},{a.y}) holding: {holding}")
        print("\nItems on map:")
        for item in self.env.itemList:
            status = "N/A"
            if hasattr(item, "chopped"):
                status = f"Chopped: {item.chopped} ({item.cur_chopped_times}/{item.required_chopped_times})"
            elif hasattr(item, "containing"):
                status = (
                    f"Contains: {', '.join([f.rawName for f in item.containing])}"
                    if item.containing
                    else "Empty"
                )
            print(f"{item.rawName} at ({item.x},{item.y}) - {status}")
        print("===========================\n")

    def run(self):
        self.env.game.on_init()
        new_obs, _ = self.env.reset()
        self.env.render()
        data = [
            [
                "obs",
                "action_human",
                "action_ai",
                "new_obs",
                "reward_human",
                "reward_ai",
                "done",
            ]
        ]

        while True:
            obs = new_obs
            row = [obs["human"]]
            self.step += 1

            self.print_state_debug()

            prev_pos = (self.env.agent[1].x, self.env.agent[1].y)
            prev_human_pos = (self.env.agent[0].x, self.env.agent[0].y)

            if self.human_agent:
                input_human_action = self.human_agent._forward_inference(
                    {"obs": [obs["human"]]}
                )["actions"]
                input_human = [
                    list(self.ACTION_MAPPING.keys())[
                        list(self.ACTION_MAPPING.values()).index(input_human_action[0])
                    ]
                ]
                if self.debug:
                    print(f"Human Agent Action: {input_human[0]}")
            else:
                # Use real human input
                input_human = input("Input Human: ").strip().split(" ")

            if input_human == ["p"]:
                self.save_data(data)
                continue

            # Get AI agent action
            input_ai = self.agent._forward_inference({"obs": [obs["ai"]]})["actions"]
            print("AI Action was: " + str(input_ai))
            input_ai = [
                list(self.ACTION_MAPPING.keys())[
                    list(self.ACTION_MAPPING.values()).index(input_ai[0])
                ]
            ]

            action = {
                "human": self.ACTION_MAPPING[input_human[0]],
                "ai": self.ACTION_MAPPING[input_ai[0]],
            }

            row.append(action["human"])
            row.append(action["ai"])

            new_obs, reward, done, _, _ = self.env.step(action)

            # Check if agents moved
            new_pos = (self.env.agent[1].x, self.env.agent[1].y)
            new_human_pos = (self.env.agent[0].x, self.env.agent[0].y)
            ai_moved = new_pos != prev_pos
            human_moved = new_human_pos != prev_human_pos
            print("AI Pos is: " + str(new_pos))
            if self.debug:
                print(
                    f"[STEP RESULT] actions: [H:{input_human[0]}, AI:{input_ai[0]}] | Rewards: H={reward['human']} AI={reward['ai']} | Done: {done['__all__']}"
                )
                if not ai_moved:
                    print("[WARNING] AI AGENT DID NOT MOVE!")
                if self.human_agent and not human_moved:
                    print("[WARNING] HUMAN AGENT DID NOT MOVE!")

            if hasattr(self.agent, "last_result"):
                self.agent.last_result = f"Executed: {input_ai[0]}, {'Success' if ai_moved else 'Failed to move'}"

            row.append(new_obs["human"])
            row.append(reward["human"])
            row.append(reward["ai"])
            row.append(done["__all__"])

            data.append(copy.deepcopy(row))

            self.env.render()

            if done["__all__"]:
                self.save_data(data)
                break
        return data

    def save_data(self, data):
        columns = data[0]
        data = data[1:]
        df = pd.DataFrame(data, columns=columns)
        csv_filename = "output.csv"
        df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_dim", type=int, nargs=2, default=[5, 5], help="Grid world size"
    )
    parser.add_argument("--task", type=int, default=6, help="The recipe agent cooks")
    parser.add_argument("--map_type", type=str, default="A", help="The type of map")
    parser.add_argument(
        "--mode",
        type=str,
        default="vector",
        help="The type of observation (vector/image)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="Whether to print debug information and render",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="human",
        help="Type of agent, e.g. (human|llm|multimodal|random|stationary)",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help='LLM model to use (e.g. "openai/gpt-4.1")',
    )
    parser.add_argument(
        "--human",
        type=str,
        default="interactive",
        help="Type of human player (interactive|stationary|random|llm|multimodal)",
    )
    parser.add_argument(
        "--horizon-length", type=int, default=3, help="Set the planning horizon length"
    )

    params = vars(parser.parse_args())

    player = Player(**params)
    player.run()
