import argparse
import copy
from environment.Overcooked import Overcooked_multi
from Agents import *
import pandas as pd

TASKLIST = [
    "tomato salad", "lettuce salad", "onion salad",
    "lettuce-tomato salad", "onion-tomato salad",
    "lettuce-onion salad", "lettuce-onion-tomato salad"
]

class Player:
    ACTION_MAPPING = {
        "w": 3,
        "d": 0,
        "a": 2,
        "s": 1,
        "q": 4
    }

    REWARD_LIST = {
        "subtask finished": 10,
        "metatask failed": -5,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.1
    }

    def __init__(self, grid_dim, task, map_type, mode, debug, agent='human', llm_model=None, auto_play=False):
        self.env_params = {
            'grid_dim': grid_dim,
            'task': TASKLIST[task],
            'rewardList': self.REWARD_LIST,
            'map_type': map_type,
            'mode': mode,
            'debug': debug
        }
        self.env = Overcooked_multi(**self.env_params)

        if agent == 'stationary':
            self.agent = AlwaysStationaryRLM(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True
            )
        elif agent == 'random':
            self.agent = RandomRLM(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True
            )
        elif agent == 'llm':
            self.agent = LLMAgent(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True,
                llm_model=llm_model,
                environment=self.env
            )
            if debug:
                print(f"LLM agent configured with grid size: {grid_dim} and task: {TASKLIST[task]}")
        elif agent == 'human':
            self.agent = 'human'
        else:
            raise NotImplementedError(f'{agent} is unknown')

        self.rewards = 0
        self.discount = 1
        self.step = 0
        self.auto_play = auto_play
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
                status = f"Contains: {', '.join([f.rawName for f in item.containing])}" if item.containing else "Empty"
            print(f"{item.rawName} at ({item.x},{item.y}) - {status}")
        print("===========================\n")

    def run(self):
        self.env.game.on_init()
        new_obs, _ = self.env.reset()
        self.env.render()
        data = [["obs", "action_human", "action_ai", "new_obs", "reward_human", "reward_ai", "done"]]

        while True:
            obs = new_obs
            row = [obs['human']]
            self.step += 1

            self.print_state_debug()

            prev_pos = (self.env.agent[1].x, self.env.agent[1].y)

            input_human = ["q"] if self.auto_play else input("Input Human: ").strip().split(" ")

            if input_human == ['p']:
                self.save_data(data)
                continue

            input_ai = self.agent._forward_inference({"obs": [obs['ai']]})['actions']
            input_ai = [list(self.ACTION_MAPPING.keys())[list(self.ACTION_MAPPING.values()).index(input_ai[0])]]

            action = {
                "human": self.ACTION_MAPPING[input_human[0]],
                "ai": self.ACTION_MAPPING[input_ai[0]]
            }

            row.append(action['human'])
            row.append(action['ai'])

            new_obs, reward, done, _, _ = self.env.step(action)

            new_pos = (self.env.agent[1].x, self.env.agent[1].y)
            moved = new_pos != prev_pos
            if self.debug:
                print(f"[STEP RESULT] primitive actions: [{action['human']}, {action['ai']}] | Rewards: H={reward['human']} AI={reward['ai']} | Done: {done['__all__']}")
                if not moved:
                    print("[WARNING] AI DID NOT MOVE!")

            self.agent.last_result = f"Executed: {input_ai[0]}, {'Success' if moved else 'Failed to move'}"

            row.append(new_obs['human'])
            row.append(reward['human'])
            row.append(reward['ai'])
            row.append(done['__all__'])

            data.append(copy.deepcopy(row))

            self.env.render()

            if done['__all__']:
                self.save_data(data)
                break

    def save_data(self, data):
        columns = data[0]
        data = data[1:]
        df = pd.DataFrame(data, columns=columns)
        csv_filename = "output.csv"
        df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help='Grid world size')
    parser.add_argument('--task', type=int, default=6, help='The recipe agent cooks')
    parser.add_argument('--map_type', type=str, default="A", help='The type of map')
    parser.add_argument('--mode', type=str, default="vector", help='The type of observation (vector/image)')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information and render')
    parser.add_argument('--agent', type=str, default='human', help='Type of agent, e.g. (human|llm)')
    parser.add_argument('--llm_model', type=str, default=None, help='LLM model to use (e.g. "openai/gpt-4o")')
    parser.add_argument('--auto-play', type=bool, default=False, help='Keep the human stationary')

    params = vars(parser.parse_args())

    player = Player(**params)
    player.run()
