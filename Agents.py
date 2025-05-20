from collections import defaultdict, deque
import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from llm_agent import BaseLLMWrapper
from environment.items import Food, Plate, Knife, Delivery
from environment.Overcooked import ITEMNAME
import litellm
import re

class AlwaysStationaryRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [4] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


class RandomRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [self.action_space.sample()] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]

ACTION_MAPPING = {
    "d": 0,
    "s": 1,
    "a": 2,
    "w": 3,
    "q": 4
}

class BaseLLMWrapper:
    def __init__(self, model="openai/gpt-4o", memory_limit=10, horizon_length=3):
        self.model = model
        self.planner_memory = deque(maxlen=memory_limit)
        self.executor_memory = deque(maxlen=memory_limit)
        self.plan = None
        self.horizon_length = horizon_length

    def call(self, messages):
        try:
            print("\n=== LLM Prompt ===")
            print(messages[-1]["content"])
            print("==================\n")

            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )

            content = response.choices[0].message.content

            print("=== LLM Response ===")
            print(content)
            print("====================\n")

            return content
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return "Action: w\nVerify: no"

    def extract_action_and_verify(self, text):
        action_match = re.search(r"Action:\s*\[?([wsad])\]?", text, re.IGNORECASE)
        verify_match = re.search(r"Verify:\s*(yes|no)", text, re.IGNORECASE)

        action = action_match.group(1).lower() if action_match else None
        verified = verify_match.group(1).lower() if verify_match else "no"

        return action if verified == "yes" and action else "w"

    def update_plan(self, task):
        prompt = (
            "You are a planner for Overcooked.\n\n"
            f"Task: {task}\n\n"
            "Generate a high-level plan in steps.\n\n"
            f"- Plan for approximately {self.horizon_length} key steps ahead.\n"
            "NOTE:\n"
            "- To pick up, chop, serve, or interact with any item — MOVE into the item's direction.\n"
            "- The [q] action does nothing in this game.\n"
            "- Only dishes placed on PLATES can be served at the DELIVERY counter (marked as '*').\n"
            "- To pick up items at (row,0), you must stand at (row,1) and interact LEFT.\n"
            "- Always stay within the grid and avoid stepping into walls.\n"
            "- You cannot move into another agent's tile.\n"
            "- Chopping requires multiple interactions: You must stay next to the knife and move into it repeatedly until the food is chopped."
            "- DO NOT leave the knife or touch anything else until the food is chopped (usually 3 interactions)."
            "- You must PICK UP the plated tomato before you serve it!"
            "- Once the food is plated, PICK IT UP BY INTERACTING WITH IT, then move to the delivery counter with the plated food"
            "- Please dont forget to PICK UP necessary items!"
        )
        self.plan = self.call([{"role": "user", "content": prompt}])
        self.planner_memory.append(self.plan)

    def get_action(self, env, agent_idx, last_action=None, last_result=None):
        agent = env.agent[agent_idx]
        task = ", ".join(env.task) if isinstance(env.task, list) else env.task

        if not self.plan:
            self.update_plan(task)

        holding = self._describe_holding(agent)
        state_summary = self._describe_env(env)
        debug_full_state = self._describe_env_debug(env)
        rendered_grid = self._render_grid(env)
        possible_moves = self._describe_possible_moves(env, agent)
        memory_text = "\n".join(self.executor_memory)

        feedback = ""
        max_attempts = 5
        for attempt in range(max_attempts):
            moved = "Unknown"
            if last_result:
                moved = "Failed to move" if "Failed" in last_result else "Success"

            prev = getattr(self, "prev_holding", "Nothing")
            curr = holding

            if prev == "Nothing" and "tomato" in curr:
                self.executor_memory.append("Agent picked up tomato")
            elif "tomato" in prev and curr == "Nothing":
                self.executor_memory.append("Agent dropped tomato")
            elif curr == "Nothing" and self.executor_memory and "chop" in self.executor_memory[-1].lower():
                self.executor_memory.append("Agent tried to chop without holding anything")

            self.prev_holding = curr

            prompt = f"""You are the AI agent in Overcooked. Task: {task}

    Your position: ({agent.x}, {agent.y})
    Holding: {holding}

    Last action: {last_action or "None"}
    Result: {last_result or "None"} — Movement status: {moved}

    Plan:
    {self.plan}

    Environment (Summary):
    {state_summary}

    Grid:
    {rendered_grid}

    Memory:
    {memory_text}

    INSTRUCTIONS:

    - Your immediate goal is one of: [get ingredient, chop ingredient, pick plate, plate food, deliver dish].
    - Construct a ROUTE of exactly {self.horizon_length} steps: list of (row, col) steps needed to reach an tile next to the target (not directly on the target if it's an item).

    IMPORTANT:
    - [w]: Move up or perform an action on the tile above
    - [a]: Move left or perform an action on the tile to the left
    - [s]: Move down or perform an action on the tile below
    - [d]: Move right or perform an action on the tile to the right
    - [q]: Do nothing
    - Remember, coords are not given in x,y but in row/col. (i.e. to navigate vertically we need to change x and when we navigate horizontally we change y)
    - If the previous action failed to move, re-evaluate and fix your decision.

    {feedback}

    Respond strictly in the following format:
    Thought: ...
    Route: ...
    Plan: ...
    Action: [w/a/s/d/q] - [reason]
    Verify: [yes/no] - [reason]
    """

            response = self.call([{"role": "user", "content": prompt}])
            self.executor_memory.append(response)

            action_letter = self.extract_action_and_verify(response)

            route_match = re.search(r"Route:\s*\[([^\]]*)\]", response, re.IGNORECASE)
            route = route_match.group(1).strip() if route_match else "[]"

            #verifier_response = self.verify_plan_and_action(env, agent, route, action_letter, response, agent_positions, possible_moves)

            #print("=== LLM Verifier ===")
            #print(verifier_response)
            #print("====================\n")

            #sanity_ok_match = re.search(r"SanityOK:\s*(yes|no)", verifier_response, re.IGNORECASE)
            #if sanity_ok_match and sanity_ok_match.group(1).strip().lower() == "yes":
            #    return action_letter
            #else:
            #    feedback = f"\nFEEDBACK FROM VERIFIER:\n{verifier_response}\n\nUpdate your plan and action to fix these issues."
            return action_letter
        print("[ERROR] Agent failed to generate valid move after retries. Defaulting to [q].")
        return "q"


    def _describe_holding(self, agent):
        if not agent.holding:
            return "Nothing"
        if isinstance(agent.holding, Food):
            return f"{agent.holding.rawName} (Chopped: {agent.holding.chopped})"
        if isinstance(agent.holding, Plate):
            if agent.holding.containing:
                return "Plate with " + ", ".join(f.rawName for f in agent.holding.containing)
            return "Empty Plate"
        return "Unknown item"

    def _describe_env(self, env):
        description = []

        # --- Agent states ---
        description.append("Agents:")
        for i, agent in enumerate(env.agent):
            role = "Human" if i == 0 else "AI"
            holding = self._describe_holding(agent)
            description.append(f"{role} at ({agent.x}, {agent.y}) holding: {holding}")

        # --- Full grid view ---
        description.append("\nMap Grid (x rows, y cols):")
        xlen, ylen = env.xlen, env.ylen
        grid = [["" for _ in range(ylen)] for _ in range(xlen)]

        # Populate base layer from env.map
        for x in range(xlen):
            for y in range(ylen):
                tile_idx = env.map[x][y]
                tile_type = ITEMNAME[tile_idx]
                grid[x][y] = tile_type.upper() if tile_type == "counter" else tile_type.capitalize()

        # Override with item info
        for item in env.itemList:
            label = item.rawName.capitalize()
            if isinstance(item, Food) and item.chopped:
                label += " (chopped)"
            elif isinstance(item, Plate) and item.containing:
                contents = ", ".join(f.rawName for f in item.containing)
                label = f"Plate ({contents})"
            grid[item.x][item.y] = label

        # Override with agents
        for i, agent in enumerate(env.agent):
            role = "Human" if i == 0 else f"AI{i}"
            grid[agent.x][agent.y] = role

        for x in range(xlen):
            row = []
            for y in range(ylen):
                row.append(f"[{x},{y}] {grid[x][y]:<18}")
            description.append(" ".join(row))

        return "\n".join(description)




    def _render_grid(self, env):
        width = env.width if hasattr(env, "width") else 5
        height = env.height if hasattr(env, "height") else 5
        grid = [["Empty" for _ in range(width)] for _ in range(height)]

        for item in env.itemList:
            x, y = item.x, item.y
            name = item.rawName.lower()
            if name == "knife":
                grid[y][x] = "Knife"
            elif name == "tomato":
                desc = "Tomato"
                if hasattr(item, "chopped") and item.chopped:
                    desc += " (chopped)"
                grid[y][x] = desc
            elif name == "lettuce":
                grid[y][x] = "Lettuce"
            elif name == "onion":
                grid[y][x] = "Onion"
            elif name == "plate":
                if item.containing:
                    contents = ", ".join(f.rawName for f in item.containing)
                    grid[y][x] = f"Plate ({contents})"
                else:
                    grid[y][x] = "Plate"
            elif name == "delivery":
                grid[y][x] = "Delivery"
            elif name == "counter":
                grid[y][x] = "Counter"

        for idx, agent in enumerate(env.agent):
            agent_descr = "Human" if idx == 0 else "Agent"
            grid[agent.y][agent.x] = f"{agent_descr}{idx}"

        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    grid[y][x] = "Counter"

        output = ["Grid:"]
        for y in range(height):
            row = []
            for x in range(width):
                row.append(f"[{y},{x}] {grid[y][x]:<20}")
            output.append(" ".join(row))
        return "\n".join(output)

    def _describe_possible_moves(self, env, agent):
        directions = {'w': (-1, 0), 's': (1, 0), 'a': (0, -1), 'd': (0, 1)}
        results = []
        width = env.width if hasattr(env, "width") else 5
        height = env.height if hasattr(env, "height") else 5

        for dir_letter, (dy, dx) in directions.items():
            new_row = agent.y + dy  # row (y)
            new_col = agent.x + dx  # col (x)

            if not (0 <= new_col < width and 0 <= new_row < height):
                what = "Counter (Out of bounds)"
            elif new_col == 0 or new_row == 0 or new_row == height - 1:
                what = "Counter (Edge)"
            else:
                what = "empty space"
                for i, other_agent in enumerate(env.agent):
                    if other_agent.x == new_col and other_agent.y == new_row:
                        what = f"Agent #{i} (Human if 0, AI if 1)"
                        break
                else:
                    for item in env.itemList:
                        if item.x == new_col and item.y == new_row:
                            if isinstance(item, Food):
                                what = f"{item.rawName} (chopped: {item.chopped})"
                            elif isinstance(item, Plate):
                                what = "Plate with " + ", ".join(f.rawName for f in item.containing) if item.containing else "Empty Plate"
                            elif isinstance(item, Knife):
                                what = "Cutting Station"
                            elif isinstance(item, Delivery):
                                what = "Delivery Counter"
                            else:
                                what = item.rawName
                            break

            results.append(f"[{dir_letter}] -> ({new_row}, {new_col}): {what}")
        return "\n".join(results)


    def _describe_env_debug(self, env):
            parts = []
            parts.append("Agents:")
            for i, agent in enumerate(env.agent):
                holding = "Nothing"
                if agent.holding:
                    if hasattr(agent.holding, "rawName"):
                        holding = agent.holding.rawName
                    else:
                        holding = str(agent.holding)
                parts.append(f"Agent #{i} at ({agent.x},{agent.y}) holding: {holding}")

            parts.append("\nItems on map:")
            for item in env.itemList:
                if hasattr(item, "chopped"):
                    status = f"Chopped: {item.chopped} ({item.cur_chopped_times}/{item.required_chopped_times})"
                elif hasattr(item, "containing"):
                    status = f"Contains: {', '.join(f.rawName for f in item.containing)}" if item.containing else "Empty"
                else:
                    status = "N/A"
                parts.append(f"{item.rawName} at ({item.x},{item.y}) - {status}")

            return "\n".join(parts)
    
    def verify_plan_and_action(self, env, agent, route, action_letter, full_agent_thought, agnet_positions, nearby_tiles):
        summary = self._describe_env(env)
        grid = self._render_grid(env)
        current_pos = f"({agent.x}, {agent.y})"
        holding = self._describe_holding(agent)

        prompt = f"""
    You are an Overcooked logic validator.

    You will evaluate an AI agent's action plan and sanity based on the current game state.


    Agent Info:
    - Position: {current_pos}
    - Holding: {holding}
    - Thought/Plan: {full_agent_thought}
    - Agent Positions: {agnet_positions}


    Planned Route: {route}
    Proposed Action: [{action_letter}]

    Task: {env.task}

    Evaluate the following:
    1. Route is valid: the positions are reachable, safe (no walls, no agents), and make sense given the task.
    2. Action is valid: it's aligned with the next step in the route, does not walk into a wall or agent.
    3. Sanity check: does the action logically match what the agent *should* do now (e.g., not chopping air, not delivering nothing, not trying to pick up twice, etc).
    4. Remember that sometimes the agent must interact with items sometimes! Also remember that the tomato and chopped tomato might be under knife K!
    5. Remember that the agent might have already placed the tomato on the board or plate! Therefore it might try interacting without holding. This is if the tomato is not visible on map. 
    6. Interaction is not explicitly stated in the action - it is done by moves. No explicit interaction action is therefore necessary. However "stay" is not a valid action. Also, knife is same as cutting board.
    Remember that the agent should then cut the ingredients fully!
    Respond in format:
    ValidRoute: [yes/no] - [why]
    ValidAction: [yes/no] - [why]
    SanityOK: [yes/no] - [why]
    Suggestion: [What would be a better plan or correction if something is wrong?]
    """
        return self.call([{"role": "user", "content": prompt}])



class LLMAgent(RLModule):
    def __init__(self, observation_space, action_space, inference_only=True, llm_model=None, environment=None, horizon_length=3):
        super().__init__()
        self.llm = BaseLLMWrapper(model=llm_model, horizon_length=horizon_length)
        self.env = environment
        self.last_action = None
        self.last_result = None

    def _forward_inference(self, input_dict):
        action_letter = self.llm.get_action(self.env, agent_idx=1, last_action=self.last_action, last_result=self.last_result)
        self.last_action = action_letter
        action_index = ACTION_MAPPING.get(action_letter, 4)
        return {"actions": [action_index]}
