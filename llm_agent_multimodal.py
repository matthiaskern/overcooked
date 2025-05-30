import re
import litellm
from collections import deque
from environment.items import Delivery, Food, Knife, Plate
from environment.Overcooked import ITEMNAME
import base64
import os
import logging

logger = logging.getLogger('multimodal_agent')


class MultiModalOvercookedAgent:
    def __init__(self, model="openai/gpt-4.1", memory_limit=10, horizon_length=3):
        self.model = model
        self.planner_memory = deque(maxlen=memory_limit)
        self.executor_memory = deque(maxlen=memory_limit)
        self.plan = None
        self.horizon_length = horizon_length
        self.image_step_counter = 0
        self.agent_role = None

    def call_llm_with_full_context(self, prompt, image_path):
        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }]

        logger.info("===CALLING LLM===")
        logger.info(prompt)

        response = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.3
        )
        

        result = response.choices[0].message.content
        logger.info("===RESPONSE FROM LLM===")
        logger.info(result)

        return result

    def extract_action(self, llm_output):
        match = re.search(r"Action:\s*\[?([wsadq])\]?", llm_output, re.IGNORECASE)
        return match.group(1).lower() if match else "q"

    def get_action(self, env, image_path, agent_idx, last_action=None, last_result=None):
        agent = env.agent[agent_idx]
        task = ", ".join(env.task) if isinstance(env.task, list) else env.task
        
        self.agent_role = "Human" if agent_idx == 0 else "AI"

        self.update_plan(task)
        if not self.plan:
            logger.info("[ERROR] No plan was generated.")
            self.plan = "No plan available."

        logger.info("[DEBUG] Current Plan:\n" + str(self.plan))

        holding = self._describe_holding(agent)
        state_summary = self._describe_env(env)
        logger.info(str(self.plan))
        logger.info("STATE SUMMARY")
        logger.info(str(state_summary))
        logger.info("DEBUG")
        debug_full_state = self._describe_env_debug(env)
        logger.info(str(debug_full_state))
        rendered_grid = self._render_grid(env)
        logger.info("GRID")
        logger.info(str(rendered_grid))
        possible_moves = self._describe_possible_moves(env, agent)
        memory_text = "\n".join(self.executor_memory)

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

            agent_identity = f"YOU ARE THE {self.agent_role.upper()} PLAYER!"
            
            prompt = f"""You are the {self.agent_role} agent in Overcooked. Task: {task}

Your position: ({agent.x}, {agent.y})
Holding: {holding}

Last action: {last_action or "None"}
Result: {last_result or "None"} — Movement status: {moved}

Plan:
{self.plan}

Environment (Summary):
{state_summary}



Memory:
{memory_text}

INSTRUCTIONS:

- Your immediate goal is one of: [get ingredient, chop ingredient, pick plate, plate food, deliver dish].
- Construct a ROUTE of exactly {self.horizon_length} steps: list of (row, col) next steps needed to reach a tile next to the target (not directly on the target if it's an item).
- You will also get an image of the game state. Inspect it carefully. You are the {self.agent_role.lower()}. 
- {agent_identity}
- You cannot move where onto a tile where another player already is.
- You cannot move into counters.
- Collaborate with other players as needed.
IMPORTANT:
- [w]: Move up or perform an action on the tile above
- [a]: Move left or perform an action on the tile to the left
- [s]: Move down or perform an action on the tile below
- [d]: Move right or perform an action on the tile to the right
- [q]: Do nothing
- Remember, coords are not given in x,y but in row/col. (i.e. to navigate vertically we need to change x and when we navigate horizontally we change y)
- If the previous action failed to move, re-evaluate and fix your decision.


Respond strictly in the following format:
Thought: I am seeing ..., my current position is X, my current goal is Y, the other player is doing X, therefore I should ...
Plan: ...
Route: ...
Action: [w/a/s/d/q] - [reason]
Verify: [yes/no] - [reason]
"""

            response = self.call_llm_with_full_context(prompt, image_path)
            self.executor_memory.append(response)

            action_letter = self.extract_action_and_verify(response)

            route_match = re.search(r"Route:\s*\[([^\]]*)\]", response, re.IGNORECASE)
            route = route_match.group(1).strip() if route_match else "[]"

            logger.info(f"NEW ROUTE: {route}")

            return action_letter
        logger.info("[ERROR] Agent failed to generate valid move after retries. Defaulting to [q].")
        return "q"

    def extract_action_and_verify(self, text):
        action_match = re.search(r"Action:\s*\[?([wsad])\]?", text, re.IGNORECASE)
        verify_match = re.search(r"Verify:\s*(yes|no)", text, re.IGNORECASE)

        action = action_match.group(1).lower() if action_match else None
        verified = verify_match.group(1).lower() if verify_match else "no"

        return action if verified == "yes" and action else "w"

    def update_plan(self, task):
        #image_path = f"step_{self.image_step_counter}.png"
        image_path = "step.png"
        if not os.path.exists(image_path):
            logger.info(f"[Planner ERROR] Image file not found: {image_path}")
            self.plan = "No plan generated (missing image)"
            return

        with open(image_path, "rb") as image_file:
            b64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_data_url = f"data:image/png;base64,{b64_image}"

        description_prompt = (
            "You are an Overcooked vision system.\n"
            "Describe what you see in the image: items, agents, locations of interest.\n"
            "Be specific about:\n"
            "- Ingredients: tomato, chopped tomato, plates\n"
            "- Tools: knives/cutting boards\n"
            "- Delivery counter (usually marked with *)\n"
            "- Agent positions and what they are holding (if visible)\n"
            "Format:\n"
            "Observation: ...\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": description_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        }]

        description_response = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.2
        )

        visual_description = description_response.choices[0].message.content.strip()
        logger.info("[Vision] What the agent sees:\n" + str(visual_description))

        agent_identity = self.agent_role.lower()
        
        planning_prompt = (
            "You are a planner for Overcooked.\n\n"
            f"Task: {task}\n\n"
            "Observation:\n"
            f"{visual_description}\n\n"
            "Now generate a high-level plan in steps.\n\n"
            f"You are the {agent_identity} player."
            f"The other players have the same task as you, you need to collaborate to complete the task."
            f"You cannot communicate with the other players."
            f"- Plan for approximately {self.horizon_length} key steps ahead.\n"
            "NOTE:\n"
            "- Move toward an object to interact with it (e.g., pick up, chop, plate).\n"
            "- [q] does nothing.\n"
            "- Chopping takes multiple moves.\n"
            "- Food must be placed on a plate before delivery.\n"
            "- Plates and delivery counters may be far apart.\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": planning_prompt},
            ]
        }]


        logger.info("[Planner] Generating Plan:\n" + planning_prompt)

        plan_response = litellm.completion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.3
        )

        self.plan = plan_response.choices[0].message.content.strip()
        self.planner_memory.append(self.plan)
        logger.info("[Planner] Generated Plan:\n" + str(self.plan))


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
        description.append("Agents:")
        for i, agent in enumerate(env.agent):
            role = "Human" if i == 0 else "AI"
            holding = self._describe_holding(agent)
            description.append(f"{role} at ({agent.x}, {agent.y}) holding: {holding}")
        description.append("\nMap Grid (x rows, y cols):")
        xlen, ylen = env.xlen, env.ylen
        grid = [["" for _ in range(ylen)] for _ in range(xlen)]
        for x in range(xlen):
            for y in range(ylen):
                tile_idx = env.map[x][y]
                tile_type = ITEMNAME[tile_idx]
                grid[x][y] = tile_type.upper() if tile_type == "counter" else tile_type.capitalize()
        for item in env.itemList:
            label = item.rawName.capitalize()
            if isinstance(item, Food) and item.chopped:
                label += " (chopped)"
            elif isinstance(item, Plate) and item.containing:
                contents = ", ".join(f.rawName for f in item.containing)
                label = f"Plate ({contents})"
            grid[item.x][item.y] = label
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
            new_row = agent.y + dy
            new_col = agent.x + dx
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
