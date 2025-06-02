import re
import litellm
from collections import deque
from environment.items import Agent, Delivery, Food, Knife, Plate
from environment.Overcooked import ITEMNAME
import base64
import os
import logging
import asyncio

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

    async def call_llm_with_full_context(self, prompt, image_path):
        with open(image_path, "rb") as img_file:
            b64_image = base64.b64encode(img_file.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }]

        # logger.info(f"[{self.agent_role}] ===CALLING LLM===")
        # logger.info(f"[{self.agent_role}] {prompt}")

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.3
        )
        

        result = response.choices[0].message.content
        logger.info(f"[{self.agent_role}] ===RESPONSE FROM LLM===")
        logger.info(f"[{self.agent_role}] {result}")

        return result

    def extract_action(self, llm_output):
        match = re.search(r"Action:\s*\[?([wsadq])\]?", llm_output, re.IGNORECASE)
        return match.group(1).lower() if match else "q"

    async def get_action(self, env, image_path, agent_idx, last_action=None, last_result=None):
        agent = env.agent[agent_idx]
        task = ", ".join(env.task) if isinstance(env.task, list) else env.task
        
        self.agent_role = "Human (not blue)" if agent_idx == 0 else "AI (blue)"

        if not self.plan:
            await self.update_plan(task)
            logger.info(f"[{self.agent_role}] [ERROR] No plan was generated.")
            self.plan = "No plan available."

        logger.info(f"[{self.agent_role}] [DEBUG] Current Plan:\n" + str(self.plan))

        holding = self._describe_holding(agent)
        state_summary = self._describe_env(env)
        logger.info(f"[{self.agent_role}] {str(state_summary)} ({agent.y},{agent.x})")
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

Your position: ({agent.y}, {agent.x})
Holding: {holding}

Past results:
{memory_text}

Plan:
{self.plan}

Environment (Summary):
{state_summary}


INSTRUCTIONS:

- Your immediate goal is one of: [get ingredient, chop ingredient, pick plate, plate food, deliver dish].
- Construct a ROUTE of exactly {self.horizon_length} steps: list of (x, y) next steps needed to reach a tile next to the target (not directly on the target if it's an item).
- {agent_identity}
- The other player will move at the same time as you.
- You cannot move where onto a tile where another player already is.
- You cannot move into counters.
- You cannot exchange items with other players directly, i.e. you need to place food on a counter before plating or need to have a plate on a counter before putting food on it.
IMPORTANT:
- When plating, the player who first has the plate should plate and deliver.
- [w]: Move up or perform an action on the tile above
- [a]: Move left or perform an action on the tile to the left
- [s]: Move down or perform an action on the tile below
- [d]: Move right or perform an action on the tile to the right
- [q]: Do nothing
- Always evaluate, what is the most optimal action for me to take to complete the task, and assume that the other player will also take the most optimal action. 
- We always need to be planning one step ahead, as each player's next action will be executed simultaneously.'
- If the previous action failed to move, re-evaluate and fix your decision.


Respond strictly in the following format:
Observation: I am at position (x/y), on top of me is a counter, left of me is free, right of me is player... 
Thought: I am seeing ..., my current position is (x/y), my current goal is Y, my last action was X, the other player's last action was Y, the other player is likely doing X or moving to (x/y), therefore I should ...
Plan: ...
Route: ...
Action: [w/a/s/d/q] - [reason]
Verify: [yes/no] - [reason]
"""

            response = await self.call_llm_with_full_context(prompt, image_path)
            self.executor_memory.append(response)

            action_letter = self.extract_action_and_verify(response)

            route_match = re.search(r"Route:\s*\[([^\]]*)\]", response, re.IGNORECASE)
            route = route_match.group(1).strip() if route_match else "[]"

            logger.info(f"[{self.agent_role}] NEW ROUTE: {route}")

            return action_letter
        logger.info(f"[{self.agent_role}] [ERROR] Agent failed to generate valid move after retries. Defaulting to [q].")
        return "q"

    def extract_action_and_verify(self, text):
        action_match = re.search(r"Action:\s*\[?([wsad])\]?", text, re.IGNORECASE)
        verify_match = re.search(r"Verify:\s*(yes|no)", text, re.IGNORECASE)

        action = action_match.group(1).lower() if action_match else None
        verified = verify_match.group(1).lower() if verify_match else "no"

        return action if verified == "yes" and action else "w"

    async def update_plan(self, task):
        #image_path = f"step_{self.image_step_counter}.png"
        image_path = "step.png"
        if not os.path.exists(image_path):
            logger.info(f"[{self.agent_role}] [Planner ERROR] Image file not found: {image_path}")
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
            "- Agent being a human or AI agent\n"
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

        description_response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.2
        )

        visual_description = description_response.choices[0].message.content.strip()
        logger.info(f"[{self.agent_role}] [Vision] What the agent sees:\n" + str(visual_description))

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
            "- You can only plate food when either plate or food is on a counter. You cannot plate food between agents.\n"
            "- Plates and delivery counters may be far apart.\n"
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": planning_prompt},
            ]
        }]


        # logger.info(f"[{self.agent_role}] [Planner] Generating Plan:\n" + planning_prompt)

        plan_response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            max_tokens=10000,
            temperature=0.3
        )

        self.plan = plan_response.choices[0].message.content.strip()
        self.planner_memory.append(self.plan)
        logger.info(f"[{self.agent_role}] [Planner] Generated Plan:\n" + str(self.plan))


    def _describe_holding(self, agent):
        if not agent.holding:
            return "Nothing"
        if isinstance(agent.holding, Food):
            return f"{agent.holding.name}"
        if isinstance(agent.holding, Plate):
            if agent.holding.containing:
                return "Plate with " +  agent.holding.containedName
            return "Empty Plate"
        return "Unknown item"

    def _describe_env(self, env):
        description = []
        description.append("Agents:")
        for i, agent in enumerate(env.agent):
            role = "Human" if i == 0 else "AI"
            holding = self._describe_holding(agent)
            description.append(f"{role} at ({agent.y}, {agent.x}) holding: {holding}")

        description.append("\nMap Grid (x, y):")
        xlen, ylen = env.xlen, env.ylen
        grid = [["" for _ in range(ylen)] for _ in range(xlen)]
        
        for x in range(xlen):
            for y in range(ylen):
                tile_idx = env.map[x][y]
                tile_type = ITEMNAME[tile_idx]
                if tile_type == "counter":
                    grid[y][x] = "COUNTER"
                else:
                    grid[y][x] = "Space"
        
        for item in env.itemList:
            if isinstance(item, Agent):
                continue

            label = item.name
            if isinstance(item, Food) and item.chopped:
                label += " (chopped)"
            elif isinstance(item, Plate) and item.containing:
                contents = item.containedName
                label = f"Plate ({contents})"
            
            current_content = "COUNTER" if grid[item.y][item.x] == "Space" else grid[item.y][item.x]
            grid[item.y][item.x] = f"{current_content} + {label}"
        
        for i, agent in enumerate(env.agent):
            role = "Human" if i == 0 else f"AI{i}"
            grid[agent.y][agent.x] = role + f" holding {self._describe_holding(agent)}"
        
        for x in range(xlen):
            row = []
            for y in range(ylen):
                row.append(f"[{y},{x}] {grid[y][x]:<18}")
            description.append(" ".join(row))
        return "\n".join(description)

