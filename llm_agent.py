import re
import litellm
from collections import deque
from environment.items import Food, Plate


class BaseLLMWrapper:
    def __init__(self, model="openai/gpt-4o", memory_limit=10):
        self.model = model
        self.planner_memory = deque(maxlen=memory_limit)
        self.executor_memory = deque(maxlen=memory_limit)
        self.plan = None

    def call(self, messages):
        try:
            print("\n=== LLM Prompt ===")
            print(messages[-1]["content"])
            print("==================\n")

            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
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
        prompt = f"""You are a planner for Overcooked.

    Task: {task}

    Generate a high-level plan in steps.

    IMPORTANT RULES:
    - You MUST put chopped ingredients onto a plate before serving.
    - You CANNOT serve chopped ingredients alone.
    - Delivery is ONLY valid when you move a plate (with ingredients) into the delivery tile (⭐).
    - Plates are labeled 'P' on the map.
    - Delivery areas are labeled '*' on the map.

    NOTE: To pick up, chop, serve, or interact — just MOVE into the item's direction. The [q] action does nothing in this game.
    """
        self.plan = self.call([{"role": "user", "content": prompt}])
        self.planner_memory.append(self.plan)


    def get_action(self, env, agent_idx, last_action=None, last_result=None):
        agent = env.agent[agent_idx]
        task = ", ".join(env.task) if isinstance(env.task, list) else env.task

        if not self.plan:
            self.update_plan(task)

        state = self._describe_env(env)
        holding = self._describe_holding(agent)
        memory_text = "\n".join(self.executor_memory)
        grid = self._render_grid(env)

        prompt = f"""You are the AI agent in Overcooked.

    Task: {task}

    Your position: ({agent.x}, {agent.y})
    Holding: {holding}

    Last action: {last_action or "None"}
    Result: {last_result or "None"}

    Plan:
    {self.plan}

    Environment:
    {state}

    Grid View:
    {grid}

    Memory:
    {memory_text}

    IMPORTANT RULES:
    - If holding chopped ingredients, move into a Plate (P) to place the food.
    - If holding a Plate with food, move into the Delivery (⭐) to serve it.
    - Delivery is ONLY possible with a plate, NOT directly with food.
    - If standing directly on an item, move off first, then approach again from side.

    NOTE: To pick up, chop, serve, or interact — MOVE into adjacent item's tile. The [q] action does nothing.

    Respond with:
    Thought: ...
    Plan: ...
    Action: [w/a/s/d] - [explanation]
    Verify: [yes/no] - [explanation]
    """
        response = self.call([{"role": "user", "content": prompt}])
        self.executor_memory.append(response)
        return self.extract_action_and_verify(response)

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
        parts = []
        for item in env.itemList:
            if isinstance(item, Food):
                status = f"Chopped: {item.chopped} ({item.cur_chopped_times}/{item.required_chopped_times})"
            elif isinstance(item, Plate):
                contents = ", ".join(f.rawName for f in item.containing) if item.containing else "Empty"
                status = f"Contains: {contents}"
            else:
                status = "N/A"
            parts.append(f"{item.rawName} at ({item.x}, {item.y}) - {status}")
        return "\n".join(parts)

    def _render_grid(self, env):
        width = env.width if hasattr(env, "width") else 5
        height = env.height if hasattr(env, "height") else 5
        grid = [["." for _ in range(width)] for _ in range(height)]
        for agent in env.agent:
            grid[agent.y][agent.x] = "A"
        for item in env.itemList:
            symbol = item.rawName[0].upper()
            if isinstance(item, Plate):
                symbol = "P"
            if isinstance(item, Food):
                symbol = item.rawName[0].lower()
            if item.rawName.lower() == "delivery":
                symbol = "*"
            grid[item.y][item.x] = symbol
        return "\n".join("".join(row) for row in grid)
