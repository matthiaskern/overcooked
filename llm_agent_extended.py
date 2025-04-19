import litellm
import re
from collections import deque
from environment.items import Plate, Food

class LLMPlannerExecutorAgent:
    def __init__(self, model="openai/gpt-4o", memory_limit=10):
        self.model = model
        self.planner_memory = deque(maxlen=memory_limit)
        self.executor_memory = deque(maxlen=memory_limit)
        self.high_level_plan = None

    def create_context(self, env, last_action=None, last_result=None):
        agent = env.agent[1]
        task = env.task[0]
        state_summary = self._summarize_state(env)

        context = f"""You are the AI agent in Overcooked. Task: {task}

Your position: ({agent.x}, {agent.y})
Holding: {self._describe_holding(agent)}

Last action: {last_action if last_action else "None"}
Result: {last_result if last_result else "N/A"}

Plan:
{self.high_level_plan if self.high_level_plan else "None yet"}

Environment:
{state_summary}

Memory:
{"; ".join(self.executor_memory)}

NOTE: To pick up, chop, serve, or interact with any item — just MOVE into the item's direction. The [q] action does nothing in this game.

Respond with:
Thought: ...
Plan: ...
Action: [w/a/s/d] - [explanation]
Verify: [yes/no] - [explanation]
"""
        return context

    def _summarize_state(self, env):
        parts = []
        for item in env.itemList:
            if isinstance(item, Food):
                status = f"Chopped: {item.chopped} ({item.cur_chopped_times}/{item.required_chopped_times})"
            elif isinstance(item, Plate):
                status = f"Contains: {', '.join([f.rawName for f in item.containing])}" if item.containing else "Empty"
            else:
                status = "N/A"
            parts.append(f"{item.rawName} at ({item.x},{item.y}) - {status}")
        return "\n".join(parts)

    def _describe_holding(self, agent):
        if agent.holding is None:
            return "Nothing"
        if isinstance(agent.holding, Food):
            return f"{agent.holding.rawName} (Chopped: {agent.holding.chopped})"
        if isinstance(agent.holding, Plate):
            if agent.holding.containing:
                return f"Plate with: {', '.join(f.rawName for f in agent.holding.containing)}"
            else:
                return "Empty Plate"
        return "Unknown"

    def call_llm(self, messages):
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def update_plan(self, env):
        task = env.task[0]
        plan_prompt = f"""You are a planner for the Overcooked game.

Task: {task}
Given the current environment and task, list a high-level step-by-step plan to complete it.

Respond in format:
1. ...
2. ...
3. ...
"""
        messages = [{"role": "user", "content": plan_prompt}]
        plan = self.call_llm(messages)
        self.planner_memory.append(plan)
        self.high_level_plan = plan
        return plan

    def get_validated_action(self, env, last_action=None, last_result=None):
        context = self.create_context(env, last_action, last_result)
        messages = [{"role": "user", "content": context}]
        reply = self.call_llm(messages)
        self.executor_memory.append(f"Context: {last_action} => {reply}")

        action_match = re.search(r"Action:\s*([wsad])", reply)
        verify_match = re.search(r"Verify:\s*(yes|no)", reply, re.IGNORECASE)

        if verify_match and verify_match.group(1).strip().lower() == "yes":
            if action_match:
                return action_match.group(1)
        return "q"
