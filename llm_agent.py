import re
import numpy as np
import litellm
from environment.items import Plate, Food

class OvercookedLLM:
    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.model = llm_model

    def create_prompt_from_environment(self, env):
        grid_size = (env.xlen, env.ylen)
        
        task_idx = np.argmax(env.oneHotTask) if np.max(env.oneHotTask) > 0 else -1
        task_names = [
            "tomato salad", "lettuce salad", "onion salad",
            "lettuce-tomato salad", "onion-tomato salad",
            "lettuce-onion salad", "lettuce-onion-tomato salad"
        ]
        task = task_names[task_idx] if 0 <= task_idx < len(task_names) else "unknown task"
        
        ai_agent = env.agent[1] if len(env.agent) > 1 else None
        
        if not ai_agent:
            raise Exception("AI Agent not found in env")
        
        ai_x, ai_y = ai_agent.x, ai_agent.y
        
        items_info = []
        
        for idx, agent in enumerate(env.agent):
            agent_role = "Human" if idx == 0 else "AI"
            holding_status = f", holding item: {'Yes' if agent.holding else 'No'}"
            items_info.append(f"- {agent_role} agent: position ({agent.x}, {agent.y}){holding_status}")
            
            if agent.holding:
                holding_info = f"  └─ Holding: "
                if isinstance(agent.holding, Food):
                    holding_info += f"{agent.holding.rawName.capitalize()}"
                    if agent.holding.chopped:
                        holding_info += " (Chopped)"
                    else:
                        holding_info += f" (Chopping progress: {agent.holding.cur_chopped_times}/{agent.holding.required_chopped_times})"
                elif isinstance(agent.holding, Plate):
                    holding_info += "Plate"
                    if agent.holding.containing:
                        holding_info += " containing: "
                        food_items = []
                        for food in agent.holding.containing:
                            food_items.append(f"{food.rawName.capitalize()} (Chopped)" if food.chopped else f"{food.rawName.capitalize()}")
                        holding_info += ", ".join(food_items)
                items_info.append(holding_info)
        
        for tomato in env.tomato:
            status = f", chopped progress: {tomato.cur_chopped_times}/{tomato.required_chopped_times}"
            chopped_text = "Chopped" if tomato.chopped else "Fresh"
            items_info.append(f"- {chopped_text} Tomato: position ({tomato.x}, {tomato.y}){status}")
            
        for lettuce in env.lettuce:
            status = f", chopped progress: {lettuce.cur_chopped_times}/{lettuce.required_chopped_times}"
            chopped_text = "Chopped" if lettuce.chopped else "Fresh"
            items_info.append(f"- {chopped_text} Lettuce: position ({lettuce.x}, {lettuce.y}){status}")
            
        for onion in env.onion:
            status = f", chopped progress: {onion.cur_chopped_times}/{onion.required_chopped_times}"
            chopped_text = "Chopped" if onion.chopped else "Fresh"
            items_info.append(f"- {chopped_text} Onion: position ({onion.x}, {onion.y}){status}")
        
        for plate in env.plate:
            containing_status = ""
            if plate.containing:
                containing_status = ", contains: "
                food_items = []
                for food in plate.containing:
                    food_items.append(f"{food.rawName.capitalize()} (Chopped)" if food.chopped else f"{food.rawName.capitalize()}")
                containing_status += ", ".join(food_items)
            items_info.append(f"- Plate: position ({plate.x}, {plate.y}){containing_status}")
        
        for knife in env.knife:
            holding_info = ""
            if knife.holding:
                if isinstance(knife.holding, Food):
                    food = knife.holding
                    progress = f", chopping progress: {food.cur_chopped_times}/{food.required_chopped_times}"
                    holding_info = f", holding: {food.rawName.capitalize()}{progress}"
                elif isinstance(knife.holding, Plate):
                    plate = knife.holding
                    if plate.containing:
                        food_items = []
                        for food in plate.containing:
                            food_items.append(f"{food.rawName.capitalize()} (Chopped)" if food.chopped else f"{food.rawName.capitalize()}")
                        holding_info = f", holding: Plate containing {', '.join(food_items)}"
                    else:
                        holding_info = ", holding: Empty plate"
            items_info.append(f"- Cutting board: position ({knife.x}, {knife.y}){holding_info}")
        
        for delivery in env.delivery:
            items_info.append(f"- Delivery counter: position ({delivery.x}, {delivery.y})")
        
        prompt = f"""You are an AI agent playing Overcooked, a cooperative cooking game. Your goal is to prepare and deliver {task}.

Current Game State:
- Your position: ({ai_x}, {ai_y})
- Grid size: {grid_size[0]}x{grid_size[1]}

Items in the environment:
{chr(10).join(items_info)}

Actions:
d: Move RIGHT
s: Move DOWN 
a: Move LEFT
w: Move UP
q: STAY (perform action at current position)

Rules:
1. You need to collect raw ingredients as defined in the task
2. Chop ingredients after collecting by going to the knife's position and using it
3. Place chopped ingredients on a plate
4. Deliver the completed dish to the delivery counter

Respond with a single action letter and a brief explanation of your reasoning.
Format your response as: "Action: [dsawq] - [explanation]"
"""
        return prompt
    
    def parse_response(self, response: str) -> str:
        action_match = re.search(r'Action:\s*([dsawq])', response)
        if action_match:
            return action_match.group(1)
        
        return "q"
    
    def get_action(self, env) -> str:
        try:
            prompt = self.create_prompt_from_environment(env)
            print(prompt)
            
            response = self.call_llm(prompt)
            print(response)
            
            action = self.parse_response(response)
            return action
        except Exception as e:
            print(f"Error in get_action: {e}")
            return "q"

    def call_llm(self, prompt: str) -> str:
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Action: q - Error"
