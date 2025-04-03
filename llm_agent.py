import re
import numpy as np
import litellm
        
class OvercookedLLM:
    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.model = llm_model

    def extract_items_from_observation(self, observations):
        print("Observation length:", len(observations))
        
        items = []
        i = 0
        
        while i < len(observations) - 7:
            x = observations[i]
            y = observations[i+1]
            i += 2
            
            if i < len(observations) - 7:
                status = observations[i]
                if 0 <= status <= 1:
                    items.append({"x": x, "y": y, "status": status})
                    i += 1
                else:
                    items.append({"x": x, "y": y})
            else:
                items.append({"x": x, "y": y})
        
        agent_count = 0
        food_count = {'tomato': 0, 'lettuce': 0, 'onion': 0}
        plate_count = 0
        knife_count = 0
        delivery_count = 0
        
        for idx, item in enumerate(items):
            if idx < 2:
                # first items are agents
                item["type"] = "agent"
                item["subtype"] = "human" if idx == 0 else "ai"
                agent_count += 1
            elif "status" in item and idx < 5:
                if food_count["tomato"] == 0:
                    item["type"] = "food"
                    item["subtype"] = "tomato"
                    food_count["tomato"] += 1
                elif food_count["lettuce"] == 0:
                    item["type"] = "food"
                    item["subtype"] = "lettuce"
                    food_count["lettuce"] += 1
                elif food_count["onion"] == 0:
                    item["type"] = "food"
                    item["subtype"] = "onion"
                    food_count["onion"] += 1
                else:
                    item["type"] = "food"
                    item["subtype"] = "unknown"
            elif plate_count < 1:
                item["type"] = "plate"
                plate_count += 1
            elif knife_count < 1:
                item["type"] = "knife"
                knife_count += 1
            elif delivery_count < 1:
                item["type"] = "delivery"
                delivery_count += 1
            else:
                item["type"] = "unknown"
        
        # Extract one-hot task encoding
        task_encoding = observations[-7:]
        task_idx = np.argmax(task_encoding) if np.max(task_encoding) > 0 else -1
        
        print(f"Processed {len(items)} items from observation vector")
        for idx, item in enumerate(items):
            item_type = item.get("type", "unknown")
            item_subtype = item.get("subtype", "")
            status_info = f", status: {item['status']:.1f}" if "status" in item else ""
            print(f"Item {idx}: {item_type} {item_subtype} at ({item['x']:.2f}, {item['y']:.2f}){status_info}")
        
        return items, task_idx
    
    def create_prompt(self, observations: np.ndarray) -> str:
        grid_size = 5  # Default grid size
        
        items_data, task_idx = self.extract_items_from_observation(observations)
        
        items_info = []
        agent_positions = []
        
        for item in items_data:
            x = int(item["x"] * grid_size)
            y = int(item["y"] * grid_size)
            item_type = item.get("type", "unknown")
            
            if item_type == "agent":
                agent_positions.append((x, y))
                agent_role = item.get("subtype", "unknown")
                holding_status = ""
                if "status" in item:
                    holding_status = f", holding item: {'Yes' if item['status'] > 0 else 'No'}"
                items_info.append(f"- {agent_role.capitalize()} agent: position ({x}, {y}){holding_status}")
                
            elif item_type == "food":
                food_type = item.get("subtype", "unknown")
                chopped_status = ""
                if "status" in item:
                    chopped_status = f", chopped progress: {item['status']:.1f}"
                    chopped_text = "Chopped" if item['status'] >= 1.0 else "Fresh"
                    items_info.append(f"- {chopped_text} {food_type.capitalize()}: position ({x}, {y}){chopped_status}")
                else:
                    items_info.append(f"- {food_type.capitalize()}: position ({x}, {y})")
                    
            elif item_type == "plate":
                containing_status = ""
                if "status" in item:
                    containing_status = f", contains food: {'Yes' if item['status'] > 0 else 'No'}"
                items_info.append(f"- Plate: position ({x}, {y}){containing_status}")
                
            elif item_type == "knife":
                holding_status = ""
                if "status" in item:
                    holding_status = f", holding food: {'Yes' if item['status'] > 0 else 'No'}"
                items_info.append(f"- Cutting board: position ({x}, {y}){holding_status}")
                
            elif item_type == "delivery":
                items_info.append(f"- Delivery counter: position ({x}, {y})")
                
            else:
                status_info = f", status: {item['status']:.1f}" if "status" in item else ""
                items_info.append(f"- Unknown item: position ({x}, {y}){status_info}")

        # print(items_info)
        
        task_names = [
            "tomato salad", "lettuce salad", "onion salad",
            "lettuce-tomato salad", "onion-tomato salad",
            "lettuce-onion salad", "lettuce-onion-tomato salad"
        ]
        task = task_names[task_idx] if 0 <= task_idx < len(task_names) else "unknown task"
        
        agent_x, agent_y = agent_positions[1] if len(agent_positions) > 1 else (0, 0)
        
        prompt = f"""You are an AI agent playing Overcooked, a cooperative cooking game. Your goal is to prepare and deliver {task}.

Current Game State:
- Your position: ({agent_x}, {agent_y})
- Grid size: {grid_size}x{grid_size}

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
    
    def get_action(self, observations: np.ndarray) -> str:
        # print("Raw observation:", observations)
        
        try:
            prompt = self.create_prompt(observations)
            # print(prompt)

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
