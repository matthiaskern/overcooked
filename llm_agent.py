import re
import numpy as np
import litellm

class OvercookedLLM:
    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.model = llm_model

    def create_prompt(self, observations: np.ndarray, last_response: str = None, confirm: bool = False) -> str:
        grid_size = 5
        items_info = []

        try:
            tomato_x = int(observations[0] * grid_size)
            tomato_y = int(observations[1] * grid_size)
            tomato_status = observations[2]
            items_info.append(f"- Tomato: position ({tomato_x}, {tomato_y}), chopped progress: {tomato_status:.1f}")

            lettuce_x = int(observations[3] * grid_size)
            lettuce_y = int(observations[4] * grid_size)
            lettuce_status = observations[5]
            items_info.append(f"- Lettuce: position ({lettuce_x}, {lettuce_y}), chopped progress: {lettuce_status:.1f}")

            onion_x = int(observations[6] * grid_size)
            onion_y = int(observations[7] * grid_size)
            onion_status = observations[8]
            items_info.append(f"- Onion: position ({onion_x}, {onion_y}), chopped progress: {onion_status:.1f}")

            plate1_x = int(observations[9] * grid_size)
            plate1_y = int(observations[10] * grid_size)
            plate2_x = int(observations[11] * grid_size)
            plate2_y = int(observations[12] * grid_size)
            items_info.append(f"- Plate 1: position ({plate1_x}, {plate1_y})")
            items_info.append(f"- Plate 2: position ({plate2_x}, {plate2_y})")

            knife1_x = int(observations[13] * grid_size)
            knife1_y = int(observations[14] * grid_size)
            knife2_x = int(observations[15] * grid_size)
            knife2_y = int(observations[16] * grid_size)
            items_info.append(f"- Knife 1: position ({knife1_x}, {knife1_y})")
            items_info.append(f"- Knife 2: position ({knife2_x}, {knife2_y})")

            delivery_x = int(observations[17] * grid_size)
            delivery_y = int(observations[18] * grid_size)
            items_info.append(f"- Delivery counter: position ({delivery_x}, {delivery_y})")

            agent1_x = int(observations[19] * grid_size)
            agent1_y = int(observations[20] * grid_size)
            agent2_x = int(observations[21] * grid_size)
            agent2_y = int(observations[22] * grid_size)

            items_info.append(f"- Human agent: position ({agent1_x}, {agent1_y})")
            task = "tomato salad"

        except (IndexError, ValueError):
            raise Exception("Error parsing observation")

        if confirm and last_response:
            prompt = f"""You previously suggested: {last_response}
Make a detailed plan of the steps you are going to do next. Then, execute the following based on your current position.
Will this action bring us closer to completing the task: "{task}"?
Respond with "yes" or "no" and explain briefly."""
        else:
            prompt = f"""You are an AI agent playing Overcooked, a cooperative cooking game. Your goal is to prepare and deliver {task}.

Current Game State:
- Your position: ({agent2_x}, {agent2_y})
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
2. Chop ingredients using knives
3. Place chopped ingredients on a plate
4. Deliver the completed dish to the delivery counter

Respond with a single action letter and a brief explanation of your reasoning.
Format your response as: "Action: [dsawq] - [explanation]\""""
        return prompt

    def parse_response(self, response: str) -> str:
        action_match = re.search(r'Action:\s*([dsawq])', response)
        if action_match:
            return action_match.group(1)
        return "q"

    def confirm_action(self, reasoning: str) -> bool:
        prompt = self.create_prompt(np.zeros(23), last_response=reasoning, confirm=True)
        response = self.call_llm(prompt)
        return 'yes' in response.lower()

    def get_action(self, observations: np.ndarray) -> str:
        reasoning = None
        confirmed = False
        attempts = 0

        while not confirmed and attempts < 5:
            prompt = self.create_prompt(observations)
            reasoning = self.call_llm(prompt)
            confirmed = self.confirm_action(reasoning)
            attempts += 1

        action = self.parse_response(reasoning)
        return action

    def call_llm(self, prompt: str) -> str:
        #("llm called with " + prompt)
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")

            return "Action: q - Error"
