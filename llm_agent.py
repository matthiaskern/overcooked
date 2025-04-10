import re
import numpy as np
import litellm
from environment.state_decoder import decode_vector_observation

class OvercookedLLM:
    def __init__(self, llm_model: str = "openai/gpt-4o"):
        self.model = llm_model

    def create_prompt(self, observations: np.ndarray, last_response: str = None, confirm: bool = False) -> str:
        grid_size = 5
        state = decode_vector_observation(observations, grid_size)
        items_info = [
            f"- Tomato: position ({state['tomato']['x']}, {state['tomato']['y']}), chopped progress: {state['tomato']['progress']:.1f}",
            f"- Lettuce: position ({state['lettuce']['x']}, {state['lettuce']['y']}), chopped progress: {state['lettuce']['progress']:.1f}",
            f"- Onion: position ({state['onion']['x']}, {state['onion']['y']}), chopped progress: {state['onion']['progress']:.1f}",
            f"- Plate 1: position ({state['plate1']['x']}, {state['plate1']['y']})",
            f"- Plate 2: position ({state['plate2']['x']}, {state['plate2']['y']})",
            f"- Knife 1: position ({state['knife1']['x']}, {state['knife1']['y']})",
            f"- Knife 2: position ({state['knife2']['x']}, {state['knife2']['y']})",
            f"- Delivery counter: position ({state['delivery']['x']}, {state['delivery']['y']})",
            f"- Human agent: position ({state['human']['x']}, {state['human']['y']})"
        ]
        task = "tomato salad"
        if confirm and last_response:
            prompt = f"""You previously suggested: {last_response}
Make a detailed plan of the steps you are going to do next. Then, execute the following based on your current position.
Will this action bring us closer to completing the task: "{task}"?
Respond with "yes" or "no" and explain briefly."""
        else:
            prompt = f"""You are an AI agent playing Overcooked, a cooperative cooking game. Your goal is to prepare and deliver {task}.

Current Game State:
- Your position: ({state['ai']['x']}, {state['ai']['y']})
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
