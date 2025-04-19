import litellm

class LLMClient:
    def __init__(self, model: str = "openai/gpt-4o", system_prompt: str = None):
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def reset(self):
        self.messages = self.messages[:1] if self.messages and self.messages[0]["role"] == "system" else []

    def call(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = litellm.completion(
                model=self.model,
                messages=self.messages,
                max_tokens=200,
                temperature=0.7
            )
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            return f"Error calling LLM: {e}"
