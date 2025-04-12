from collections import defaultdict
import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from llm_agent import OvercookedLLM

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


class LLMAgent(RLModule):
    def __init__(self, *args, environment, llm_model="openai/gpt-4o",  **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_agent = OvercookedLLM(llm_model=llm_model)
        self.environment = environment

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        action = self.llm_agent.get_action(self.environment)
        
        action_mapping = {
            "w": 3,  # UP
            "d": 0,  # RIGHT
            "a": 2,  # LEFT
            "s": 1,  # DOWN
            "q": 4   # STAY
        }
        
        numeric_action = action_mapping.get(action, 4)
        return {Columns.ACTIONS: np.array([numeric_action])}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "LLMAgent is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
        
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]
