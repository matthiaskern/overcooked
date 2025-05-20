from collections import defaultdict, deque
import numpy as np
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from llm_agent import BaseLLMWrapper
from environment.items import Food, Plate, Knife, Delivery
from environment.Overcooked import ITEMNAME
import litellm
import re

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

ACTION_MAPPING = {
    "d": 0,
    "s": 1,
    "a": 2,
    "w": 3,
    "q": 4
}


class LLMAgent(RLModule):
    def __init__(self, observation_space, action_space, inference_only=True, llm_model=None, environment=None, horizon_length=3):
        super().__init__()
        self.llm = BaseLLMWrapper(model=llm_model, horizon_length=horizon_length)
        self.env = environment
        self.last_action = None
        self.last_result = None

    def _forward_inference(self, input_dict):
        action_letter = self.llm.get_action(self.env, agent_idx=1, last_action=self.last_action, last_result=self.last_result)
        self.last_action = action_letter
        action_index = ACTION_MAPPING.get(action_letter, 4)
        return {"actions": [action_index]}
