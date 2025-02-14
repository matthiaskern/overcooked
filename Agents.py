from collections import defaultdict

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

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