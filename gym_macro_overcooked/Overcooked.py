from math import trunc
import gymnasium as gym
import numpy as np
from .render.game import Game
from gymnasium import spaces
from .items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import Counter

DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX = {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
AGENTCOLOR = ["blue", "robot", "green", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]


class Overcooked_multi(MultiAgentEnv):

    def __init__(self, grid_dim, task, rewardList, map_type="A", n_agent=2, obs_radius=2, mode="vector", debug=False):
        super().__init__()
        self.step_count = 0
        self.agents = self.possible_agents = ["human", "ai"]

        self.xlen, self.ylen = grid_dim

        self.task = task
        self.rewardList = rewardList
        self.mapType = map_type
        self.debug = debug
        self.n_agent = n_agent
        self.mode = mode
        self.obs_radius = obs_radius

        self.reward = None

        self.initMap = self._initialize_map()
        self.map = copy.deepcopy(self.initMap)

        self.oneHotTask = [1 if t in self.task else 0 for t in TASKLIST]

        counter = Counter(self.task)
        self.taskCompletionStatus = [counter[element] if element in counter else 0 for element in TASKLIST]

        self._createItems()
        self.n_agent = len(self.agent)

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        self._initObs()
        self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(len(self._get_obs()[agent]),), dtype=np.float64) for agent in self.agents}

        self.game = Game(self)

    def _initialize_map(self):
        if self.xlen == 3 and self.ylen == 3:
            return self._map_3x3()
        elif self.xlen == 5 and self.ylen == 5:
            return self._map_5x5()
        elif self.xlen == 3 and self.ylen == 5:
            return self._map_3x5()
        elif self.xlen == 7 and self.ylen == 7:
            return self._map_7x7()
        elif self.xlen == 9 and self.ylen == 9:
            return self._map_9x9()
        else:
            return []
    
    def _map_3x3(self):
        if self.n_agent == 2:
            if self.mapType == "A":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]] 
            elif self.mapType == "B":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]] 
            elif self.mapType == "C":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
            elif self.mapType == "B":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
            elif self.mapType == "C":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
        return map
    
    def _map_5x5(self):
        if self.n_agent == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1],
                        [6, 0, 0, 2, 1],
                        [3, 0, 0, 0, 1],
                        [7, 0, 0, 2, 1],
                        [1, 5, 1, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 8, 1, 1, 1],
                        [6, 2, 1, 0, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 4, 1, 1, 1]] 
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        elif self.n_agent == 3:
            if self.mapType == "A":
                map =  [[1, 1, 5, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 6],
                        [7, 0, 2, 0, 1],
                        [1, 1, 5, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 1, 1, 1, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 2, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        return map
    
    def _map_3x5(self):
        if self.n_agent == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 1],
                        [7, 0, 0, 0, 1],
                        [1, 5, 1, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 8, 1, 1, 1],
                        [6, 2, 1, 0, 1],
                        [3, 0, 5, 2, 6]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        elif self.n_agent == 3:
            if self.mapType == "A":
                map =  [[1, 1, 5, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 6],
                        [7, 0, 2, 0, 1],
                        [1, 1, 5, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 1, 1, 1, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 2, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
            return map
        
    def _map_7x7(self):
        if self.n_agent == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 4],
                        [6, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 5, 1, 0, 1, 4, 1],
                        [1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 4, 1, 1, 0, 1],
                        [1, 0, 2, 1, 2, 0, 1],
                        [7, 0, 0, 1, 0, 0, 5],
                        [1, 6, 1, 1, 7, 6, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 2, 1, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]] 
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
        return map

    def _map_9x9(self):
        if self.n_agent == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 5, 1, 0, 1, 1, 4, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 4, 1, 7, 0, 0, 1],
                        [1, 0, 2, 0, 1, 0, 2, 0, 1],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 6, 1, 1, 1, 6, 5, 1, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        elif self.n_agent == 3:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 1, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        return map


    def _createItems(self):
        """
        Initialize the items in the environment based on the map configuration.

        This function iterates through the map grid and creates instances of various items
        (e.g., agents, knives, delivery points, tomatoes, lettuce, onions, plates) at their
        respective positions. It populates the item dictionary and item list with these instances.

        The item dictionary (itemDic) maps item names to lists of item instances, excluding "space" and "counter".
        The item list (itemList) is a flattened list of all item instances.
        The agent list (agent) is a list of all agent instances.

        The function also assigns colors to agents based on their index.

        Returns:
            None
        """
        self.itemDic = {name: [] for name in ITEMNAME if name != "space" and name != "counter"}
        agent_idx = 0

        for x in range(self.xlen):
            for y in range(self.ylen):
                item_type = ITEMNAME[self.map[x][y]]
                if item_type == "agent":
                    self.itemDic[item_type].append(Agent(x, y, color=AGENTCOLOR[agent_idx]))
                    agent_idx += 1
                elif item_type == "knife":
                    self.itemDic[item_type].append(Knife(x, y))
                elif item_type == "delivery":
                    self.itemDic[item_type].append(Delivery(x, y))
                elif item_type == "tomato":
                    self.itemDic[item_type].append(Tomato(x, y))
                elif item_type == "lettuce":
                    self.itemDic[item_type].append(Lettuce(x, y))
                elif item_type == "onion":
                    self.itemDic[item_type].append(Onion(x, y))
                elif item_type == "plate":
                    self.itemDic[item_type].append(Plate(x, y))

        self.itemList = [item for sublist in self.itemDic.values() for item in sublist]
        self.agent = self.itemDic["agent"]


    def _initObs(self):
        """
        Initialize the observations for the agents.

        This function creates an observation list by normalizing the positions of items
        and appending additional information if the item is of type Food. It then extends
        the observation list with one-hot encoded task information. Finally, it assigns
        the observation list to each agent and returns a list of observations for all agents.

        Returns:
            list: A list containing the observation arrays for each agent.
        """
        obs = []
        for item in self.itemList:
            obs.extend([item.x / self.xlen, item.y / self.ylen])
            if isinstance(item, Food):
                obs.append(item.cur_chopped_times / item.required_chopped_times)
        obs.extend(self.oneHotTask)

        for agent in self.agent:
            agent.obs = obs
        return [np.array(obs)] * self.n_agent


    def _get_vector_state(self):
        """
        Get the global state of the environment.

        This function creates a global state representation by normalizing the positions of items
        and appending additional information based on the item type. It includes the positions of
        all items, their chopped status if they are food, whether plates contain food, and whether
        knives or agents are holding items. The state is extended with one-hot encoded task information.

        Returns:
            list: A list containing the state arrays for each agent.
        """
        state = []
        for item in self.itemList:
            x = item.x / self.xlen
            y = item.y / self.ylen
            state.extend([x, y])
            
            if isinstance(item, Food):
                state.append(item.cur_chopped_times / item.required_chopped_times)
            elif isinstance(item, Plate):
                state.append(1 if item.containing else 0)
            elif isinstance(item, Knife):
                state.append(1 if item.holding else 0)
            elif isinstance(item, Agent):
                state.append(1 if item.holding else 0)

        state.extend(self.oneHotTask)
        return [np.array(state)] * self.n_agent


    def _get_image_state(self):
        """
        Retrieve the current image state for each agent.

        This method returns a list containing the current image observation 
        of the game state, repeated for each agent in the environment.

        Returns:
            list: A list where each element is the current image observation 
              of the game state, repeated for the number of agents.
        """
        return [self.game.get_image_obs()] * self.n_agent

    def _get_obs(self):
        """
        Returns
        -------
        obs : dict
            observation for each agent.
        """
        vec_obs = self._get_vector_obs()

        # If observation radius is greater than 0, agents have partial observability
        if self.obs_radius > 0:
            if self.mode == "vector":
                return {agent: np.asarray(vec_obs[i], dtype=np.float64) for i, agent in enumerate(self.agents)}
            elif self.mode == "image":
                return self._get_image_obs()
        else:
            # If observation radius is 0, agents have full observability
            if self.mode == "vector":
                return {agent: np.asarray(self._get_vector_state()[i], dtype=np.float64) for i, agent in enumerate(self.agents)}
            elif self.mode == "image":
                return self._get_image_state()

    def _get_vector_obs(self):

        """
        Returns
        -------
        vector_obs : list
            vector observation for each agent.
        """

        po_obs = []
        for agent in self.agent:
            obs = []
            idx = 0


            if self.xlen == 3 and self.ylen == 3:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
            elif self.xlen == 5 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 3 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 7 and self.ylen == 7:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap= [[1, 1, 1, 0, 1, 1, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 1, 1, 1, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1]]
            elif self.xlen == 9 and self.ylen == 9:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    
                    agent.pomap= [[1, 1, 1, 1, 0, 1, 1, 1, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                  [1, 0, 0, 1, 1, 1, 0, 0, 1],
                                  [1, 0, 2, 0, 1, 0, 2, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1]]

            for item in self.itemList:
                # Check if the item is within the agent's observation radius or if the radius is 0 (full observability)
                if (agent.x - self.obs_radius <= item.x <= agent.x + self.obs_radius and
                    agent.y - self.obs_radius <= item.y <= agent.y + self.obs_radius) or self.obs_radius == 0:
                    # Normalize item position and add to observation
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status to observation
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)
                        idx += 1
                else:
                    # If the item is outside the observation radius, use its initial position
                    x = agent.obs[idx] * self.xlen
                    y = agent.obs[idx + 1] * self.ylen
                    if not (agent.x - self.obs_radius <= x <= agent.x + self.obs_radius and
                            agent.y - self.obs_radius <= y <= agent.y + self.obs_radius):
                        x = item.initial_x
                        y = item.initial_y
                    x /= self.xlen
                    y /= self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status from the agent's previous observation
                    if isinstance(item, Food):
                        obs.append(agent.obs[idx] / item.required_chopped_times)
                        idx += 1
                # Update the agent's partial observability map
                agent.pomap[int(x * self.xlen)][int(y * self.ylen)] = ITEMIDX[item.rawName]
            # Mark the agent's position on the partial observability map
            agent.pomap[agent.x][agent.y] = ITEMIDX["agent"]
            # Add one-hot encoded task information to the observation
            obs.extend(self.oneHotTask)
            agent.obs = obs
            po_obs.append(np.array(obs))
        return po_obs

    def _get_image_obs(self):

        """
        Returns
        -------
        image_obs : list
            image observation for each agent.
        """

        po_obs = []
        frame = self.game.get_image_obs()
        old_image_width, old_image_height, channels = frame.shape
        new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
        new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
        color = (0,0,0)
        obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame

        for idx, agent in enumerate(self.agent):
            agent_obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)
            po_obs.append(agent_obs)
        return po_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def _findItem(self, x, y, itemName):
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    @property
    def state_size(self):
        return self.get_state().shape[0]

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self, *, seed=None, options=None):
        """
        Returns
        -------
        obs : list
            observation for each agent.
        """

        self.map = copy.deepcopy(self.initMap)
        self._createItems()
        self.step_count = 0

        # Reset task completion status
        counter = Counter(self.task)
        self.taskCompletionStatus = [counter[element] if element in counter else 0 for element in TASKLIST]

        self._initObs()

        return self._get_obs(), {}
    
    def step(self, action):

        """
        Parameters
        ----------
        action: list
            action for each agent

        Returns
        -------
        obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        # print('step了么')

        action = [action['human'], action['ai']] # some ugly hack to make the environment work with rllib.

        # 每调用一次step方法，计数加一
        self.step_count += 1


        # 执行任意一个action，都要花费一个step，都要先penalty一下
        self.reward = self.rewardList["step penalty"]
        done = False


        # 如果步骤计数达到24，标记done为True并重置计数器

        info = {}
        info['cur_mac'] = action
        info['mac_done'] = [True] * self.n_agent
        info['collision'] = []

        all_action_done = False

        for agent in self.agent:

            agent.moved = False

        if self.debug:
            print("in overcooked primitive actions:", action)
            pass

        while not all_action_done:
            for idx, agent in enumerate(self.agent):
                agent_action = action[idx]
                if agent.moved:
                    continue
                agent.moved = True

                if agent_action < 4:
                    target_x = agent.x + DIRECTION[agent_action][0]
                    target_y = agent.y + DIRECTION[agent_action][1]
                    target_name = ITEMNAME[self.map[target_x][target_y]]

                    if target_name == "agent":
                        target_agent = self._findItem(target_x, target_y, target_name)
                        if not target_agent.moved:
                            agent.moved = False
                            target_agent_action = action[AGENTCOLOR.index(target_agent.color)]
                            if target_agent_action < 4:
                                new_target_agent_x = target_agent.x + DIRECTION[target_agent_action][0]
                                new_target_agent_y = target_agent.y + DIRECTION[target_agent_action][1]
                                if new_target_agent_x == agent.x and new_target_agent_y == agent.y:
                                    target_agent.move(new_target_agent_x, new_target_agent_y)
                                    agent.move(target_x, target_y)
                                    agent.moved = True
                                    target_agent.moved = True

                    # 如果是space，那就直接移动
                    elif  target_name == "space":
                        self.map[agent.x][agent.y] = ITEMIDX["space"]
                        agent.move(target_x, target_y)
                        self.map[target_x][target_y] = ITEMIDX["agent"]

                    #pickup and chop
                    # 如果agent没有持有任何东西
                    elif not agent.holding:
                        # 如果是这四类可以移动的item
                        if target_name == "tomato" or target_name == "lettuce" or target_name == "plate" or target_name == "onion":
                            item = self._findItem(target_x, target_y, target_name)
                            agent.pickup(item)
                            # 因为取走了这些可移动的item了，所以把地图中对应的位置变成counter
                            self.map[target_x][target_y] = ITEMIDX["counter"]
                            
                            # reward += self.rewardList["metatask finished"]
                        

                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            # 如果切菜板上面有盘子，则agent会取走盘子
                            if isinstance(knife.holding, Plate):
                                item = knife.holding
                                knife.release()
                                agent.pickup(item)
                                # reward += self.rewardList["metatask finished"]
                            # 如果切菜板上面是食物，则判断是否已经切好
                            elif isinstance(knife.holding, Food):
                                # 如果已经切好了，则取走
                                if knife.holding.chopped:
                                    item = knife.holding
                                    knife.release()
                                    agent.pickup(item)
                                    # reward += self.rewardList["goodtask finished"]
                                # 如果还没有切好，则切一次，判断是否切好，如果切好，判断所切的item是否属于当前task中的，是的话则赋予10的奖励
                                # ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
                                else:
                                    knife.holding.chop()
                                    self.reward += self.rewardList["goodtask finished"]
                                    if knife.holding.chopped:
                                        # if knife.holding.rawName in self.task:
                                        for task in self.task:
                                            if knife.holding.rawName in task:
                                                # 不鼓励切菜和取走，只鼓励装盘
                                                self.reward += self.rewardList["minitask finished"]
                    #put down
                    # 如果agent当前已经持有东西
                    elif agent.holding:
                        # 如果移动的目标是counter，则会放下手中的东西
                        if target_name == "counter":
                            if agent.holding.rawName in ["tomato", "lettuce", "onion", "plate"]:
                                # 把该counter变成agent手中持有的可移动item，这个rawName应该是一些数字
                                self.map[target_x][target_y] = ITEMIDX[agent.holding.rawName]
                            # 恢复非持物状态
                            agent.putdown(target_x, target_y)

                            self.reward += self.rewardList["metatask failed"]
                        # 如果移动目标是盘子
                        elif target_name == "plate":
                            # 如果手中拿的是食物，判断是否切好，未切好不能装盘
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    # 给装盘一个较大的奖励
                                    self.reward += self.rewardList["subtask finished"]
                                    plate = self._findItem(target_x, target_y, target_name)
                                    item = agent.holding
                                    # 放下手中的物品，恢复未持物状态
                                    agent.putdown(target_x, target_y)
                                    # 把食物装进盘子里
                                    plate.contain(item)
                            else:
                                self.reward += self.rewardList["metatask failed"]
                        # 如果移动目标是切菜板
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            # 如果切菜板是空的，则把agent手中的东西放置下来
                            if not knife.holding:
                                item = agent.holding
                                agent.putdown(target_x, target_y)
                                knife.hold(item)
                                if isinstance(item, Food):
                                    if item.chopped:
                                        # 把切好的菜放回去要减分
                                        self.reward += self.rewardList["metatask failed"]
                                    else:
                                        # 只有把没切好的食物放在切菜板上才加分
                                        self.reward += self.rewardList["goodtask finished"]
                                else:
                                    self.reward += self.rewardList["metatask failed"]
                            # 如果切菜板上是食物，agent手中拿的是盘子，则判断食物是否切好了，如果切好了则把食物装进盘子中
                            elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
                                item = knife.holding
                                if item.chopped:
                                    self.reward += self.rewardList["subtask finished"]
                                    knife.release()
                                    agent.holding.contain(item)
                                else:
                                    # 没切好就拿盘子装，减分
                                    self.reward += self.rewardList["metatask failed"]
                            elif isinstance(knife.holding, Food) and not isinstance(agent.holding, Plate):
                                # 切菜板上是食物，但是agent拿的不是盘子，是食物，减分
                                self.reward += self.rewardList["metatask failed"]
                            # 如果切菜板上的是盘子，agent手中拿的是食物，则判断食物是否切好了，如果切好了，把食物放在盘子中，要注意此时agent需要先拿起盘子，再让盘子装起食物
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
                                plate_item = knife.holding
                                food_item = agent.holding
                                if food_item.chopped:
                                    self.reward += self.rewardList["subtask finished"]
                                    knife.release()
                                    # a little different
                                    agent.pickup(plate_item)
                                    agent.holding.contain(food_item)
                                else:
                                    # 切菜板上是盘子，agent拿着没切好的食物，减分
                                    self.reward += self.rewardList["metatask failed"]
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Plate):
                                # 切菜板上是盘子，agent拿着盘子
                                self.reward += self.rewardList["metatask failed"]
                        # 如果移动目标是配送站
                        elif target_name == "delivery":
                            # 如果agent手中拿的是盘子，起码大方向对了，需要继续判断
                            if isinstance(agent.holding, Plate):
                                # 如果盘子里装着food了
                                if agent.holding.containing:
                                    dishName = ""
                                    # 所有菜品名称，哪怕是组合菜，也都是按照lettuce，onion，tomato的顺序命名的，实际的组合顺序任意，但是菜名的顺序是这样的
                                    foodList = [Lettuce, Onion, Tomato]

                                    # 先把foodInPlate变成[-1, -1, -1]
                                    foodInPlate = [-1] * len(foodList)
                                    # 遍历盘子中的东西，比如lettuce，tomato
                                    for f in range(len(agent.holding.containing)):
                                        for i in range(len(foodList)):
                                            # 如果是食物列表中的
                                            if isinstance(agent.holding.containing[f], foodList[i]):
                                                # 把foodInPlate对应的位置从-1变成盘子中物品的index，说明盘子中的这个物品是任务列表中所需要的item的其中之一
                                                foodInPlate[i] = f
                                    # 再次遍历
                                    for i in range(len(foodList)):
                                        # 如果盘子中的物品属于所需要的物品之一，则拼接到菜品名称中，用-分开
                                        if foodInPlate[i] > -1:
                                            dishName += agent.holding.containing[foodInPlate[i]].rawName + "-"
                                    # 最后加上salad后缀
                                    dishName = dishName[:-1] + " salad"
                                    # if dishName == self.task:
                                    if dishName in self.task:
                                        item = agent.holding
                                        # agent放下手中的东西
                                        agent.putdown(target_x, target_y)

                                        # 让上菜处hold一下agent手中的菜，其实就是上菜成功的意思
                                        # 如果注释掉，就是去掉上菜的视觉效果
                                        # self.delivery[0].hold(item)





                                        """下面的代码也是让蔬菜进行刷新"""
                                        food = item.containing
                                        # 盘子release，刷新
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        # 所有盘子中的food刷新，我发现有个核心代码模式就是，self.map[x][y] = ITEMIDX[name]，应该是改变或重置地图中某个位置的意思
                                        for f in food:
                                            f.refresh()
                                            self.map[f.x][f.y] = ITEMIDX[f.rawName]

                                        # print(dishName)
                                        # print(self.taskCompletionStatus)
                                        """如果是多recipe的，不能只是完成一个recipe就done了"""
                                        index = TASKLIST.index(dishName)
                                        if self.taskCompletionStatus[index] > 0:  # 确保不会减成负值
                                            self.taskCompletionStatus[index] -= 1
                                            self.reward += self.rewardList["correct delivery"]
                                            # print(self.taskCompletionStatus[index])
                                            # print('Done one task')
                                        else:
                                            self.reward += self.rewardList["wrong delivery"]
                                            # print('overdone')

                                        if all(value == 0 for value in self.taskCompletionStatus):
                                            self.reward += self.rewardList["correct delivery"]
                                            # self.reward += self.rewardList["correct delivery"]
                                            # self.reward += self.rewardList["correct delivery"]
                                            # print('Completed!')
                                            done = True
                                        # print(self.taskCompletionStatus)
                                    else:
                                        self.reward += self.rewardList["wrong delivery"]
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        # 盘子release，刷新
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        # 所有盘子中的food刷新，我发现有个核心代码模式就是，self.map[x][y] = ITEMIDX[name]，应该是改变或重置地图中某个位置的意思
                                        for f in food:
                                            f.refresh()
                                            self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                # 如果盘子里是空的，那是一种wrong delivery
                                else:
                                    self.reward += self.rewardList["wrong delivery"]
                                    plate = agent.holding
                                    # agent放下手中的东西，手头变空
                                    agent.putdown(target_x, target_y)
                                    # 下面两行是刷新盘子
                                    plate.refresh()
                                    self.map[plate.x][plate.y] = ITEMIDX[plate.name]
                            # 如果把food直接deliver了，则是一种wrong delivery，此时会扣分，同时刷新（1）agent手中东西放下变空，（2）food刷新位置，
                            else:
                                self.reward += self.rewardList["wrong delivery"]
                                food = agent.holding
                                # agent放下手中的东西，手头变空
                                agent.putdown(target_x, target_y)
                                # 下面两行是刷新food
                                food.refresh()
                                self.map[food.x][food.y] = ITEMIDX[food.rawName]

                        # 如果移动目标是食物，则只有（1）agent手中拿着盘子，（2）食物已经切好了，才能执行put down的操作。pickup当然没问题，但是put down只有满足这个条件才能进行
                        elif target_name in ["tomato", "lettuce", "onion"]:
                            item = self._findItem(target_x, target_y, target_name)
                            if item.chopped and isinstance(agent.holding, Plate):
                                self.reward += self.rewardList["subtask finished"]
                                agent.holding.contain(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]
                            elif not item.chopped and isinstance(agent.holding, Plate):
                                # 如果食物没切好就想去装盘，减分
                                self.reward += self.rewardList["metatask failed"]
                            elif isinstance(agent.holding, Food):
                                # 面向食物，agent手里拿的也是食物，减分
                                self.reward += self.rewardList["metatask failed"]

            # for idx, agent in enumerate(self.agent)循环结束，所有agent的action都执行完了
            all_action_done = True

            # 只要还有agent没move，all_action_done就不能为True
            for agent in self.agent:
                if agent.moved == False:
                    all_action_done = False

        terminateds = {"__all__": done or self.step_count >= 80}
        rewards = {agent: self.reward for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        truncated =  False

        return self._get_obs(), rewards, terminateds, {'__all__': truncated}, infos

    # render到界面中
    def render(self, mode='human'):
        return self.game.on_render()

    





