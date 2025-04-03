import numpy as np
from collections import namedtuple
from .Overcooked import TASKLIST

ParsedAgent = namedtuple('ParsedAgent', ['x', 'y', 'holding', 'color'])
ParsedFood = namedtuple('ParsedFood', ['x', 'y', 'type', 'chopped_progress'])
ParsedPlate = namedtuple('ParsedPlate', ['x', 'y', 'containing'])
ParsedKnife = namedtuple('ParsedKnife', ['x', 'y', 'holding'])
ParsedDelivery = namedtuple('ParsedDelivery', ['x', 'y'])


def parse_observation(observation, grid_dims, item_counts=None):
    """
    Parse observation vector back into object representations.
    
    Parameters:
    -----------
    observation : np.ndarray
        The observation vector returned by the environment
    grid_dims : tuple
        The dimensions of the grid (xlen, ylen)
    item_counts : dict, optional
        Dictionary containing the count of each item type in the environment
        Example: {'agent': 2, 'tomato': 2, 'lettuce': 1, 'plate': 2, 'knife': 1, 'delivery': 1, 'onion': 1}
        If not provided, it will try to infer from the observation length
        
    Returns:
    --------
    dict:
        A dictionary containing parsed objects grouped by type:
        {
            'agents': list of ParsedAgent objects,
            'foods': list of ParsedFood objects,
            'plates': list of ParsedPlate objects,
            'knives': list of ParsedKnife objects,
            'deliveries': list of ParsedDelivery objects,
            'tasks': list of active tasks
        }
    """
    xlen, ylen = grid_dims
    
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
    
    if item_counts is None:
        obs_len = len(observation)
        task_len = len(TASKLIST)
        
        # Each agent has 2 positions (x, y) + holding status
        # Each food has 2 positions (x, y) + chopped status
        # Each plate has 2 positions (x, y) + containing status
        # Each knife has 2 positions (x, y) + holding status
        # Each delivery has 2 positions (x, y)
        # Tasks are one-hot encoded (7 values)
        
        # TODO make dynamic
        remaining_len = obs_len - task_len
        item_counts = {
            'agent': 2,  # Default to 2 agents
            'tomato': 1,
            'lettuce': 1,
            'onion': 1,
            'plate': 2,
            'knife': 1,
            'delivery': 1
        }
    
    # Initialize result dictionary
    result = {
        'agents': [],
        'foods': [],
        'plates': [],
        'knives': [],
        'deliveries': [],
        'tasks': []
    }
    
    # Calculate how many items we're actually parsing
    # Each agent has position (x, y)
    # Each food has position (x, y) and chopped status
    # Each plate has position (x, y)
    # Each knife has position (x, y)
    # Each delivery has position (x, y)
    # Tasks are one-hot encoded at the end
    
    items_space = 0
    for item_type, count in item_counts.items():
        if item_type == 'agent':
            items_space += count * 2
        elif item_type in ['tomato', 'lettuce', 'onion']:
            items_space += count * 3
        elif item_type == 'plate':
            items_space += count * 2
        elif item_type == 'knife':
            items_space += count * 2
        elif item_type == 'delivery':
            items_space += count * 2
    
    task_start_idx = items_space
    
    if task_start_idx > len(observation):
        print(f"Warning: Calculated task_start_idx {task_start_idx} exceeds observation length {len(observation)}")
        task_start_idx = max(0, len(observation) - len(TASKLIST))
    
    if task_start_idx + len(TASKLIST) <= len(observation):
        task_vector = observation[task_start_idx:task_start_idx + len(TASKLIST)]
        result['tasks'] = [TASKLIST[i] for i, active in enumerate(task_vector) if active > 0.5]
    else:
        result['tasks'] = []
    
    idx = 0
    
    for i in range(item_counts.get('agent', 0)):
        if idx + 1 >= len(observation):
            break
            
        x = int(observation[idx] * xlen)
        y = int(observation[idx+1] * ylen)
        
        result['agents'].append(ParsedAgent(x=x, y=y, holding=None, color=None))
        idx += 2
    
    for food_type in ['tomato', 'lettuce', 'onion']:
        for _ in range(item_counts.get(food_type, 0)):
            if idx + 2 >= len(observation):
                break
                
            x = int(observation[idx] * xlen)
            y = int(observation[idx+1] * ylen)
            chopped_progress = float(observation[idx+2]) if idx + 2 < len(observation) else 0.0
                
            result['foods'].append(ParsedFood(x=x, y=y, type=food_type, chopped_progress=chopped_progress))
            idx += 3
    
    for _ in range(item_counts.get('plate', 0)):
        if idx + 1 >= len(observation):
            break
            
        x = int(observation[idx] * xlen)
        y = int(observation[idx+1] * ylen)
        result['plates'].append(ParsedPlate(x=x, y=y, containing=None))
        idx += 2
    
    for _ in range(item_counts.get('knife', 0)):
        if idx + 1 >= len(observation):
            break
            
        x = int(observation[idx] * xlen)
        y = int(observation[idx+1] * ylen)
        result['knives'].append(ParsedKnife(x=x, y=y, holding=None))
        idx += 2
    
    for _ in range(item_counts.get('delivery', 0)):
        if idx + 1 >= len(observation):
            break
            
        x = int(observation[idx] * xlen)
        y = int(observation[idx+1] * ylen)
        result['deliveries'].append(ParsedDelivery(x=x, y=y))
        idx += 2
    
    
    return result

def reconstruct_environment_state(observation, grid_dims, item_counts=None):
    parsed = parse_observation(observation, grid_dims, item_counts)
    
    for agent_idx, agent in enumerate(parsed['agents']):
        for food in parsed['foods']:
            if abs(agent.x - food.x) < 0.1 and abs(agent.y - food.y) < 0.1:
                parsed['agents'][agent_idx] = agent._replace(holding=food)
                break
                
        for plate in parsed['plates']:
            if abs(agent.x - plate.x) < 0.1 and abs(agent.y - plate.y) < 0.1:
                parsed['agents'][agent_idx] = agent._replace(holding=plate)
                break
    
    for plate_idx, plate in enumerate(parsed['plates']):
        containing = []
        for food in parsed['foods']:
            if abs(plate.x - food.x) < 0.1 and abs(plate.y - food.y) < 0.1:
                containing.append(food)
        
        if containing:
            parsed['plates'][plate_idx] = plate._replace(containing=containing)
    
    for knife_idx, knife in enumerate(parsed['knives']):
        for food in parsed['foods']:
            if abs(knife.x - food.x) < 0.1 and abs(knife.y - food.y) < 0.1:
                parsed['knives'][knife_idx] = knife._replace(holding=food)
                break
                
        for plate in parsed['plates']:
            if abs(knife.x - plate.x) < 0.1 and abs(knife.y - plate.y) < 0.1:
                parsed['knives'][knife_idx] = knife._replace(holding=plate)
                break
    
    return parsed
