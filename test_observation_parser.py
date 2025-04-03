import numpy as np
import unittest
from environment.observation_parser import parse_observation, reconstruct_environment_state
from environment.Overcooked import Overcooked_multi, TASKLIST

class TestObservationParser(unittest.TestCase):
    
    def setUp(self):
        grid_dim = (3, 3)
        task = ["lettuce-tomato salad"]  
        reward_list = {
            "step penalty": -0.1,
            "subtask finished": 1.0,
            "correct delivery": 10.0,
            "wrong delivery": -5.0,
            "metatask failed": -1.0
        }
        self.env = Overcooked_multi(grid_dim, task, reward_list, map_type="A", debug=False)
        
    def test_parse_observation_structure(self):
        """Test that the parser returns the expected structure."""
        obs, _ = self.env.reset()
        agent_obs = obs['human']  
        
        item_counts = {
            'agent': len(self.env.agent),
            'tomato': len(self.env.tomato),
            'lettuce': len(self.env.lettuce),
            'onion': len(self.env.onion) if hasattr(self.env, 'onion') else 0,
            'plate': len(self.env.plate),
            'knife': len(self.env.knife),
            'delivery': len(self.env.delivery)
        }
        
        parsed = parse_observation(agent_obs, (self.env.xlen, self.env.ylen), item_counts)
        
        expected_keys = ['agents', 'foods', 'plates', 'knives', 'deliveries', 'tasks']
        for key in expected_keys:
            self.assertIn(key, parsed, f"Missing key: {key}")
        
        self.assertEqual(parsed['tasks'], ["lettuce-tomato salad"])
        
        self.assertEqual(len(parsed['agents']), item_counts['agent'])
        self.assertEqual(len(parsed['plates']), item_counts['plate'])
        self.assertEqual(len(parsed['knives']), item_counts['knife'])
        self.assertEqual(len(parsed['deliveries']), item_counts['delivery'])
        
        expected_food_count = item_counts['tomato'] + item_counts['lettuce'] + item_counts['onion']
        self.assertEqual(len(parsed['foods']), expected_food_count)
    
    def test_normalized_observation_structure(self):
        """Test that the observation structure is correct with normalized positions."""
        obs, _ = self.env.reset()
        agent_obs = obs['human']
        
        item_counts = {
            'agent': len(self.env.agent),
            'tomato': len(self.env.tomato),
            'lettuce': len(self.env.lettuce),
            'onion': len(self.env.onion) if hasattr(self.env, 'onion') else 0,
            'plate': len(self.env.plate),
            'knife': len(self.env.knife),
            'delivery': len(self.env.delivery)
        }
        
        parsed = parse_observation(agent_obs, (self.env.xlen, self.env.ylen), item_counts)
        
        self.assertEqual(len(parsed['agents']), item_counts['agent'])
        self.assertEqual(len(parsed['knives']), item_counts['knife'])
        self.assertEqual(len(parsed['deliveries']), item_counts['delivery'])
        
        for agent in parsed['agents']:
            self.assertGreaterEqual(agent.x, 0)
            self.assertLess(agent.x, self.env.xlen)
            self.assertGreaterEqual(agent.y, 0)
            self.assertLess(agent.y, self.env.ylen)
            
        for knife in parsed['knives']:
            self.assertGreaterEqual(knife.x, 0)
            self.assertLess(knife.x, self.env.xlen)
            self.assertGreaterEqual(knife.y, 0)
            self.assertLess(knife.y, self.env.ylen)
            
        for delivery in parsed['deliveries']:
            self.assertGreaterEqual(delivery.x, 0)
            self.assertLess(delivery.x, self.env.xlen)
            self.assertGreaterEqual(delivery.y, 0)
            self.assertLess(delivery.y, self.env.ylen)
    
    def test_chopped_status(self):
        """Test that chopped status is correctly parsed."""
        obs, _ = self.env.reset()
        agent_obs = obs['human']
        
        item_counts = {
            'agent': len(self.env.agent),
            'tomato': len(self.env.tomato),
            'lettuce': len(self.env.lettuce),
            'onion': len(self.env.onion) if hasattr(self.env, 'onion') else 0,
            'plate': len(self.env.plate),
            'knife': len(self.env.knife),
            'delivery': len(self.env.delivery)
        }
        
        parsed = parse_observation(agent_obs, (self.env.xlen, self.env.ylen), item_counts)
        
        for food in parsed['foods']:
            if food.type == 'tomato' and self.env.tomato:
                env_food = next((t for t in self.env.tomato 
                              if abs(t.x - food.x) < 0.5 and abs(t.y - food.y) < 0.5), None)
            elif food.type == 'lettuce' and self.env.lettuce:
                env_food = next((l for l in self.env.lettuce 
                               if abs(l.x - food.x) < 0.5 and abs(l.y - food.y) < 0.5), None)
            elif food.type == 'onion' and hasattr(self.env, 'onion') and self.env.onion:
                env_food = next((o for o in self.env.onion 
                               if abs(o.x - food.x) < 0.5 and abs(o.y - food.y) < 0.5), None)
            else:
                continue
                
            if env_food:
                expected_progress = env_food.cur_chopped_times / env_food.required_chopped_times
                self.assertAlmostEqual(food.chopped_progress, expected_progress, delta=0.1)
    
    def test_reconstruct_environment(self):
        """Test the higher-level reconstruction function."""
        obs, _ = self.env.reset()
        agent_obs = obs['human']
        
        item_counts = {
            'agent': len(self.env.agent),
            'tomato': len(self.env.tomato),
            'lettuce': len(self.env.lettuce),
            'onion': len(self.env.onion) if hasattr(self.env, 'onion') else 0,
            'plate': len(self.env.plate),
            'knife': len(self.env.knife),
            'delivery': len(self.env.delivery)
        }
        
        reconstructed = reconstruct_environment_state(agent_obs, (self.env.xlen, self.env.ylen), item_counts)
        
        expected_keys = ['agents', 'foods', 'plates', 'knives', 'deliveries', 'tasks']
        for key in expected_keys:
            self.assertIn(key, reconstructed)
        

    def test_with_modified_environment(self):
        obs, _ = self.env.reset()
        
        action = {'human': 4, 'ai': 4}  
        try:
            new_obs, _, _, _, _ = self.env.step(action)
            agent_obs = new_obs['human']
        except Exception:
            agent_obs = obs['human']
        
        item_counts = {
            'agent': len(self.env.agent),
            'tomato': len(self.env.tomato),
            'lettuce': len(self.env.lettuce),
            'onion': len(self.env.onion) if hasattr(self.env, 'onion') else 0,
            'plate': len(self.env.plate),
            'knife': len(self.env.knife),
            'delivery': len(self.env.delivery)
        }
        
        parsed = parse_observation(agent_obs, (self.env.xlen, self.env.ylen), item_counts)
        
        self.assertEqual(parsed['tasks'], ["lettuce-tomato salad"])
        
        self.assertEqual(len(parsed['agents']), item_counts['agent'])

    def test_default_play_parameters(self):
        """Test observation parser with default parameters from play.py."""
        grid_dim = [5, 5]
        task_idx = 6  
        map_type = "A"
        mode = "vector"
        
        reward_list = {
            "subtask finished": 10,
            "metatask failed": -5,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1
        }
        
        env = Overcooked_multi(
            grid_dim=grid_dim,
            task=[TASKLIST[task_idx]],
            rewardList=reward_list,
            map_type=map_type,
            mode=mode,
            debug=False
        )
        
        obs, _ = env.reset()
        agent_obs = obs['human']
        
        item_counts = {
            'agent': len(env.agent),
            'tomato': len(env.tomato),
            'lettuce': len(env.lettuce),
            'onion': len(env.onion) if hasattr(env, 'onion') else 0,
            'plate': len(env.plate),
            'knife': len(env.knife),
            'delivery': len(env.delivery)
        }
        
        parsed = parse_observation(agent_obs, (env.xlen, env.ylen), item_counts)
        
        self.assertEqual(parsed['tasks'], ["lettuce-onion-tomato salad"])
        
        self.assertGreater(len(parsed['agents']), 0, "No agents found")
        self.assertGreater(len(parsed['foods']), 0, "No foods found")
        self.assertGreater(len(parsed['plates']), 0, "No plates found")
        self.assertGreater(len(parsed['knives']), 0, "No knives found")
        self.assertGreater(len(parsed['deliveries']), 0, "No delivery points found")
        
        self.assertEqual(len(parsed['agents']), item_counts['agent'])
        
        self._print_grid_layout(parsed, env.xlen, env.ylen)
        
        print("\nItem positions:")
        for item_type, items in parsed.items():
            if item_type != 'tasks':
                positions = [(item.x, item.y) for item in items]
                print(f"  {item_type}: {positions}")
    
    def _print_grid_layout(self, parsed, rows, cols):
        """Print a visual representation of the grid with items."""
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        
        for agent in parsed['agents']:
            x, y = int(agent.x), int(agent.y)
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = 'A'
        
        for food in parsed['foods']:
            x, y = int(food.x), int(food.y)
            if 0 <= x < rows and 0 <= y < cols:
                if food.type == 'tomato':
                    grid[x][y] = 'T'
                elif food.type == 'lettuce':
                    grid[x][y] = 'L'
                elif food.type == 'onion':
                    grid[x][y] = 'O'
        
        for plate in parsed['plates']:
            x, y = int(plate.x), int(plate.y)
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = 'P'
        
        for knife in parsed['knives']:
            x, y = int(knife.x), int(knife.y)
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = 'K'
        
        for delivery in parsed['deliveries']:
            x, y = int(delivery.x), int(delivery.y)
            if 0 <= x < rows and 0 <= y < cols:
                grid[x][y] = 'D'
        
        print(f"\nGrid {rows}x{cols}:")
        for row in grid:
            print(' '.join(row))
        print()

if __name__ == '__main__':
    unittest.main()
