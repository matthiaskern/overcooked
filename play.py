import argparse
from gym_macro_overcooked.Overcooked import Overcooked_multi

TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

def play(env_id, grid_dim, task, map_type, n_agent, obs_radius, mode, debug):

    rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
    env_params = {'grid_dim': grid_dim,
                    'task': TASKLIST[task],
                    'rewardList': rewardList,
                    'map_type': map_type,
                    'n_agent': n_agent,
                    'obs_radius': obs_radius,
                    'mode': mode,
                    'debug': debug
                }
    env = Overcooked_multi(**env_params)
    actionMapping = {
        "w": 3,
        "d": 0,
        "a": 2,
        "s": 1,
        "q": 4
    }
    rewards = 0
    discount = 1
    step = 0
    env.game.on_init()
    obs = env.reset()     
    env.render()

    while(True):
        step += 1
        inputHuman = input("input Human:").split(" ")
        inputAI = input("input AI:").split(" ")
        action = {"human": actionMapping[inputHuman[0]], "ai": actionMapping[inputAI[0]]}
        obs, reward, done, _,  info = env.step(action)
        env.render()
        rewards += discount * reward['human']
        done = done['__all__']
        discount *= 0.99
        if done:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id',                 action='store',        type=str,             default='Overcooked-v1',    help='Domain name')
    parser.add_argument('--n_agent',                action='store',        type=int,             default=2,                     help='Number of agents')
    parser.add_argument('--grid_dim',               action='store',        type=int,  nargs=2,   default=[5,5],                 help='Grid world size')
    parser.add_argument('--task',                   action='store',        type=int,             default=6,                     help='The receipt agent cooks')
    parser.add_argument('--map_type',               action='store',        type=str,             default="A",                   help='The type of map')
    parser.add_argument('--obs_radius',             action='store',        type=int,             default=2,                     help='The radius of the agents')
    parser.add_argument('--mode',                   action='store',        type=str,             default="vector",              help='The type of the observation(vector/image)')    
    parser.add_argument('--debug',                  action='store',        type=bool,            default=True,                  help='Whehter print the debug information and render')                
   
    params = vars(parser.parse_args())
    play(**params)



