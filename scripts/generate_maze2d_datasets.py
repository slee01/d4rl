import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--fixed-target', action='store_true', help='Fixed target point')
    parser.add_argument('--env-name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num-samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--num-episodes', type=int, default=int(1e3), help='Num episodes to collect')
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    args = parser.parse_args()
    
    data = reset_data()
    
    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)
    
    for episode in range(args.num_episodes):
        s = env.reset()
        act = env.action_space.sample()
        done, episode_step = False, 0
        
        if args.fixed_target:
            env.set_target(np.array([1.0, 1.0]))
        else:
            env.set_target()
        
        while not done:
            position = s[0:2]
            velocity = s[2:4]
            act, _done = controller.get_action(episode_step, position, velocity, env._target)
            if args.noisy:
                act = act + np.random.randn(*act.shape) * 0.5

            act = np.clip(act, -1.0, 1.0)
            if episode_step >= max_episode_steps:
                done = True
                
            append_data(data, s, act, env._target, done, env.sim.data)
    
            ns, _, _, _ = env.step(act)

            if len(data['observations']) % 10000 == 0:
                print(len(data['observations']))

            episode_step += 1
            
            if done:
                break
            else:
                s = ns
    
            if args.render:
                env.render()
    
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        fname = '%s.hdf5' % args.env_name
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
