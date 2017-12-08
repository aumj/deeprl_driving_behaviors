import os
import sys
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_path, 'gym-bstriner'))
sys.path.insert(0, os.path.join(repo_path, 'gym-traffic'))


import gym
import gym_traffic
from gym.wrappers import Monitor
import random
import time
from tqdm import tqdm

gui = True
monitor = False
num_episodes_total = 500
num_episodes_env = 3

envs_gui = ['Traffic-Simple-gui-v0', 'Traffic-Cross2-gui-v0', 'Traffic-Cross4Lane-gui-v0', 'Traffic-TrafficEnvMulti-gui-v0']
envs_cli = ['Traffic-Simple-cli-v0', 'Traffic-Cross2-cli-v0', 'Traffic-Cross4Lane-cli-v0', 'Traffic-TrafficEnvMulti-cli-v0']

if gui:
    env = gym.make(random.choice(envs_gui))
    # env = gym.make(envs_gui[3])
else:
    env = gym.make(random.choice(envs_cli))
    
if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)

for i_episode in tqdm(range(num_episodes_total)):
    # Restart SUMO with a random environment every few episodes
    if i_episode > 0 and i_episode % num_episodes_env == 0:
        env.unwrapped._stop_sumo()
        if gui:
            env = gym.make(random.choice(envs_gui))
            # env = gym.make(envs_gui[3])
        else:
            env = gym.make(random.choice(envs_cli))
    observation = env.reset()
    total_reward = 0
    for t in tqdm(range(500)):
        # env.render()
        # print(observation)
        # print "\n Observation: {}".format(observation)
        action = env.action_space.sample()
        # print "\n Action: {}".format(action)
        # time.sleep(1)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # print (observation)
        # print "---------------- Observations ----------------"
        # print observation
        # print "\n Reward: {}".format(reward)
        # print "------------------------------------------------"
        if done:
            print("Episode", i_episode, " finished after {} timesteps".format(t+1), " with reward =", total_reward)
            break
