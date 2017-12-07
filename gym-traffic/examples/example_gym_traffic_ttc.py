import os
import sys
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo_path, 'gym-bstriner'))
sys.path.insert(0, os.path.join(repo_path, 'gym-traffic'))


import gym
import gym_traffic
from gym.wrappers import Monitor
import time

env = gym.make('Traffic-Simple-gui-v0')
from tqdm import tqdm
monitor = False
# env = gym.make('Traffic-Simple-cli-v0')

#TODO: Change simulation step size
#TODO: Add more traffic flows
#TODO: Scene image generation

if monitor:
    env = Monitor(env, "output/traffic/simple/random", force=True)
for i_episode in tqdm(range(500)):
    observation = env.reset()
    total_reward = 0
    for t in tqdm(range(1000)):
        # env.render()
        # print(observation)
        # print "\n Observation: {}".format(observation)
        env = env.unwrapped
        action = env.action_from_ttc()
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
            print("Episode finished after {} timesteps".format(t+1), " with reward =", total_reward)
            break
