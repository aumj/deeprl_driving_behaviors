import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from gym_traffic.envs.traffic_env import TrafficEnv
from gym_traffic.envs.traffic_env_simple import TrafficEnvSimple
