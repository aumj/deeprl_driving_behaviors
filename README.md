A Deep Reinforcement Learning Approach to Behavioral Planning for Autonomous Vehicles
=====================================================================================

Behavioral planning for autonomous vehicles in scenarios such as intersections and lane-changing/merging is a challenging task. The code in this repository is for a couple of the experiments performed for [this paper](https://www.overleaf.com/read/djkbmsqqwfgf).

Installation
------------

The code of this repo has been tested on Ubuntu 14.04 and with Python 2.7

1. [Install SUMO 0.30](http://sumo.dlr.de/wiki/Installing)

   Execute the following:

   `sudo apt-get install sumo sumo-tools sumo-doc`

   `sudo add-apt-repository ppa:sumo/stable`

   `sudo apt-get update`

   `sudo apt-get install sumo sumo-tools sumo-doc`

   Please make sure these instructions are followed exactly.

2. Additional Python Packages install via pip:

   `pip install moviepy imageio tqdm tensorflow==1.4 requests scipy matplotlib ipython`

3. Include `export SUMO_HOME=/usr/share/sumo` in your `~/.bashrc` and `source ~/.bashrc`

Training
--------

To train the DRQN:
Add a folder called Center in the gym-traffic folder. In the Center folder, create a subdirectory called frames. This is where the learning result frames are stored.
Create a folder in the examples folder called drqn. This is where the weights checkpoint files are stored. The checkpoint files are created every 100 episodes, so you must train at least 100 in order to run the DRQN.

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_drqn.py train`

Experiments
-----------
For each of these experiments, you will probably want to set the delay to 0ms once the GUI opens. This will decrease the amount of time you spend waiting for the ego-vehicle to reach the intersection. The simulation should slow down when the ego-vehicle arrives, but you may still need to manage the speed manually via the delay setting.

To test a random routine:

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_random.py`


To test a TTC (Time to Collision) rule based approach:

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_ttc.py`


To test the DRQN (Deep Recurrent Q-Network) based approach (you must have trained at least 100 episodes before running this):

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_drqn.py test`
