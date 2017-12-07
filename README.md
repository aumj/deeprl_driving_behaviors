A Deep Reinforcement Learning Approach to Behavioral Planning for Autonomous Vehicles
=====================================================================================

Behavioral planning for autonomous vehicles in scenarios such as intersections and lane-changing/merging is a challenging task. The code in this repository is for a couple of the experiments performed for [this paper](https://www.overleaf.com/read/djkbmsqqwfgf).

Installation
------------

The code of this repo has been tested on Ubuntu 14.04 and with Python 2.7

1. [Install SUMO 0.30](http://sumo.dlr.de/wiki/Installing)
   Execute the following:

  1. `sudo apt-get install sumo sumo-tools sumo-doc`
  2. `sudo add-apt-repository ppa:sumo/stable`
  3. `sudo apt-get update`
  4. `sudo apt-get install sumo sumo-tools sumo-doc`

   Please make sure these instructions are followed exactly.

2. Additional Python Packages install via `pip install [package-name]`:
  1. `moviepy`
  2. `imageio`
  3. `tqdm`
  4. `tensorflow==1.4`

3. Include `export SUMO_HOME=/usr/share/sumo` in your `~/.bashrc` and `source ~/.bashrc`

Training
--------

To train the DRQM:

`cd [repo_root]/gym-traffic/examples`

`python example_gym_traffic_drqn.py train`

Experiments
-----------

TODO: STILL NEED TO SHOW FILE PATH DEPENDENCIES HERE

To test a random routine:
`cd [repo_root]/gym-traffic/examples`
`python example_gym_traffic_random.py`

To test a TTC (Time to Collision) rule based approach
`cd [repo_root]/gym-traffic/examples`
`python example_gym_traffic_ttc.py`

To test the DRQN (Deep Recurrent Q-Netowork based approach
`cd [repo_root]/gym-traffic/examples`
`python example_gym_traffic_drqn.py test`