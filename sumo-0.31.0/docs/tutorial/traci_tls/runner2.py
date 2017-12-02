#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci

def run():
    """execute the TraCI control loop"""
    step = 0
    # we start with phase 2 where EW has green
    traci.trafficlights.setPhase("0", 2)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        if traci.trafficlights.getPhase("0") == 2:
            # we are not already switching
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                # there is a vehicle from the north, switch
                traci.trafficlights.setPhase("0", 3)
            else:
                # otherwise try to keep green for EW
                traci.trafficlights.setPhase("0", 2)
        step += 1
    traci.close()
    sys.stdout.flush()



# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"]

    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start(sumoCmd)
    run()
