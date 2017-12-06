from traffic_env import TrafficEnv
from ego_vehicle import EgoVehicle
import os
import numpy as np


class TrafficEnvCross4Lane(TrafficEnv):
    def __init__(self, mode="gui"):
        lanes = ["n_m_0", "s_m_0", "e_m_0", "w_m_0", "m_n_0", "m_s_0", "m_e_0", "m_w_0"]

        # This env doesn't use lights or loops
        lights=[]
        loops = []
        exitloops = []

        basepath = os.path.join(os.path.dirname(__file__), "config", "left2")
        netfile = os.path.join(basepath, "left2.net.xml")
        routefile = os.path.join(basepath, "left2.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "left2.add.xml")


        ego_vehicles = [EgoVehicle('ego_car', 'route_sn', 'EgoCar', 290., 13., 0.),
                        EgoVehicle('ego_car', 'route_se', 'EgoCar', 290., 13., 0.),
                        EgoVehicle('ego_car', 'route_sw', 'EgoCar', 290., -12.5, 0.)]

        super(TrafficEnvCross4Lane, self).__init__(ego_vehicles=ego_vehicles, mode=mode, lights=lights, netfile=netfile,
                                               routefile=routefile, guifile=guifile, loops=loops, addfile=addfile,
                                               step_length="0.1", simulation_end=3000, lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        ew = np.random.normal(0.10, 0.02)
        we = np.random.normal(0.10, 0.02)
        ew2 = np.random.normal(0.03, 0.02)
        we2 = np.random.normal(0.03, 0.02)
        ns = np.random.normal(0.05, 0.02)
        sn = 0.01

        routes = {"ns": ns, "sn": sn, "ew": ew, "we": we, "ew2": ew2, "we2": we2}

        return routes
