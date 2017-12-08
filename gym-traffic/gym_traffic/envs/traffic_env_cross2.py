from traffic_env import TrafficEnv
from ego_vehicle import EgoVehicle
import os
import numpy as np

class TrafficEnvCross2(TrafficEnv):
    def __init__(self, mode="gui"):
        lanes = ["n_m_0", "s_m_0", "e_m_0", "w_m_0", "m_n_0", "m_s_0", "m_e_0", "m_w_0"]

        # We don't use lights or loops
        lights=[]
        loops = []
        exitloops = []

        basepath = os.path.join(os.path.dirname(__file__), "config", "cross2")
        netfile = os.path.join(basepath, "cross2.net.xml")
        routefile = os.path.join(basepath, "cross2.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "cross2.add.xml")

        ego_vehicles = [EgoVehicle('ego_car', 'route_sn', 'EgoCar', 294.,  11., 0.),
                        EgoVehicle('ego_car', 'route_se', 'EgoCar', 294.,  11., 0.),
                        EgoVehicle('ego_car', 'route_sw', 'EgoCar', 294., -11., 0.)]

        super(TrafficEnvCross2, self).__init__(ego_vehicles=ego_vehicles, mode=mode, lights=lights, netfile=netfile,
                                               routefile=routefile, guifile=guifile, loops=loops, addfile=addfile,
                                               step_length="0.1", simulation_end=3000, lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        ew = np.random.normal(0.12, 0.02)
        we = np.random.normal(0.12, 0.02)
        ew2 = np.random.normal(0.02, 0.02)
        we2 = np.random.normal(0.23, 0.02)
        ns = np.random.normal(0.08, 0.02)
        sn = 0.01

        routes = {"ns": ns, "sn": sn, "ew": ew, "we": we, "ew2": ew2, "we2": we2}

        return routes
