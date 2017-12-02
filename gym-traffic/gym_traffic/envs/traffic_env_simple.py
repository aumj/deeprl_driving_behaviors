from traffic_env import TrafficEnv
from traffic_lights import TrafficLightTwoWay
import os


class TrafficEnvSimple(TrafficEnv):
    def __init__(self, mode="gui"):
        lights = [TrafficLightTwoWay(id="0", yield_time=5)]
        loops = ["loop{}".format(i) for i in range(12)]
        lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0", "0_n_0", "0_s_0", "0_e_0", "0_w_0"]
        basepath = os.path.join(os.path.dirname(__file__), "config", "cross")
        netfile = os.path.join(basepath, "cross.net.xml")
        routefile = os.path.join(basepath, "cross.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "cross.add.xml")
        exitloops = ["loop4", "loop5", "loop6", "loop7"]
        super(TrafficEnvSimple, self).__init__(mode=mode, lights=lights, netfile=netfile, routefile=routefile,
                                               guifile=guifile, loops=loops, addfile=addfile, simulation_end=3000,
                                               lanes=lanes, exitloops=exitloops)

    def route_sample(self):
        # if self.np_random.uniform(0, 1) > 0.5:
        ew = 0.25
        we = 0.15
        ns = 0.10
        sn = 0.00001
        wn = 0.05
        en = 0.05

        return {"ns": ns,
                "sn": sn,
                "ew": ew,
                "we": we,
                "wn": wn,
                "en": en,
                }
