from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
# from scipy.misc import imsave
import matplotlib.pyplot as plt
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lights, netfile, routefile, guifile, addfile, loops=[], lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml",
                 pngfile="tmp.png", mode="gui", detector="detector0", step_length = "0.1",  simulation_end=3600, sleep_between_restart=1):
        # "--end", str(simulation_end),
        self.simulation_end = simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.mode = mode
        self._seed()
        self.loops = loops
        self.exitloops = exitloops
        self.loop_variables = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_TIME_SINCE_DETECTION, tc.LAST_STEP_VEHICLE_NUMBER]
        self.lanes = lanes
        self.detector = detector
        args = ["--net-file", netfile, "--route-files", tmpfile, "--additional-files", addfile, "--step-length", step_length]
        if mode == "gui":
            binary = "sumo-gui"
            args += ["-S", "-Q", "--gui-settings-file", guifile]
        else:
            binary = "sumo"
            args += ["--no-step-log"]

        with open(routefile) as f:
            self.route = f.read()
        self.tmpfile = tmpfile
        self.pngfile = pngfile
        self.sumo_cmd = [binary] + args
        self.sumo_step = 0
        self.lights = lights

        self.action_space = spaces.Discrete(3)
        self.throttle_actions = {0: 0., 1: 1., 2:-1.}

        # TO DO: re-define observation space !!
        # trafficspace = spaces.Box(low=float('-inf'), high=float('inf'),
        #                           shape=(len(self.loops) * len(self.loop_variables),))
        # lightspaces = [spaces.Discrete(len(light.actions)) for light in self.lights]
        # self.observation_space = spaces.Tuple([trafficspace] + lightspaces)

        self.sumo_running = False
        self.viewer = None

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def write_routes(self):
        self.route_info = self.route_sample()
        with open(self.tmpfile, 'w') as f:
            f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_sumo(self):
        if not self.sumo_running:
            self.write_routes()
            traci.start(self.sumo_cmd)
            for loopid in self.loops:
                traci.inductionloop.subscribe(loopid, self.loop_variables)
            self.sumo_step = 0
            self.sumo_deltaT = traci.simulation.getDeltaT()/1000 # Simulation timestep in seconds
            self.sumo_running = True
            traci.vehicle.add(vehID='ego_car', routeID='route_sn', pos = 0, speed = 1, typeID='EgoCar')
            traci.vehicle.setSpeedMode(vehID='ego_car', sm=0) # All speed checks are off
            # import IPython
            # IPython.embed()
            self.screenshot()

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    # TO DO: re-define reward function!!
    def _reward(self):
        reward = -0.1
        # for lane in self.lanes:
        #    reward -= traci.lane.getWaitingTime(lane)
        # return reward
        # speed = traci.multientryexit.getLastStepMeanSpeed(self.detector)
        # count = traci.multientryexit.getLastStepVehicleNumber(self.detector)
        # reward = speed * count
        # print("Speed: {}".format(traci.multientryexit.getLastStepMeanSpeed(self.detector)))
        # print("Count: {}".format(traci.multientryexit.getLastStepVehicleNumber(self.detector)))
        # reward = np.sqrt(speed)
        # print "Reward: {}".format(reward)
        # return speed
        # reward = 0.0
        # for loop in self.exitloops:
        #    reward += traci.inductionloop.getLastStepVehicleNumber(loop)
        return reward

    def _step(self, action):
        # action = self.action_space(action)
        self.start_sumo()
        self.sumo_step += 1

        # print "action = ", self.throttle_actions[action]
        new_speed = traci.vehicle.getSpeed('ego_car') + self.sumo_deltaT * self.throttle_actions[action]
        traci.vehicle.setSpeed('ego_car', new_speed)

        print("Step = ", self.sumo_step, "   | action = ", action, "   | car speed = ", traci.vehicle.getSpeed('ego_car'))

        traci.simulationStep()
        observation = self._observation()
        reward = self._reward()
        done = self.sumo_step > self.simulation_end or ('ego_car' not in traci.vehicle.getIDList())
        self.screenshot()
        if done:
            self.stop_sumo()
        return observation, reward, done, self.route_info

    def screenshot(self):
        if self.mode == "gui":
            traci.gui.screenshot("View #0", self.pngfile)

    def _observation(self):
        state = []
        visible = []
        ego_car_in_scene=False
        if 'ego_car' in traci.vehicle.getIDList():
            ego_car_pos = traci.vehicle.getPosition('ego_car')
            ego_car_in_scene = True
        for i in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(i)
            pos = traci.vehicle.getPosition(i)
            angle = traci.vehicle.getAngle(i)
            laneid = traci.vehicle.getRouteID(i)
            state_tuple = (i,pos[0], pos[1], angle, speed, laneid)
            state.append(state_tuple)
            if ego_car_in_scene:
                if(np.linalg.norm(np.asarray(pos)-np.asarray(ego_car_pos))<50) and i not in 'ego_car':
                    visible.append(state_tuple)

        def location2bounds(x, y, orientation):
            bound = 101
            car_length = 5 # meters
            car_width = 1.8 # meters
            # continuous bounds
            car_c_bound_x_1 = 0
            car_c_bound_x_2 = 0
            car_c_bound_y_1 = 0
            car_c_bound_y_2 = 0
            if orientation == 'vertical':
                car_c_bound_x_1 = x-(car_width/2.0)
                car_c_bound_x_2 = x+(car_width/2.0)
                car_c_bound_y_1 = y-(car_length/2.0)
                car_c_bound_y_2 = y+(car_length/2.0)
            elif orientation == 'horizontal':
                car_c_bound_x_1 = x-(car_length/2.0)
                car_c_bound_x_2 = x+(car_length/2.0)
                car_c_bound_y_1 = y-(car_width/2.0)
                car_c_bound_y_2 = y+(car_width/2.0)

            # discrete bounds
            car_d_bound_x_1 = np.floor(car_c_bound_x_1)+np.floor(bound/2.0)
            car_d_bound_x_2 = np.floor(car_c_bound_x_2)+np.floor(bound/2.0)
            car_d_bound_y_1 = np.floor(car_c_bound_y_1)+np.floor(bound/2.0)
            car_d_bound_y_2 = np.floor(car_c_bound_y_2)+np.floor(bound/2.0)

            if (car_d_bound_x_1 < 0):
                car_d_bound_x_1 = 0
            if (car_d_bound_x_2 < 0):
                car_d_bound_x_2 = 0
            if (car_d_bound_y_1 < 0):
                car_d_bound_y_1 = 0
            if (car_d_bound_y_2 < 0):
                car_d_bound_y_2 = 0
            if (car_d_bound_x_1 >= bound):
                car_d_bound_x_1 = bound-1
            if (car_d_bound_x_2 >= bound):
                car_d_bound_x_2 = bound-1
            if (car_d_bound_y_1 >= bound):
                car_d_bound_y_1 = bound-1
            if (car_d_bound_y_2 >= bound):
                car_d_bound_y_2 = bound-1

            return (car_d_bound_x_1, car_d_bound_x_2, car_d_bound_y_1, car_d_bound_y_2)

        if ego_car_in_scene:
            bound = 101
            obstacle_image = np.zeros((bound,bound)) # 1 meter descretization image

            # insert ego car
            car_bounds = location2bounds(0.0, 0.0, 'vertical')
            for x in range(int(car_bounds[0]), int(car_bounds[1]+1)):
                for y in range(int(car_bounds[2]), int(car_bounds[3]+1)):
                    obstacle_image[100-y,x] = 1

            for other_car in visible:
                #if vertical
                if (other_car[5] == 'route_ns') or (other_car[5] == 'route_sn'):
                    car_bounds = location2bounds(other_car[1]-ego_car_pos[0], other_car[2]-ego_car_pos[1], 'vertical')
                    for x in range(int(car_bounds[0]), int(car_bounds[1]+1)):
                        for y in range(int(car_bounds[2]), int(car_bounds[3]+1)):
                            obstacle_image[100-y,x] = 1
                #if horizontal
                if (other_car[5] == 'route_ew') or (other_car[5] == 'route_we'):
                    car_bounds = location2bounds(other_car[1]-ego_car_pos[0], other_car[2]-ego_car_pos[1], 'horizontal')
                    for x in range(int(car_bounds[0]), int(car_bounds[1]+1)):
                        for y in range(int(car_bounds[2]), int(car_bounds[3]+1)):
                            obstacle_image[100-y,x] = 1

            plt.imsave('test.jpg', obstacle_image)
            # import IPython
            # IPython.embed()
            plt.ion()
            plt.imshow(obstacle_image)
            plt.draw()
            # plt.show(block=False)
            plt.show()
            return (state,visible,obstacle_image)

        return (state,visible)

    def _reset(self):
        self.stop_sumo()
        # sleep required on some systems
        if self.sleep_between_restart > 0:
            time.sleep(self.sleep_between_restart)
        self.start_sumo()
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.mode == "gui":
            img = imread(self.pngfile, mode="RGB")
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            raise NotImplementedError("Only rendering in GUI mode is supported. Please use Traffic-...-gui-v0.")
