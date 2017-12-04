class EgoVehicle(object):
    def __init__(self, vehID, routeID, typeID, start_pos, goal_pos, start_speed):
        self.vehID = vehID
        self.routeID = routeID
        self.typeID = typeID
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_speed = start_speed

    def reached_goal(self, vehicle_pos):
        if self.routeID == 'route_sn':
            if vehicle_pos[1] > self.goal_pos: return True
        elif self.routeID == 'route_se':
            if vehicle_pos[0] > self.goal_pos: return True
        elif self.routeID == 'route_sw':
            if vehicle_pos[0] < self.goal_pos: return True

        return False
