# Example sensors for webots
class WebotsSensors:
    def __init__(self, robot):
        self.robot = robot
        self.gps = robot.getDevice("gps")
        self.gps.enable(32)

    def read(self):
        return {
            "robot_pose": self.gps.getValues(),
            # add detected person pose if available
        }
