# Example driver for webots

class WebotsDriver:
    def __init__(self, robot):
        self.robot = robot
        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")

    def set_base_velocity(self, v_left, v_right):
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)

    def stop(self):
        self.set_base_velocity(0.0, 0.0)

        