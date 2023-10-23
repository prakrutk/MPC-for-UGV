import pybullet as p
import numpy as np

class Racecar:
    def __init__(self, car_id, wheel_ids, max_force=20.0, max_speed=10.0):
        self.car_id = car_id
        self.wheel_ids = wheel_ids
        self.max_force = max_force
        self.max_speed = max_speed

    def step(self, throttle, steer):
        # Compute the force to apply to the wheels
        force = self.max_force * throttle
        steer_angle = np.radians(30.0) * steer

        # Apply the force to the wheels
        for wheel_id in self.wheel_ids:
            p.setJointMotorControl2(self.car_id, wheel_id, p.TORQUE_CONTROL, force=force)

        # Apply the steering angle to the front wheels
        p.setJointMotorControl2(self.car_id, self.wheel_ids[0], p.POSITION_CONTROL, targetPosition=steer_angle)
        p.setJointMotorControl2(self.car_id, self.wheel_ids[1], p.POSITION_CONTROL, targetPosition=steer_angle)

        # Step the simulation forward by one timestep
        p.stepSimulation()
