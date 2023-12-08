import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("Pybullet/box.urdf", [0, 0, 1])

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

while (1):
    p.stepSimulation()
    time.sleep(1. / 240.)

