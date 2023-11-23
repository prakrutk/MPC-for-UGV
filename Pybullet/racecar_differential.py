import pybullet as p
import time
import pybullet_data
import math
import cv2
from Waypoint_generation.Waypoint_new import PerspectiveTransform

class pybullet_dynamics:

  def sim():
    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
      p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    useRealTimeSim = 1

    distance = 100000
    
    #for video recording (works best on Mac and Linux, not well on Windows)
    #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
    p.setRealTimeSimulation(useRealTimeSim)  # either this
    p.loadSDF("stadium.sdf")
    # p.loadURDF("plane.urdf")
    # Wall1Id = p.createCollisionShape(p.GEOM_BOX,
    #                                   halfExtents=[0.05,5,0.5])
    # Wall2Id = p.createCollisionShape(p.GEOM_BOX,
    #                                   halfExtents=[0.05,2,0.5])
    # Wall3Id = p.createCollisionShape(p.GEOM_BOX,
    #                                   halfExtents=[0.05,1.5,0.5])
    # Wall4Id = p.createCollisionShape(p.GEOM_BOX,
    #                                   halfExtents=[0.05,3.5,0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall1Id,basePosition=[5, 0, 0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall1Id,basePosition=[0, 5, 0.5],baseOrientation=p.getQuaternionFromEuler([0,0,67.55]))
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall1Id,basePosition=[0, -5, 0.5],baseOrientation=p.getQuaternionFromEuler([0,0,-67.55]))
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall1Id,basePosition=[-5, 0, 0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall2Id,basePosition=[-1, 3, 0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall3Id,basePosition=[-3.5, 1, 0.5],baseOrientation=p.getQuaternionFromEuler([0,0,67.55]))
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall4Id,basePosition=[1, -1.5, 0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall3Id,basePosition=[3.5, 2, 0.5],baseOrientation=p.getQuaternionFromEuler([0,0,67.55]))
    orn = p.getQuaternionFromEuler([0, 0, math.pi / 2.])
    car = p.loadURDF("racecar/racecar_differential.urdf",[3,0,1],orn)  #, [0,0,2],useFixedBase=True)
    for i in range(p.getNumJoints(car)):
      print(p.getJointInfo(car, i))
    for wheel in range(p.getNumJoints(car)):
      p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
      p.getJointInfo(car, wheel)

    for i in range(100):
      p.stepSimulation()
      time.sleep(1. / 240.)

    wheels = [8, 15]
    print("----------------")

    #p.setJointMotorControl2(car,10,p.VELOCITY_CONTROL,targetVelocity=1,force=10)
    c = p.createConstraint(car,
                          9,
                          car,
                          11,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car,
                          10,
                          car,
                          13,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,
                          9,
                          car,
                          13,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,
                          16,
                          car,
                          18,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(car,
                          16,
                          car,
                          19,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,
                          17,
                          car,
                          19,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,
                          1,
                          car,
                          18,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    c = p.createConstraint(car,
                          3,
                          car,
                          19,
                          jointType=p.JOINT_GEAR,
                          jointAxis=[0, 1, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    return car, wheels, distance

  def loop(velo, force, delta, wheels, car, distance):
    steering = [0, 2]
    img_w, img_h = 120, 80
    maxForce = force
    targetVelocity = velo
    steeringAngle = delta
    useRealTimeSim = 1
    #print(targetVelocity)

    for wheel in wheels:
      p.setJointMotorControl2(car,
                              wheel,
                              p.VELOCITY_CONTROL,
                              targetVelocity=targetVelocity,
                              force=maxForce)

    for steer in steering:
      p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=-steeringAngle)
      agent_pos, agent_orn =p.getBasePositionAndOrientation(car)

      yaw = p.getEulerFromQuaternion(agent_orn)[-1]
      xA, yA, zA = agent_pos
      zA = zA + 0.3 # make the camera a little higher than the robot

      # compute focusing point of the camera
      xB = xA + math.cos(yaw) * distance
      yB = yA + math.sin(yaw) * distance
      zB = zA

      view_matrix = p.computeViewMatrix(
                          cameraEyePosition=[xA, yA, zA],
                          cameraTargetPosition=[xB, yB, zB],
                          cameraUpVector=[0, 0, 1.0]
                      )

      projection_matrix = p.computeProjectionMatrixFOV(
                              fov=90, aspect=1.5, nearVal=0.02, farVal=3.5)

      imgs = p.getCameraImage(img_w, img_h,
                              view_matrix,
                              projection_matrix, shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
      frame = cv2.resize(imgs[2], (640, 480))
        
      midpoint = PerspectiveTransform()
      midx, midy = midpoint.midpoint(frame)

    steering
    if (useRealTimeSim == 0):
      p.stepSimulation()
    time.sleep(0.01)
    return p.getBasePositionAndOrientation(car)