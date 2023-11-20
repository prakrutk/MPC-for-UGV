import jax.numpy as jnp
from dynamics.cardynamics import dynamics
import cvxpy as cvx
import pybullet as p
import pybullet_data
import time
import math

Nc = 5
Np = 10
initial_state = jnp.array([0,0,0,0,0,0,0,0])
x_t = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
u_t = jnp.array([0.0,0.0])
xr = jnp.array([1.5,0.5,0.0,0.0,0.0,0.0])
ur = jnp.array([0.0,0.0])
delu = 0.1*jnp.ones((2*Nc,1))
Yreff = jnp.ones((3*Np,1))
Q = 100*jnp.identity(3*Np)
R = 10*jnp.identity(2*Nc)
tolerance = 0.01*jnp.ones((3*Np,1))
Ymax = Yreff + tolerance
Ymin = Yreff - tolerance
rho = 1
epi = 0
MAX_TIME = 100 # seconds

class state:

    def __init__(self, x, y, theta, xdot, ydot, thetadot):
        self.x = x
        self.y = y
        self.theta = theta
        self.xdot = xdot
        self.ydot = ydot
        self.thetadot = thetadot


coeff = dynamics(state = x_t
                ,input = u_t
                ,inputr = ur
                ,stater = xr
                ,delu = delu
                ,cl = 1
                ,sf = 1
                ,cc = 1
                ,sr = 1
                ,m = 1
                ,alphaf = 1
                ,lf = 1
                ,lr = 1
                ,iz = 1
                ,T = 0.1
                ,Nc = Nc
                ,Np = Np)

Ynext = coeff.Y(x_t,u_t)

def linearmpc(x,u_t):
    u = cvx.Variable((2*Nc +1,1))
    # u_t=u_t.reshape(2,1)
    E = coeff.phi().dot(jnp.concatenate((x_t-xr,u_t-ur),axis=0)).reshape(3*Np,1) - Yreff
    print('E=',E)

    cost = 0.0
    constraints = []
    the_c=jnp.concatenate((coeff.theta(),jnp.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = jnp.transpose(the_c).dot(Q).dot(the_c) + R 
    H = jnp.append(H,jnp.zeros((1,H.shape[1])),axis=0)
    c = jnp.zeros((H.shape[0],1))
    c = c.at[-1].set(rho)
    H = jnp.append(H,c,axis=1)
    cost += cvx.quad_form(u,H) + jnp.transpose(E).dot(Q).dot(the_c)@u[0:2*Nc,:]
    for k in range(2*Nc):
        constraints += [u[k,:] <= 0.5]
        constraints += [u[k,:] >= -0.5]
        #constraints += [u_t[k_t,:] <= 5]
    constraints += [Ymin - u[2*Nc,:] <= coeff.Y(x,u)]
    constraints += [coeff.Y(x,u) <= Ymax + u[2*Nc,:]]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()
    return u.value

def check_goal(state, goal):
    if abs(4 + state[0] - goal.x) < 0.5 and abs(-4 + state[1] - goal.y) < 0.5:
        return True
    else:
        return False

def traj_gen(start,goal):
    x = start.x
    y = start.y
    theta = start.theta
    xdot = start.xdot
    ydot = start.ydot
    thetadot = start.thetadot
    xg = goal.x
    yg = goal.y
    thetag = goal.theta
    xdotg = goal.xdot
    ydotg = goal.ydot
    thetadotg = goal.thetadot
    t = 0
    while t < MAX_TIME:
        x = x + xdot*0.1
        y = y + ydot*0.1
        theta = theta + thetadot*0.1
        xdot = xdot + xdotg*0.1
        ydot = ydot + ydotg*0.1
        thetadot = thetadot + thetadotg*0.1
        t += 0.1
        yield state(x,y,theta,xdot,ydot,thetadot)
        

def pyconnect(v,d,wheels,car,useRealTimeSim):
    distance = 100000
    img_w, img_h = 120, 80
    steering = [0, 2]
    maxForce = 20
    targetVelocity = v
    steeringAngle = d
    print("targetVelocity=",targetVelocity,"steeringAngle=",steeringAngle)
    pos,orn =p.getBasePositionAndOrientation(car)
    p.addUserDebugLine(pos,[pos[0],pos[1],0.1],lineColorRGB=[1,0,0],lineWidth=5)
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
    steering
    if (useRealTimeSim == 0):
        p.stepSimulation()
    time.sleep(0.01)
    return p.getBasePositionAndOrientation(car)


def simulate(initial_state,goal):
    goal = goal 
    state = initial_state
    time = 0.0
    
    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    useRealTimeSim = 1
    p.setRealTimeSimulation(useRealTimeSim) 


    p.loadURDF("plane.urdf")
    car = p.loadURDF("racecar/racecar_differential.urdf",[-4,4,1])
    for i in range(100):
        p.stepSimulation()
    for i in range(p.getNumJoints(car)):
        print(p.getJointInfo(car, i))
    for wheel in range(p.getNumJoints(car)):
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.getJointInfo(car, wheel)
    wheels = [8, 15]
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
    u = u_t
    while MAX_TIME >= time:
        u = linearmpc(state,u)
        for i in range(Nc):
            x,phi = pyconnect(2*u[2*i,0],u[2*i+1,0],wheels,car,useRealTimeSim)
        state = jnp.array([x[0],x[1],x[-1],0,0,0])
        x_t = state
        time += 0.1
        i +=1

        if check_goal(state, goal):
            print("Goal")
            break

def main():
    initial_state = state(0,0,0,0,0,0)
    goal = state(1.5,0.5,0,0,0,0)
    simulate(initial_state,goal)

if __name__ == '__main__':
    main()