import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import pathlib
import pybullet as p
import time
import pybullet_data
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from Waypoint_generation.segment import Segment
import cv2

import cubic_spline_planner

NX = 6  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 1.0, 0.5, 0.0, 0.0])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 2.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 0.1  # [m]
WIDTH = 0.05  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.01  # [m]
WB = 0.1  # [m]

MAX_STEER = np.deg2rad(10.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(1.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = False


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, xdot=0.0, ydot=0.0, yawdot=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.xdot = xdot
        self.ydot = ydot
        self.yawdot = yawdot
        self.predelta = None


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(x, u):

    lr = 0.05
    lf = 0.15
    Cc = 8.2
    m = 6.38
    Iz = 0.058
    A = np.identity(NX)*DT
    A[0,2] = 0
    A[0,3] = math.cos(x[2])*DT
    A[0,4] = -math.sin(x[2])*DT
    A[1,2] = 0
    A[1,3] = math.sin(x[2])*DT
    A[1,4] = math.cos(x[2])*DT
    A[2,4] = lr*DT
    A[3,3] += (Cc*u*(lr +lf))/m*lr*DT
    A[3,4] = x[5]*DT
    A[3,5] = x[4]*DT
    A[4,3] = ((Cc*(lr +lf))/m*lr + x[5])*DT
    A[4,5] = x[3]*DT
    A[5,3] = Cc*(lf**2 - lr**2)/Iz*DT

    # print(A)


    B = np.zeros((NX, NU))
    B[3, 0] = DT
    # B[3, 1] = (Cc*x[3]*(lf +lr)/m*lr - 2*u*Cc/m)*DT
    B[4, 1] = Cc/m*DT
    B[5, 1] = Cc*(lf**2 - lr**2)/Iz*DT

    # print(B)

    return A, B


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, a, delta,car,wheels,useRealTimeSim):

    lr = 0.05
    lf = 0.15
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER
    
    agent_pos, agent_orn,vel,ome,midx,midy = pyconnect(state.xdot,delta,wheels,car,useRealTimeSim)
    # state.x = agent_pos[0]
    # state.y = agent_pos[1] 
    # state.yaw = p.getEulerFromQuaternion(agent_orn)[-1]
    # state.xdot = vel[0]
    # state.ydot = vel[1]
    # state.yawdot = ome[2]

    state.x = state.x + state.xdot * math.cos(state.yaw) * DT - state.ydot * math.sin(state.yaw) * DT
    state.y = state.y + state.xdot * math.sin(state.yaw) * DT + state.ydot * math.cos(state.yaw) * DT
    state.yaw = state.yaw + state.yawdot * DT
    state.xdot = state.xdot + a * DT
    state.ydot = state.ydot 
    state.yawdot = state.yawdot 

    if state.xdot > MAX_SPEED:
        state.xdot = MAX_SPEED
    elif state.xdot < MIN_SPEED:
        state.xdot = MIN_SPEED

    return state, midx, midy


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref,car,wheels,useRealTimeSim):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2], xdot=x0[3], ydot=x0[4], yawdot=x0[5])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state, midx, midy = update_state(state, ai, di,car,wheels,useRealTimeSim)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw
        xbar[3, i] = state.xdot
        xbar[4, i] = state.ydot
        xbar[5, i] = state.yawdot

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od,car,wheels,useRealTimeSim):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw, oxdot, oydot, oyawdot = None, None, None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref,car,wheels,useRealTimeSim)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, oxdot, oydot, oyawdot = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, oxdot, oydot, oyawdot


def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B = get_linear_model_matrix(
            xbar[:,t], dref[0,t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[3, :] <= MAX_SPEED]
    constraints += [x[3, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        oyaw = get_nparray_from_matrix(x.value[2, :])
        oxdot = get_nparray_from_matrix(x.value[3, :])
        oydot = get_nparray_from_matrix(x.value[4, :])
        oyawdot = get_nparray_from_matrix(x.value[5, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, oxdot, oydot, oyawdot = None, None, None, None, None, None, None, None
    
    # print(ox,oy,oyaw,oxdot,oydot,oyawdot)

    return oa, odelta, ox, oy, oyaw, oxdot, oydot, oyawdot


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    xref[3, 0] = sp[ind]
    xref[4, 0] = 0.0  
    xref[5, 0] = 0.0
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.xdot) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            xref[3, i] = sp[ind + dind]
            xref[4, 0] = 0.0  
            xref[5, 0] = 0.0
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            xref[3, i] = sp[ncourse - 1]
            xref[4, 0] = 0.0  
            xref[5, 0] = 0.0
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.xdot) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False

def pyconnect(v,d,wheels,car,useRealTimeSim):
    distance = 100000
    img_w, img_h = 120, 80
    steering = [0, 2]
    maxForce = 20
    targetVelocity = v
    steeringAngle = d
    print(d)
    #print(targetVelocity)
    pos,orn =p.getBasePositionAndOrientation(car)
    p.addUserDebugLine(pos,[pos[0],pos[1],0.1],lineColorRGB=[1,0,0],lineWidth=5)
    for wheel in wheels:
        p.setJointMotorControl2(car,
                                wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity=targetVelocity,
                                force=maxForce)

    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)
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
    
    # frame, depth = imgs[2], imgs[1]
    # cv2.imshow('depth',depth)
    # cv2.waitKey(100)
    # cv2.imshow('frame',frame)
    # cv2.waitKey(100)
    # cv2.waitKey(100000)
    frame = cv2.resize(imgs[2], (640, 480))
    vel,ome=p.getBaseVelocity(car)
    # enable = 1
    # pos = pos + np.array([0,0,2])
    # orn = p.getQuaternionFromEuler([0,90,0])
    # p.setVRCameraState(pos, orn)
    d = np.sqrt((pos[0])**2 + (pos[1])**2)
    p.resetDebugVisualizerCamera(5, 0, -70, pos)
    midpoint = Segment()
    midx,midy = midpoint.read_video(frame)
    steering

    if (useRealTimeSim == 0):
        p.stepSimulation()
    time.sleep(0.01)
    return agent_pos, agent_orn,vel,ome,midx,midy

def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state,car,wheels,useRealTimeSim):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """
    
    goal = [5, 0]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    xdot = [state.xdot]
    ydot = [state.ydot]
    yawdot = [state.yawdot]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    # pos,orn,midx,midy = pyconnect(0,0,wheels,car,useRealTimeSim)
    # cx,cy,cyaw,ck = generatetraj(midx,midy,dl)
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)
    
    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.yaw, state.xdot, state.ydot, state.yawdot]  # current state

        oa, odelta, ox, oy, oyaw, oxdot, oydot, oyawdot = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta,car,wheels,useRealTimeSim)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state, midx, midy = update_state(state, ai, di, car,wheels,useRealTimeSim)

        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        xdot.append(state.xdot)
        ydot.append(state.ydot)
        yawdot.append(state.yawdot)
        t.append(time)
        d.append(di)
        a.append(ai)

        # for i in range(midx.shape[0]):
        #     midx[i] = state.x + midx[i]*math.cos(state.yaw) - midy[i]*math.sin(state.yaw)
        #     midy[i] = state.y + midy[i]*math.cos(state.yaw) + midx[i]*math.sin(state.yaw)

        # cx1,cy1,cyaw1,ck1 = generatetraj(midx,midy,dl) 
        # sp1 = calc_speed_profile(cx1, cy1, cyaw1, TARGET_SPEED)
        # cx = np.concatenate((cx[0:target_ind],cx1))
        # cy = np.concatenate((cy[0:target_ind],cy1))
        # cyaw = np.concatenate((cyaw[0:target_ind],cyaw1))
        # ck = np.concatenate((ck[0:target_ind],ck1))
        # sp = np.concatenate((sp[0:target_ind],sp1))
        # cx.extend(cx1)
        # cy.extend(cy1)
        # cyaw.extend(cyaw1)
        # ck.extend(ck1)
        # sp.extend(sp1)
        
        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.xdot * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, xdot, ydot, yawdot, d, a



def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 8.0, 10.0, 12.0, 17.0, 20.0, 22.0]
    ay = [0.0, 0.0, 7.0, 9.0, 6.0, 4.0, 2.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_switch_back_course(dl):
    ax = [0.0, 3.0, 6.0, 10.0, 15.0]
    ay = [0.0, 0.0, 2.0, 1.0, 2.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck

def generatetraj(midx,midy,dl):
    ax = midx
    ay = midy
    # print(midx)
    # print(midy)
    cx,cy,cyaw,ck,s = cubic_spline_planner.calc_spline_course(ax,ay,ds=dl)
    return cx,cy,cyaw,ck


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course(dl)
    # cx, cy, cyaw, ck = get_straight_course2(dl)
    # cx, cy, cyaw, ck = get_straight_course3(dl)
    # cx, cy, cyaw, ck = get_forward_course(dl)
    # cx, cy, cyaw, ck = get_switch_back_course(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], xdot=0.0, ydot=0.0, yawdot=0.0)

    cid = p.connect(p.SHARED_MEMORY)
    if (cid < 0):
        p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    useRealTimeSim = 1
    p.setRealTimeSimulation(useRealTimeSim) 


    p.loadURDF("plane.urdf")

    # p.loadSDF("stadium.sdf")
    car = p.loadURDF("racecar/racecar_differential.urdf",[0,0,1])
    Wall2Id = p.createVisualShape(p.GEOM_BOX,halfExtents=[0.5,0.5,0.5],rgbaColor=[0,0.156,0,1])
    Wall1Id = p.createCollisionShape(p.GEOM_BOX,
                                      halfExtents=[0.5,0.5,0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall1Id,baseVisualShapeIndex=Wall2Id,basePosition=[6, 19.5, 0.5])
    Wall3Id = p.createVisualShape(p.GEOM_BOX,halfExtents=[0.5,0.5,0.5],rgbaColor=[0,0.156,0,1])
    Wall4Id = p.createCollisionShape(p.GEOM_BOX,
                                      halfExtents=[0.5,0.5,0.5])
    # p.createMultiBody(baseMass=0,baseCollisionShapeIndex=Wall4Id,baseVisualShapeIndex=Wall3Id,basePosition=[4, 20.5, 0.5])
    for i in range(100):
        p.stepSimulation()
    for i in range(p.getNumJoints(car)):
        print(p.getJointInfo(car, i))
    for wheel in range(p.getNumJoints(car)):
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.getJointInfo(car, wheel)
    wheels = [8, 15]
    # for i in range(len(cx)):
        # p.addUserDebugLine([cx[i],cy[i],0],[cx[i],cy[i],1],lineColorRGB=[0,1,0],lineWidth=5)
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


    t, x, y, yaw, xdot, ydot, yawot, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state,car,wheels,useRealTimeSim)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, xdot, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main2():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
    # main2()
