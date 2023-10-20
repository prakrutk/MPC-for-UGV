# FOR_Project

## Model Predictive Control (MPC) for Unmanned Ground Vehicles (UGV)

## Problem Statement:
Model Predictive Control(MPC) for trajectory tracking on Unmanned Ground Vehicle (UGV) with waypoint generation in unknown environment using perception. 

#### Possible Extensions:  
1. Dynamic obstacle avoidance
2. Implementation and demonstration on hardware

### Simulation result: 

#### PyBullet: 
 
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Pybullet/Sim.gif)

#### Gazebo: 

![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/ROS-Gazebo/Cafe_Husky.jpeg)

### Waypoint Generation: 
##### Actual Image: 
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Waypoint_generation/Test.png)

##### Canny-Edge detection: 
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Waypoint_generation/canny.png)

##### Contours: 
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Waypoint_generation/contours.png)

### Dynamics Model:
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Model.png)

##### Notations:
State variable: $ X = (x,y,\psi , \dot x, \dot y, \dot \psi  )$

Input/control variable: $U = (\delta, \omega )$

Where, \
$x,y$ are coordinates of the COM of the car in world frame. \
$\psi$ is the heading angle of the car. \
$\delta$ is the steering angle of the car. \
$\omega$ is the rotational speed of both the wheels.

$f_{f_x} ,f_{f_y} ,f_{r_x} ,f_{r_y}$ are the force acting in the body frame of front and rear wheel of the car respectively. 

$l_f$ is the distance of the front wheel from the COM of the car. \
$l_r$ is the distance of the rear wheel from the COM of the car. \
$\beta$ is the sideslip angle of the car. (Not used) 

#### Dynamics equation of the car: 

$$ m\ddot x = f_{f_x}cos\delta - f_{f_y}sin\delta + f_{r_x} + m\dot y\dot \psi$$ 

$$m\ddot y = f_{f_y}cos\delta + f_{f_x}sin\delta + f_{r_y} - m\dot x\dot \psi$$

$$I_z\ddot \psi = l_f(f_{f_x}sin\delta + f_{f_y}cos\delta) - f_{r_y}l_r$$

$C_l$ is the cornering stiffness of the tire. \
$s_f$ is the slip angle of the front wheel. \
$s_r$ is the slip angle of the rear wheel. \
$c_l$ is the longitudinal stiffness of the tire. \
$\alpha_f$ is the slip ratio of the front wheel. \
$\alpha_r$ is the slip ratio of the rear wheel. 

Assuming small slip angle and small slip ratio, the forces acting on the car can be written as:

$$ f_{f_x} = C_ls_f$$ 

$$ f_{f_y} = C_c\alpha_f$$

With the assumptions and substituting the above equations in the dynamics equation of the car, we get:

$$ m\ddot x = C_ls_f - C_c\alpha_f\delta + c_ls_r + m\dot y\dot \psi$$ 

$$m\ddot y = C_c\alpha_f + C_ls_f\delta + c_c\alpha_r - m\dot x\dot \psi $$ 

$$I_z\ddot \psi = l_f(C_ls_f\delta + C_c\alpha_f) - c_c\alpha_rl_r $$

