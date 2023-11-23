# FOR_Project

## Model Predictive Control (MPC) for Unmanned Ground Vehicles (UGV)

## Problem Statement:
Model Predictive Control(MPC) for trajectory tracking on Unmanned Ground Vehicle (UGV) with waypoint generation in unknown environment using perception. 

#### Possible Extensions: (Maybe some day in future :P)  
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

##### Hough Line: 
![](https://github.com/prakrutk/FOR_Project/blob/Prakrut/Waypoint_generation/result.png)

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
$s_f$ is the slip ratio of the front wheel. \
$s_r$ is the slip ratio of the rear wheel. \
$c_l$ is the longitudinal stiffness of the tire. \
$\alpha_f$ is the slip angle of the front wheel. \
$\alpha_r$ is the slip angle of the rear wheel. 

Assuming small slip angle and small slip ratio, the forces acting on the car can be written as:

$$ f_{f_x} = C_ls_f$$ 

$$ f_{f_y} = C_c\alpha_f$$

With the assumptions and substituting the above equations in the dynamics equation of the car, we get:

$$ m\ddot x = C_ls_f - C_c\alpha_f\delta + c_ls_r + m\dot y\dot \psi$$ 

$$m\ddot y = C_c\alpha_f + C_ls_f\delta + c_c\alpha_r - m\dot x\dot \psi $$ 

$$I_z\ddot \psi = l_f(C_ls_f\delta + C_c\alpha_f) - c_c\alpha_rl_r $$

Also,

$$ \dot x = V $$

$$ \dot y = \dot x(\alpha_f + \delta) - l_f\dot \psi $$

$$ \dot \psi = \frac{\dot y}{l_r} $$

#### MPC Formulation:

$$ \min_{\Delta U, \epsilon } \begin{bmatrix} \Delta U , \epsilon \end{bmatrix}^T H\begin{bmatrix} \Delta U , \epsilon \end{bmatrix} \begin{bmatrix} \Delta U , \epsilon \end{bmatrix} + f\begin{bmatrix} \Delta U , \epsilon \end{bmatrix}$$

$$ \Delta U_{min} \leq \Delta U \leq \Delta U_{max}$$

$$ U_{min} \leq u(t-1) + \sum_{i=t}^{t+N_c-1} \Delta U(i) \leq U_{max}$$

$$ Y_{min} - \epsilon \leq \Phi_{X(t|t)} + \Theta \Delta U(t) \leq Y_{max} + \epsilon$$

Where: 

$$H = \begin{bmatrix} \Theta ^T Q \Theta + R & 0 \\\\ 0 & \rho \end{bmatrix}$$

$$f = \begin{bmatrix} 2E^TQ\Theta & 0\end{bmatrix}$$

$$\epsilon = \text{Slack variable} $$

$$Y = \Phi_{X(t|t)} + \Theta \Delta U(t)$$

### Setup:
This is the setup for ubuntu(22.04). Not sure how it works on Windows/Mac.

**(Recommended)** Make a seperate conda environment and install the package in that environment: \
``` conda create -n FOR_Project python=3.8``` \
``` conda activate FOR_Project``` 

First clone the repository: \
``` git clone https://github.com/prakrutk/FOR_Project.git```

Checkout to the branch named 'Prakrut': \
``` git checkout Prakrut```

Then go into the directory and install the package using pip: \
``` cd FOR_Project``` \
``` pip install --upgrade pip``` \
``` pip install -e .``` \
``` pip install -r requirements.txt``` 

To run MPC code: \
``` python3 dynamics/MPC.py``` (disclaimer: Something is working now we have to figure out what exactly is working)

To run Waypoint generation code: \
``` python3 Waypoint_generation/Waypoint_new.py```

