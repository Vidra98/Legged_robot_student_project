"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quad
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
import matplotlib
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv
from  env.quadruped import Quadruped 

import argparse

def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser()
    #Env parameters
    parser.add_argument('--plot', default=True, action='store_true',
                        help='plot caracteristics of the motion')
    parser.add_argument('--render', default=True,
                        help='visualize motion')
    parser.add_argument('--on_rack', default=False, action='store_true',
                        help='Put robot on track')
    parser.add_argument('--action_repeat', default=1, 
                        help='number of repetition of motion')
    parser.add_argument('--isRLGymInterface', default=False, 
                        help='set true to use RL')
    parser.add_argument('--motor_control_mode', type=str, default="TORQUE",
                        help='motor control mode')
    parser.add_argument('--add_noise', type=bool, default=True, 
                        help='noise on surface coefficient')
    parser.add_argument('--record_video', default=False,
                        help='record motion')

    parser.add_argument('--add_cartesian_pd', default=True,action='store_true',
                        help='ADD_CARTESIAN_PD')    
    parser.add_argument('--number_of_step', type=int, default=2,
                        help='number of steps taken')        
    #hopf parameters
    parser.add_argument('--mu', type=float, default=1, 
                        help='converge to sqrt(mu)')
    parser.add_argument('--omega_swing', type=float, default=16, 
                        help='amplitude of the swing')
    parser.add_argument('--omega_stance', type=float, default=4, 
                        help='amplitude of the swing')
    parser.add_argument('--gait', type=str, default="WALK", 
                        help='change depending on desired gait')
    parser.add_argument('--coupling_strength', type=float, default=1, 
                        help='coefficient to multiply coupling matrix')
    parser.add_argument('--couple', type=bool, default=True, 
                        help='should couple')
    parser.add_argument('--time_step', type=float, default=0.001, 
                        help='time step')
    parser.add_argument('--ground_clearance', type=float, default=0.06, 
                        help='foot swing height')
    parser.add_argument('--ground_penetration', type=float, default=0.0045, 
                        help='foot stance penetration into ground')
    parser.add_argument('--robot_height', type=float, default=0.25, 
                        help='in nominal case (standing)')
    parser.add_argument('--des_step_len', type=float, default=0.042, 
                        help='desired step length')
    args = parser.parse_args()
    return args

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**1,                # converge to sqrt(mu)
                omega_swing=0,  # MUST EDIT ---------------------------------------------------------------------
                omega_stance=0, # MUST EDIT ---------------------------------------------------------------------
                gait="TROT",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,   # foot swing height 
                ground_penetration=0.01,# foot stance penetration into ground 
                robot_height=0.25,      # in nominal case (standing) 
                des_step_len=0.04,      # desired step length 
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions  
    self.X[0,:] = [3.,3.,3.,3.]#np.random.rand(4) * .1 
    self.X[1,:] = self.PHI[0] #[np.pi,0,0,np.pi]

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices 
    """
    self.PHI_trot = [[0,    np.pi, np.pi, 0],
                     [-np.pi,0,     0,     -np.pi],
                     [-np.pi,0,     0,     -np.pi],
                     [0,    np.pi, np.pi, 0]]

    self.PHI_walk = [[0,-np.pi,-np.pi/2,np.pi/2],
                     [np.pi,0,np.pi/2,3*np.pi/2],
                     [np.pi/2,-np.pi/2,0,np.pi],
                     [-np.pi/2,-3*np.pi/2,-np.pi/2,0]]

    self.PHI_bound =[[0,0,-np.pi,-np.pi],
                     [0,0,-np.pi,-np.pi],
                     [np.pi,np.pi,0,0],
                     [np.pi,np.pi,0,0]]

    self.PHI_pace = [[0,-np.pi,0,-np.pi],
                     [np.pi,0,np.pi,0],
                     [0,-np.pi,0,-np.pi],
                     [np.pi,0,np.pi,0]]

    if gait == "TROT":
      print('TROT')
      self.PHI = self.PHI_trot
      self.X[0,:] = [3.,3.,3.,3.]
      self.X[1,:] = [np.pi,0,0,np.pi]
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      print('BOUND')
      self.X[0,:] = [3.,3.,3.,3.]
      self.X[1,:] = [0,0,np.pi,np.pi]
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate  
    self._integrate_hopf_equations()
    X = self.X.copy()
    r, theta = X[0,:], X[1,:] 
    z=np.zeros((4,1))
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    x =  - self._des_step_len * r * np.cos(theta)# [TODO]   

    mask=np.sin(theta) > 0
    for i in range(4):
      if np.sin(theta[i]) > 0:
        z[i]= - self._robot_height + self._ground_clearance * np.sin(theta[i])
      else:
        z[i]= - self._robot_height + self._ground_penetration * np.sin(theta[i])

    
    return x, z,r,theta
      
        
  def _integrate_hopf_equations(self): 
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50 
    rad2deg=180/np.pi
    deg2rad=np.pi/180
    # get r_i, theta_i from X
    r, theta = X[0,:], X[1,:]
    # loop through each leg's oscillator 
    for i in range(4):
      
      
      # compute r_dot (Equation 6)
      r_dot = alpha * (self._mu - r[i]**2)*r[i] # [TODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      if np.sin(theta[i]) > 0:
          theta_dot = self._omega_swing #+ # [TODO]  
      else:
          theta_dot = self._omega_stance #+ 

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(4):
          if j != i:
            theta_dot += r[j]*self._coupling_strength*np.sin((theta[j]-theta[i]-self.PHI[i][j])) # [TODO]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X+X_dot*self._dt # [TODO]
    # mod phase variables to keep between 0 and 2pi  
    self.X[1,:] = self.X[1,:] % (2*np.pi)



if __name__ == "__main__":

  args=cli()
  ADD_CARTESIAN_PD = False
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=args.render,              # visualize
                      on_rack=args.on_rack,              # useful for debugging! 
                      isRLGymInterface=args.isRLGymInterface,#False,     # not using RL
                      time_step=args.time_step,
                      action_repeat=args.action_repeat,
                      motor_control_mode=args.motor_control_mode,
                      add_noise=args.add_noise,          # start in ideal conditions
                      record_video=args.record_video
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(mu=args.mu,                           # converge to sqrt(mu)
                omega_swing=args.omega_swing*2*np.pi,     # MUST EDIT ---------------------------------------------------------------------
                omega_stance=args.omega_stance*2*np.pi,   # MUST EDIT ---------------------------------------------------------------------
                gait=args.gait,                           # change depending on desired gait
                coupling_strength=args.coupling_strength, # coefficient to multiply coupling matrix
                couple=args.couple,                       # should couple
                time_step=args.time_step,                 # time step 
                ground_clearance=args.ground_clearance,   # foot swing height 
                ground_penetration=args.ground_penetration,# foot stance penetration into ground 
                robot_height=args.robot_height,           # in nominal case (standing) 
                des_step_len=args.des_step_len            # desired step length 
                )

  TEST_STEPS = int(args.number_of_step / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  Total_leg_xyz=np.zeros((3,4,TEST_STEPS))
  Total_tau=np.zeros((3,4,TEST_STEPS))
  Total_desired_trajectory_q=np.zeros((3,4,TEST_STEPS))
  Total_obtained_trajectory=np.zeros((3,4,TEST_STEPS))
  Total_CoT=np.zeros((TEST_STEPS))
  Total_foot_step=np.zeros((4,TEST_STEPS))
  Total_velocity=np.zeros((3,TEST_STEPS))

  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])

  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)

  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12) 
    # get desired foot positions from CPG 
    xs,zs,r,theta = cpg.update()
    #print('\nr',r,'\ntheta\n',theta)
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quad.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i] ,sideSign[i]*foot_y , zs[i,0]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quad.py)
      leg_q = env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO] 
      # Add joint PD contribution to tau for leg i (Equation 4)
      tau += np.multiply(kp,(leg_q.transpose()-q[3*i:3*i+3].transpose()))+np.multiply(kd,(0-dq[3*i:3*i+3].transpose())) # [TODO] 
      # add Cartesian PD contribution
      if args.add_cartesian_pd:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quad.py)
        # Get current foot velocity in leg frame (Equation 2)
        
        Jacob, pos =env.robot.ComputeJacobianAndPosition(i)
        p=pos
        v=np.dot(Jacob,dq[3*i:3*i+3])

        desired_pos=leg_xyz
 
        tau_tmp=np.dot(Jacob.transpose(),np.dot(kpCartesian,(desired_pos.transpose()-p.transpose()))+np.dot(kdCartesian,(-v.transpose())))

        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau += tau_tmp  #[TODO]

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

      Total_leg_xyz[:,i,j]=leg_xyz
      Total_tau[:,i,j]=tau
      Total_desired_trajectory_q[:,i,j]=q[3*i:3*i+3]
      Total_obtained_trajectory[:,i,j]=leg_q.transpose()
    # send torques to robot and simulate TIME_STEP seconds 
    env.step(action) 
    robot_mass=12.45401 #mass from urdf
    Total_foot_step[:,j]=env.robot.GetContactInfo()[3]
    motorVelocity=env.robot.GetMotorVelocities()
    motorTorque=env.robot.GetMotorTorques() 
    Total_velocity[:,j]=env.robot.GetBaseLinearVelocity()

    #Compute the CoT
    Total_CoT[j]=np.sum(np.abs(motorVelocity@motorTorque))/(np.abs(Total_velocity[0,j])*robot_mass*9.81)

  ##################################################### 
  # PLOTS
  #####################################################
  print('---------------------------------------------------------------------------------------')
  if args.plot==True:
    CoT = plt.figure()
    plt.plot(Total_CoT)
    plt.ylabel('Cost of transport')
    plt.xlabel('Time [ms]')
    plt.title('CoT')
    print('Average CoT', np.mean(Total_CoT[5000:]))
    Speed = plt.figure()
    plt.plot(Total_velocity[0,:])
    plt.plot(Total_velocity[1,:])
    plt.plot(Total_velocity[2,:])
    plt.legend(['x', 'y', 'z'])
    plt.ylabel('Speed m/s')
    plt.xlabel('Time [ms]')
    plt.title('Velocity')
    print('Average velocity', np.mean(Total_velocity[0,-1000:]))
    Foot = plt.figure()
    plt.plot(Total_foot_step[0,:])
    plt.plot(Total_foot_step[1,:])
    plt.plot(Total_foot_step[2,:])
    plt.plot(Total_foot_step[3,:])
    plt.legend(['FR', 'FL', 'RR', 'RL'])
    plt.ylabel('Foot contact [1 if there is contact]')
    plt.xlabel('Time [ms]')
    plt.title('Foot contact')
    
    steps_edge=np.array(Total_foot_step[0,1:]-Total_foot_step[0,:-1])
    number_of_steps=np.sum(steps_edge==1)
    Average_stance=np.sum(Total_foot_step[0,:])/(number_of_steps*1000)
    Average_swing=(TEST_STEPS-np.sum(Total_foot_step[0,:]))/(number_of_steps*1000)
    Step_duration=Average_stance+Average_swing
    print('Total stance time in second', np.sum(Total_foot_step[0,:])/1000, 'Average stance time in second', Average_stance,'Average swing time',Average_swing)
    print('Steps duration [s]',Average_stance+Average_swing)
    print('Duty cycle', Average_stance/Step_duration)

    gait_plt_x = plt.figure()
    plt.plot(Total_leg_xyz[0,0,:])
    plt.plot(Total_leg_xyz[0,1,:])
    plt.plot(Total_leg_xyz[0,2,:])
    plt.plot(Total_leg_xyz[0,3,:])
    plt.legend(['FR', 'FL', 'RR', 'RL'])
    plt.title('X position for each leg')

    gait_plt_y = plt.figure()
    plt.plot(Total_leg_xyz[1,0,:])
    plt.plot(Total_leg_xyz[1,1,:])
    plt.plot(Total_leg_xyz[1,2,:])
    plt.plot(Total_leg_xyz[1,3,:])
    plt.legend(['FR', 'FL', 'RR', 'RL'])
    plt.title('Y position for each leg')

    gait_plt_z = plt.figure()
    plt.plot(Total_leg_xyz[2,0,:])
    plt.plot(Total_leg_xyz[2,1,:])
    plt.plot(Total_leg_xyz[2,2,:])
    plt.plot(Total_leg_xyz[2,3,:])
    plt.legend(['FR', 'FL', 'RR', 'RL'])
    plt.title('Z position for each leg')

    leg_plt = plt.figure()
    plt.subplot(3,1,1)  
    plt.plot(Total_leg_xyz[0,1,:])
    plt.title('Goal X position relative to body')
    plt.subplot(3,1,2)  
    plt.plot(Total_leg_xyz[1,1,:])
    plt.title('Goal Y position relative to body')
    plt.subplot(3,1,3)  
    plt.plot(Total_leg_xyz[2,1,:])
    plt.title('Goal Z position relative to body')

    tau_plt = plt.figure()
    plt.subplot(3,1,1)  
    plt.plot(Total_tau[0,1,:])
    plt.title('applied X torque absolute')
    plt.subplot(3,1,2)  
    plt.plot(Total_tau[1,1,:])
    plt.title('applied Y torque absolute')
    plt.subplot(3,1,3)  
    plt.plot(Total_tau[2,1,:])
    plt.title('applied Z torque absolute')

    q_traj_plt = plt.figure()
    plt.subplot(3,1,1)  
    plt.plot(Total_desired_trajectory_q[0,1,:])
    plt.plot(Total_obtained_trajectory[0,1,:])
    plt.title('X position relative to body')
    plt.legend('Desired','Obtained')

    plt.subplot(3,1,2)  
    plt.plot(Total_desired_trajectory_q[1,1,:])
    plt.plot(Total_obtained_trajectory[1,1,:])
    plt.title('Y position relative to body')
    plt.legend('Desired','Obtained')

    plt.subplot(3,1,3)  
    plt.plot(Total_desired_trajectory_q[2,1,:])
    plt.plot(Total_obtained_trajectory[2,1,:])
    plt.title('Z position relative to body')
    plt.legend('Desired','Obtained')

    plt.show()