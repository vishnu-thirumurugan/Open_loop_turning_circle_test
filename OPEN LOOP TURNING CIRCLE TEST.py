# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:49:04 2022

@author: oe21s024
"""


import numpy as vishnu
h = 1
start = 0
end = 10000
t = vishnu.linspace(start, end, int(end/h))


# Nomoto constants 
T1 = 118
T2 = 7.8
T3 = 18.5
T = T1 +T2 - T3 # CALCULATE NOMOTO CONSTANTS 
K = 0.185       # rudder constant
rudder_angle = 10
delta = (vishnu.pi*rudder_angle)/180 # prescribed rudder angle in radians
v = 10  # velocity of ship

x  = vishnu.zeros(len(t))
y  = vishnu.zeros(len(t))


# defining states
state = vishnu.zeros((2,1))
psi = state[0] # yaw angle 
r = state[1]   # yaw rate
psi =  vishnu.zeros(len(t))
r = vishnu.zeros(len(t))

# initial conditions 
x[0] = 0
y[0] = 0 


# dynamics - state and position
def dynamics(r,psi,v,delta,K,T):
    pshi = (K* delta - r)/T
    psi_dot = r 
    X  = v * vishnu.cos(psi)
    Y  = v * vishnu.sin(psi)
    
    return pshi, psi_dot, X, Y

# RUNGE KUTTA FOURTH ORDER INTEGRATION

for i in range(0, len(t)-1):
    pshi, psi_dot, X, Y = dynamics(r[i],psi[i],v,delta,K,T)
    k1, l1, m1, n1 = h*pshi, h*psi_dot, h*X, h*Y
    
    pshi, psi_dot, X, Y = dynamics(r[i]+0.5*k1,psi[i]+0.5*l1,v,delta,K,T)
    k2, l2, m2, n2 = h*pshi, h*psi_dot, h*X, h*Y
    
    pshi, psi_dot, X, Y = dynamics(r[i]+0.5*k2,psi[i]+0.5*l2,v,delta,K,T)
    k3, l3, m3, n3 = h*pshi, h*psi_dot, h*X, h*Y
    
    pshi, psi_dot, X, Y = dynamics(r[i]+k3,psi[i]+l3,v,delta,K,T)
    k4, l4, m4, n4 = h*pshi, h*psi_dot, h*X, h*Y
    
    r[i+1] = r[i] + (k1 + (2*k2) + (2*k3) + k4) / 6
    psi[i+1] = psi[i] + (l1 + (2*l2) + (2*l3) + l4) / 6
    x[i+1]   = x[i] + (m1 + (2 * m2) + (2 * m3)+ m4) / 6
    y[i+1] = y[i] + (n1 + (2 * n2) + (2 * n3) + n4) / 6
    


import matplotlib.pyplot as plt 
def plot_learning_curve():
    
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [5, 15]
       
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    
    ax1.plot(t, psi)
    ax1.set_xlabel('Time step', color = 'C0')
    ax1.set_ylabel('Yaw angle', color = 'C0')
    ax1.tick_params(axis = 'x', colors = 'C0')
    ax1.tick_params(axis = 'y', colors = 'C0')
    
    
    ax2.plot(t, r)
    ax2.set_xlabel('Time step', color = 'C0')
    ax2.set_ylabel('Yaw rate', color = 'C0')
    ax2.tick_params(axis = 'x', colors = 'C0')
    ax2.tick_params(axis = 'y', colors = 'C0')

    ax3.plot(t, x)
    ax3.set_xlabel('Time step', color = 'C0')
    ax3.set_ylabel('x', color = 'C0')
    ax3.tick_params(axis = 'x', colors = 'C0')
    
    ax4.plot(x, y)
    ax3.tick_params(axis = 'y', colors = 'C0')
    ax4.set_xlabel('y', color = 'C0')
    ax4.set_ylabel('x', color = 'C0')
    ax4.tick_params(axis = 'x', colors = 'C0')
    ax4.tick_params(axis = 'y', colors = 'C0')
    
    # plt.legend()
    plt.show()
    
    

plot_learning_curve() 
       
    


