#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:35:39 2022

@author: parallels
"""
import sys
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from enviroment.hpendulum_2 import HPendulum
import time
import matplotlib.pyplot as plt

#FOLDER = 'Q_weights_backup/'
FOLDER = 'Q_weights_backup/tr/'


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)
### --- Hyper paramaters

GAMMA                  = 0.9           # Discount factor 
nprint                 = 10
PLOT                   = True
JOINT_COUNT            = 2
NU                     = 11
ITR                    = 200
THRESHOLD              = 1e-3

def get_critic(nx):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(JOINT_COUNT*NU)(state_out4)

    model = tf.keras.Model(inputs, outputs)

    return model
def reset_env():
    if JOINT_COUNT == 1:
        x0  = np.array([[np.pi], [0.]])
    elif JOINT_COUNT == 2:
        x0  = np.array([[np.pi, 0.], [0., 0.]])
    else:
        x0 = None
    return env.reset(x0) , 0.0 , 1
    
def simulate_folder(itr=100):
    directory = glob.glob(FOLDER+'*')
    for file in sorted(directory):
        if file.endswith(".h5"):
            print('loading file' , file)
            Q.load_weights(file)
            x , ctg , gamma_i = reset_env() 
            for i in range(ITR):      
                x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
                xu_check = np.c_[x_rep,u_list]
                pred = Q.predict(xu_check)
                u_ind = np.argmin(pred.sum(axis=1), axis=0)
                u = u_list[u_ind]
                x, cost = env.step(u)
                if (cost < THRESHOLD):
                    print (cost)
                ctg += gamma_i*cost
                gamma_i *= GAMMA
                env.render()

def simulate_sp(file_num,itr=200):
    directory = FOLDER + 'Q_weights_'
    file_name = directory + str(file_num) + '.h5'
    Q.load_weights(file_name)
    x , ctg , gamma_i = reset_env()
    for i in range(itr):      
        x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
        xu_check = np.c_[x_rep,u_list]
        pred = Q.predict(xu_check)
        u_ind = np.argmin(pred.sum(axis=1), axis=0)
        u = u_list[u_ind]
        x, cost = env.step(u)
        print(cost)
        ctg += gamma_i*cost
        gamma_i *= GAMMA
        env.render()

def simulate_folder2(itr=100):
    directory = glob.glob(FOLDER+'*')
    for file in sorted(directory):
        if file.endswith(".h5"):
            print('loading file' , file)
            Q.load_weights(file)
            x , ctg , gamma_i = reset_env() 
            for i in range(ITR):      
                u_output = Q.predict(x.reshape(1,-1))
                u = np.argmin(u_output.reshape(NU,JOINT_COUNT), axis=0)
                x, cost = env.step(u)
        #        print(cost)
                ctg += gamma_i*cost
                gamma_i *= GAMMA
                env.render()


def play_final(itr=300):
    simulate_sp('final',itr)

def simulate_sp2(eps_num,itr=100):
    directory = 'Q_weights_backup/tr/Q_weights_'
    file_name = directory + str(eps_num) + '.h5'
    Q.load_weights(file_name)
    x , ctg , gamma_i = reset_env()
    for i in range(itr):      
        u_output = Q.predict(x.reshape(1,-1))
        u = np.argmin(u_output.reshape(NU,JOINT_COUNT), axis=0)
        x, cost = env.step(u)
#        print(cost)
        ctg += gamma_i*cost
        gamma_i *= GAMMA
        env.render()

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = HPendulum(JOINT_COUNT, NU, dt=0.1)
    nx = env.nx
    Q = get_critic(nx)
    Q.summary()
    Q_target = get_critic(nx)
    Q_target.set_weights(Q.get_weights())

    t_start = t = time.time()

    # creating a matrix for controls based on JOINT_COUNT
    u_list1 = np.array(range(0, NU))
    u_list2 = np.repeat(u_list1,NU**(JOINT_COUNT-1))
    u_list = u_list2
    for i in range(JOINT_COUNT-1):
        if(i==JOINT_COUNT-2):
            u_list3 = np.tile(u_list1,NU**(JOINT_COUNT-1))            
        else:
            u_list3 = np.repeat(u_list1,NU**(JOINT_COUNT-2-i))
            u_list3 = np.tile(u_list3,NU**(i+1))        
        u_list = np.c_[u_list,u_list3]
    
    simulate_folder2(ITR)
        

    