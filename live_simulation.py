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
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)
### --- Hyper paramaters
SAMPLING_STEPS         = 4             # Steps to sample from replay buffer
BATCH_SIZE             = 64            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 20000         # Size of replay buffer
MIN_BUFFER_SIZE        = 5000         # Minimum buffer size to start training
UPDATE_Q_TARGET_STEPS  = 100           # Steps to update Q target
NEPISODES              = 10000          # Number of training episodes
MAX_EPISODE_LENGTH     = 200           # Max episode length
QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
MIN_EPSILON            = 0.001         # Minimum of exploration probability
nprint                 = 10
PLOT                   = True
JOINT_COUNT            = 2
NU                     = 13
TRAIN                  = True
THRESHOLD              = 1e-6
ITR                    = 100


def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+JOINT_COUNT))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(JOINT_COUNT)(state_out4)

    model = tf.keras.Model(inputs, outputs)

    return model

#def update(batch):
#    ''' Update the weights of the Q network using the specified batch of data '''

#def update(batch):
#    ''' Update the weights of the Q network using the specified batch of data '''


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

    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    h_ctg = []
    best_ctg = np.inf
    
    steps = 0
    epsilon = EPSILON
    threshold = THRESHOLD
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
        
    directory = glob.glob('Q_weights_backup/*')
    for file in sorted(directory):
        if file.endswith(".h5"):
            print('loading file' , file)
            Q.load_weights(file)
            x= env.reset()
            ctg = 0.0
            gamma_i = 1  
            for i in range(ITR):      
                x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
                xu_check = np.c_[x_rep,u_list]
                pred = Q.predict(xu_check)
                u_ind = np.argmin(pred.sum(axis=1), axis=0)
                u = u_list[u_ind]
                x, cost = env.step(u)
                ctg += gamma_i*cost
                gamma_i *= GAMMA
                env.render()
    