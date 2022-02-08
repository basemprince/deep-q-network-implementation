#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:35:39 2022

@author: parallels
"""
import sys
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from enviroment.hpendulum_2 import HPendulum
import time
from collections import deque
import matplotlib.pyplot as plt
from random import sample

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)
### --- Hyper paramaters
SAMPLING_STEPS         = 4             # Steps to sample from replay buffer
BATCH_SIZE             = 64            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 20000         # Size of replay buffer
MIN_BUFFER_SIZE        = 20000         # Minimum buffer size to start training
UPDATE_Q_TARGET_STEPS  = 100           # Steps to update Q target
NEPISODES              = 7000          # Number of training episodes
MAX_EPISODE_LENGTH     = 200           # Max episode length
QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
MIN_EPSILON            = 0.001         # Minimum of exploration probability
nprint                 = 10
PLOT                   = True
JOINT_COUNT            = 2
NU                     = 11
TRAIN                  = False
THRESHOLD              = 1e-6

def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx,nu):
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


def simulate(itr=100,curr=True):
    ## Load NN weights from file
    if not curr:
        Q.load_weights("Q_weights.h5")
    x= env.reset()
    ctg = 0.0
    gamma_i = 1
    
    for i in range(itr):      
        x_rep = np.repeat(x.reshape(1,-1),nu**(JOINT_COUNT),axis=0)
        xu_check = np.c_[x_rep,u_list]
        pred = Q.predict(xu_check)
        u_ind = np.argmin(pred.sum(axis=1), axis=0)
        u = u_list[u_ind]
        x, cost = env.step(u)
        ctg += gamma_i*cost
        gamma_i *= GAMMA
        env.render() 

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    nu = 11

    env = HPendulum(JOINT_COUNT, nu, dt=0.1)
    nx = env.nx
    Q = get_critic(nx,nu)
    Q.summary()
    Q_target = get_critic(nx,nu)
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
    u_list1 = np.array(range(0, nu))
    u_list2 = np.repeat(u_list1,nu**(JOINT_COUNT-1))
    u_list = u_list2
    for i in range(JOINT_COUNT-1):
        print(i)
        if(i==JOINT_COUNT-2):
            u_list3 = np.tile(u_list1,nu**(JOINT_COUNT-1))            
        else:
            u_list3 = np.repeat(u_list1,nu**(JOINT_COUNT-2-i))
            u_list3 = np.tile(u_list3,nu**(i+1))        
        u_list = np.c_[u_list,u_list3]
        
    if(not TRAIN):
        simulate(curr=False)
    else:
        count = 0
        for episode in range(NEPISODES):
            cost_to_go = 0.0
            x = env.reset()
            gamma_i = 1
            for step in range(MAX_EPISODE_LENGTH):
                
                if uniform(0,1) < epsilon:
                    u = randint(nu, size=JOINT_COUNT)
                else:
                    x_rep = np.repeat(x.reshape(1,-1),nu**(JOINT_COUNT),axis=0)
                    xu_check = np.c_[x_rep,u_list]
                    pred = Q.predict(xu_check)
                    u_ind = np.argmin(pred.sum(axis=1), axis=0)
                    u = u_list[u_ind]
                x_next, cost = env.step(u)
                if(cost <= threshold):
#                    env.render()
                    print(cost)
#                threshold = max(MIN_EPSILON*1e-1, np.exp(-EPSILON_DECAY*1e-1*episode)*1e-1)
                reached = True if cost <=threshold else False
                xu = np.c_[x.reshape(1,-1),u.reshape(1,-1)]
                xu_next = np.c_[x_next.reshape(1,-1),u.reshape(1,-1)]
                replay_buffer.append([xu, cost, xu_next,reached])
                
                if steps % UPDATE_Q_TARGET_STEPS == 0:
                    Q_target.set_weights(Q.get_weights()) 
                # Sampling from replay buffer and train
                if len(replay_buffer) >= MIN_BUFFER_SIZE and steps % SAMPLING_STEPS == 0:
                    xu_batch, cost_batch, xu_next_batch, reached_batch = zip(*sample(replay_buffer, BATCH_SIZE))
                    xu_batch = tf.convert_to_tensor(np.asarray(xu_batch).squeeze())
                    xu_next_batch = tf.convert_to_tensor(np.asarray(xu_next_batch).squeeze())
                    cost_batch1 = np2tf(np.asarray(cost_batch))
                    with tf.GradientTape() as tape:
                        # Compute Q target
                        target_values = Q_target(xu_next_batch, training=True)
                        target_values_per_input = tf.math.reduce_sum(target_values,axis=1)
                        # Compute 1-step targets for the critic loss
                        y = np.zeros(BATCH_SIZE)
                        for ind, reached_ in enumerate(reached_batch):
                            if reached_:
                                y[ind] = cost_batch1[ind]
                            else:
                                y[ind] = cost_batch1[ind] + GAMMA*target_values_per_input[ind]    
                      
                        # Compute batch of Values associated to the sampled batch of states
                        # Compute batch of Values associated to the sampled batch of states
                        Q_value = Q(xu_batch, training=True) 
                        Q_value_per_input = tf.math.reduce_sum(Q_value,axis=1)
                        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
                        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value_per_input)) 
                    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)
                    optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))
                                
                x = x_next
                cost_to_go += gamma_i * cost
                gamma_i *= GAMMA
                steps += 1
    
                if cost_to_go <= best_ctg and episode > 0.05*NEPISODES:
                    Q.save_weights("Q_weights.h5")
                    best_ctg = cost_to_go
            
            if(len(replay_buffer)==MIN_BUFFER_SIZE):
                count +=1
                epsilon = max(MIN_EPSILON, np.exp(-EPSILON_DECAY*count))
            h_ctg.append(cost_to_go)
            
            if(PLOT and episode % nprint == 0):
                plt.plot( np.cumsum(h_ctg)/range(1,len(h_ctg)+1)  )
                plt.title ("Average cost-to-go")
                plt.show()         
                
            if episode % nprint == 0:
                dt = time.time() - t
                t = time.time()
                tot_t = t - t_start
                print('Episode: #%d , cost: %.1f , buffer size: %d, epsilon: %.1f , threshold: %.4f , elapsed: %.1f s , tot. time: %.1f m' % (
                      episode, np.mean(h_ctg[-nprint:]), len(replay_buffer), 100*epsilon, threshold, dt, tot_t/60.0))
        
    
    
    
    
    
    