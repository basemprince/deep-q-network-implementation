#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 22:35:39 2022

@author: parallels
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from enviroment.hybrid_pendulum import Hybrid_Pendulum
import time
from collections import deque
import matplotlib.pyplot as plt
from random import sample

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)

# =============================================================================
# Hyper paramaters
# =============================================================================
JOINT_COUNT            = 2             # Number of joints of the pendulum
NEPISODES              = 10000         # Number of training episodes
EPISODE_LENGTH         = 500           # The number of steps per episode
BUFFER_SIZE            = 100000        # The size of the replay buffer
MIN_BUFFER_SIZE        = 5000          # The minimum size for the replay buffer to start training
MINI_BATCH_SIZE        = 64            # The batch size taken randomly from buffer
SAMPLING_STEPS         = 4             # The frequncy of sampling
UPDATE_TARGET_FREQ     = 2500          # The frequncy for updating the Q-target weights
QVALUE_LEARNING_RATE   = 0.001         # The learning rate of DQN
DISCOUNT               = 0.9           # (GAMMA) Discount factor 
EPSILON                = 1             # Initial probability of epsilon
EPSILON_DECAY          = 0.001         # The exponintial decay rate for epsilon
MIN_EPSILON            = 0.001         # The Minimum for epsilon
SAVE_MODEL             = 100           # The number of steps before saving a model
NPRINT                 = 10            # Frequncy of print out
NU                     = 11            # number of discretized controls
STAY_UP                = 50            # How many steps the bendulum needs to hold standup position before considered success
THRESHOLD_Q            = 1e-1          # Threshold for angle
THRESHOLD_V            = 1e-1          # Threshold for velocity
THRESHOLD_C            = 1e-1          # Threshold for cost
MIN_THRESHOLD          = 1e-3          # Minimum value for threshold
THRESHOLD_DECAY        = 0.003         # Decay rate for threshold
PLOT                   = True          # Plot out results
# =============================================================================
# Hyper paramaters
# =============================================================================


def create_folder_backup():
    '''To establish a new folder and saves the hyper parameters requested of the model in PARAM_LIST into a text file'''
    
    MODEL_NUM = str(randint(100000))
    PARAM_LIST = ['SAMPLING_STEPS','MINI_BATCH_SIZE ','BUFFER_SIZE'
                  ,'MIN_BUFFER_SIZE','EPISODE_LENGTH','UPDATE_TARGET_FREQ'
                  ,'QVALUE_LEARNING_RATE','JOINT_COUNT']
    FOLDER = 'Q_weights_backup/model_' + MODEL_NUM + '/'
    while os.path.exists(FOLDER):
        MODEL_NUM = str(randint(100000))
        FOLDER = 'Q_weights_backup/model_' + MODEL_NUM + '/'
    os.makedirs(FOLDER)
    f= open(FOLDER + "parameters.txt","w+")
    parameters = list(globals().items())
    for param in parameters:
        if param[0] in PARAM_LIST:
            f.write(param[0] + "=  %f\r\n" % (param[1]))
    f.close()
    return FOLDER , MODEL_NUM

def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def get_critic(nx,name):
    ''' Creates the neural network to represent the Q functions 
    * returns a neural network keras model'''
    inputs = layers.Input(shape=(nx+JOINT_COUNT))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(JOINT_COUNT)(state_out4)

    model = tf.keras.Model(inputs, outputs,name=name)

    return model

@tf.function
def update(mini_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    
    xu_batch, cost_batch, xu_next_batch, reached_batch = zip(*mini_batch)
    
    # convert to tensor objects
    xu_batch = tf.convert_to_tensor(np.asarray(xu_batch).squeeze())
    xu_next_batch = tf.convert_to_tensor(np.asarray(xu_next_batch).squeeze())
    cost_batch = np2tf(np.asarray(cost_batch))
    
    with tf.GradientTape() as tape:
       
        target_values = Q_target(xu_next_batch, training=True)
        target_values_per_input = tf.math.reduce_sum(target_values,axis=1)
        
        # Compute 1-step targets for the critic loss
        y = np.zeros(MINI_BATCH_SIZE )
        for ind, reached_ in enumerate(reached_batch):
            if reached_:
                # reduce cost if reached target
                y[ind] = cost_batch[ind]
            else:
                y[ind] = cost_batch[ind] + DISCOUNT*target_values_per_input[ind]    
      
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True) 
        Q_value_per_input = tf.math.reduce_sum(Q_value,axis=1)
        
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value_per_input)) 
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)
    # Update the critic backpropagating the gradients
    optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))  
    
    
def create_u_matrix():
    '''creates a matrix for the descritized controls based on the JOINT_COUNT
    which will generate all the possible combinations of controls on all joints to feed as input for Q
    * return a matrix of all the possible U controls'''
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
    return u_list

def choose_control(epsilon,u_list):
    '''decides whether the control will be random or choosen by the epsilon greedy method
    * takes the current epsilon and the u list combination
    * returns the chosen control u'''
    if uniform(0,1) < epsilon:
        u = randint(NU, size=JOINT_COUNT)
    else:
        x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
        xu_check = np.c_[x_rep,u_list]
        pred = Q.__call__(xu_check)
        u_ind = np.argmin(tf.math.reduce_sum(pred,axis=1), axis=0)
        u = u_list[u_ind]
    return u

def target_check(x, at_target, nv,threshold_v, threshold_c):
    '''checks if the state angle and velocity are below threshold, which means
    the pendulum is at target. keeps a count of steps on how long the pendulum held the target
    position after reaching it. if accomplished, it signals for dropping thresholds even further for
    next episodes
    * takes state, at_target count, and the thresholds
    * returns True if state is below threshold and at_target has stayed up for the STAY_UP step count'''
    
    at_target +=1 if cost <= threshold_c and (abs(x[nv:])<= threshold_v).all() else 0                  
    reached = True if at_target >= STAY_UP else False
    if(reached):
#        print(x , ',', round(cost,5),',',u)
        dec_threshold = True
    return reached, at_target, dec_threshold

def calculate_decay(minimum, initial, decay_rate, count):
    '''calculates exponitial decay of the value after applying the decay rate to it'''
    return minimum + (initial-minimum) * np.exp(-decay_rate*count)

def save_model():
    ''' Exports the weights of Q-function NN to a .h5 file for storing'''
    eps_num = str(episode).zfill(5)
    name = FOLDER + "MODEL_"+ MODEL_NUM + '_' + eps_num + ".h5"
    Q.save_weights(name)

if __name__=='__main__':
    # Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Generate a folder with a model number
    FOLDER , MODEL_NUM = create_folder_backup()
    print("Start training Model #", MODEL_NUM)
    
    # initialize enviroment
    env = Hybrid_Pendulum(JOINT_COUNT, NU, dt=0.1)
    nx = env.nx
    nv = env.nv
    
    # initialize keras neural networks
    Q = get_critic(nx,"Q")
    Q_target = get_critic(nx,"Q_target")
    Q_target.set_weights(Q.get_weights())
    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
    
    # initialize the replay buffer using deque
    replay_buffer = deque(maxlen=BUFFER_SIZE)  
    
    u_list = create_u_matrix() # creating a matrix for descritized controls based on JOINT_COUNT
    h_ctg = []
    best_ctg = np.inf   
    total_steps = 0
    count_eps = 0
    count_thresh = 0
    t_start = t = time.time()
    
    # initialize hyper paramters to calculate decay
    epsilon = EPSILON
    threshold_c = THRESHOLD_C
    threshold_v = THRESHOLD_V

    try:
        for episode in range(1,NEPISODES+1):
            ctg = 0.0
            x = env.reset()
            gamma_i = 1
            dec_threshold = False                   # a flag to signal that the thresholds need to be decreased
            at_target = 0                           # a count of the number of steps the pendulum stayed at target
            for step in range(EPISODE_LENGTH):
                
                u = choose_control(epsilon,u_list)  # choose the control based on epsilon
                x_next, cost = env.step(u)

                reached, at_target, dec_threshold = target_check(x, at_target, nv,threshold_v, threshold_c) # check if pendulum reached target

                xu = np.c_[x.reshape(1,-1),u.reshape(1,-1)]
                xu_next = np.c_[x_next.reshape(1,-1),u.reshape(1,-1)]
                replay_buffer.append([xu, cost, xu_next,reached])
                
                if total_steps % UPDATE_TARGET_FREQ == 0:
                    Q_target.set_weights(Q.get_weights()) 
                    
                # Sampling from replay buffer and train
                if len(replay_buffer) >= MIN_BUFFER_SIZE and total_steps % SAMPLING_STEPS == 0:
                    mini_batch = sample(replay_buffer, MINI_BATCH_SIZE)
                    update(mini_batch)
                                
                x = x_next
                ctg += gamma_i * cost
                gamma_i *= DISCOUNT
                total_steps += 1
                
            avg_ctg = np.average(h_ctg[-NPRINT:])
            
            if dec_threshold:
                count_thresh +=1
                threshold_c = threshold_v = calculate_decay(MIN_THRESHOLD, THRESHOLD_C,THRESHOLD_DECAY,count_thresh)

            # only start finding the best ctg after 2% of the episodes has passed
            if avg_ctg <= best_ctg and episode > 0.02*NEPISODES:
                print("cost is: ", avg_ctg, " best_ctg was: ", best_ctg)
                best_ctg = avg_ctg

            # Start decay only when the minimum size of the replay buffer has been reached
            if(len(replay_buffer)>=MIN_BUFFER_SIZE):
                count_eps +=1
                epsilon = calculate_decay(MIN_EPSILON, EPSILON,EPSILON_DECAY,count_eps)
   
            h_ctg.append(ctg)
            
            if(PLOT and episode % NPRINT == 0):
                plt.plot( np.cumsum(h_ctg)/range(1,len(h_ctg)+1)  )
                plt.title ("Average cost to go")
                plt.show()       
                
            if episode % SAVE_MODEL == 0:
                save_model()   
                
            if episode % NPRINT == 0:
                dt = time.time() - t
                t = time.time()
                tot_t = t - t_start
                print('Episode: #%d , cost: %.1f , buffer size: %d, epsilon: %.1f, threshold: %.5f, elapsed: %.1f s , tot. time: %.1f m' % (
                      episode, avg_ctg, len(replay_buffer), 100*epsilon,threshold_c, dt, tot_t/60.0))
        
        if(PLOT):
            plt.plot( np.cumsum(h_ctg)/range(1,len(h_ctg)+1)  )
            plt.xlabel("Episode Number")
            plt.title ("Average Cost to Go")
            plt.savefig(FOLDER + "ctg_training.png")
            plt.show()
            
    except KeyboardInterrupt:
        print('key pressed ...stopping and saving last weights of Q')
        name = FOLDER + 'MODEL_'+ MODEL_NUM + '_' + str(episode) + '.h5'
        Q.save_weights(name)
            
                
        
        
        
    
    
    