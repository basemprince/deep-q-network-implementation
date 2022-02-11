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
from enviroment.hpendulum_2 import HPendulum
import time
from collections import deque
import matplotlib.pyplot as plt
from random import sample

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)
### --- Hyper paramaters
SAMPLING_STEPS         = 3             # Steps to sample from replay buffer
BATCH_SIZE             = 128            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 100000        # Size of replay buffer
MIN_BUFFER_SIZE        = 5000          # Minimum buffer size to start training
NEPISODES              = 10000         # Number of training episodes
MAX_EPISODE_LENGTH     = 500           # Max episode length
UPDATE_Q_TARGET        = 2500          # Steps to update Q target
QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
MIN_EPSILON            = 0.001         # Minimum of exploration probability
SAVE_MODEL             = 100
nprint                 = 10
PLOT                   = True
JOINT_COUNT            = 2
NU                     = 11
TRAIN                  = True
STAY_UP                = 50
THRESHOLD_Q            = 1e-1
THRESHOLD_V            = 1e-1
THRESHOLD_C            = 1e-1
MIN_THRESHOLD          = 1e-3
THRESHOLD_DECAY        = 0.003

def create_folder_backup():
    MODEL_NUM = str(randint(100000))
    PARAM_LIST = ['SAMPLING_STEPS','BATCH_SIZE','REPLAY_BUFFER_SIZE'
                  ,'MIN_BUFFER_SIZE','MAX_EPISODE_LENGTH','UPDATE_Q_TARGET'
                  ,'QVALUE_LEARNING_RATE']
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
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx,name):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+JOINT_COUNT))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(JOINT_COUNT)(state_out4)

    model = tf.keras.Model(inputs, outputs,name=name)

    return model


def save_model():
    eps_num = str(episode).zfill(5)
    name = FOLDER + "MODEL_"+ MODEL_NUM + '_' + eps_num + ".h5"
    Q.save_weights(name)
#    simulate_sp(eps_num)
    


if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    FOLDER , MODEL_NUM = create_folder_backup()
    print("Start training Model #", MODEL_NUM)
    env = HPendulum(JOINT_COUNT, NU, dt=0.1)
    nx = env.nx
    nv = env.nv
    Q = get_critic(nx,"Q")
    Q.summary()
    Q_target = get_critic(nx,"Q_target")
    Q_target.set_weights(Q.get_weights())

    optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    h_ctg = []
    best_ctg = np.inf
    
    steps = 0
    epsilon = EPSILON
    t_start = t = time.time()
    threshold_c = THRESHOLD_C
    threshold_v = THRESHOLD_V
    # to clear old saved weights
#    directory = glob.glob(FOLDER + '*')
#    for file in sorted(directory):
#        if file.endswith(".h5"): 
#            os.remove(file)

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
        
    if(not TRAIN):
        simulate()
    else:
        try:
            count_eps = 0
            count_thresh = 0
            for episode in range(1,NEPISODES+1):
                ctg = 0.0
                x = env.reset()
                gamma_i = 1
                dec_threshold = False
                at_target = 0
                for step in range(MAX_EPISODE_LENGTH):
                    
                    if uniform(0,1) < epsilon:
                        u = randint(NU, size=JOINT_COUNT)
                    else:
                        x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
                        xu_check = np.c_[x_rep,u_list]
                        pred = Q.__call__(xu_check)
                        u_ind = np.argmin(tf.math.reduce_sum(pred,axis=1), axis=0)
                        u = u_list[u_ind]
                    x_next, cost = env.step(u)
#                    reached = True if cost <= THRESHOLD_C and (abs(x[nv:])<= THRESHOLD_V).all() else False
#                    reached = True if (abs(x[:nv]) <= THRESHOLD_C).all() and (abs(x[nv:])<= THRESHOLD_V).all() else False
                    at_target +=1 if cost <= threshold_c and (abs(x[nv:])<= threshold_v).all() else 0                  
                    reached = True if at_target >= STAY_UP else False
                    if(reached):
    #                    env.render()
                        print(x , cost)
                        dec_threshold = True
                        
                    xu = np.c_[x.reshape(1,-1),u.reshape(1,-1)]
                    xu_next = np.c_[x_next.reshape(1,-1),u.reshape(1,-1)]
                    replay_buffer.append([xu, cost, xu_next,reached])
                    
                    if steps % UPDATE_Q_TARGET == 0:
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
                            Q_value = Q(xu_batch, training=True) 
                            Q_value_per_input = tf.math.reduce_sum(Q_value,axis=1)
                            # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
                            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value_per_input)) 
                        Q_grad = tape.gradient(Q_loss, Q.trainable_variables)
                        optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))
                                    
                    x = x_next
                    ctg += gamma_i * cost
                    gamma_i *= GAMMA
                    steps += 1  
                avg_ctg = np.average(h_ctg[-nprint:])
                
                if dec_threshold:
                    count_thresh +=1
                    threshold_c = threshold_v = MIN_THRESHOLD + (THRESHOLD_C-MIN_THRESHOLD)* np.exp(-EPSILON_DECAY*count_thresh)

                if avg_ctg <= best_ctg and episode > 0.02*NEPISODES:
                    print("cost is: ", avg_ctg, " best_ctg was: ", best_ctg)
                    best_ctg = avg_ctg

                if(len(replay_buffer)>=MIN_BUFFER_SIZE):
                    count_eps +=1
                    epsilon = max(MIN_EPSILON, np.exp(-EPSILON_DECAY*count_eps))
   
                h_ctg.append(ctg)
                
                if(PLOT and episode % nprint == 0):
                    plt.plot( np.cumsum(h_ctg)/range(1,len(h_ctg)+1)  )
                    plt.title ("Average cost to go")
                    plt.show()       
                    
                if episode % SAVE_MODEL == 0:
                    save_model()   
                    
                if episode % nprint == 0:
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
            name = FOLDER + 'MODEL_'+ MODEL_NUM + '_' + 'final.h5'
            Q.save_weights(name)
            
                
        
        
        
    
    
    