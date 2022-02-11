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
import numpy as np
from numpy.random import uniform
from enviroment import Deep_Q as dq, config_file as c
import time
import matplotlib.pyplot as plt
from random import sample
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)


@tf.function
def update(mini_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    
    # convert all to tensor objects
    xu_batch, cost_batch, xu_next_batch, reached_batch = zip(*mini_batch)
    xu_batch = tf.convert_to_tensor(xu_batch)
    xu_next_batch = tf.convert_to_tensor(xu_next_batch)
    cost_batch = tf.convert_to_tensor(cost_batch)

    with tf.GradientTape() as tape:
        target_values = deep_q.Q_target(xu_next_batch, training=True)
        target_values_per_input = tf.squeeze(tf.math.reduce_sum(target_values,axis=2))
        # Compute 1-step targets for the critic loss
        y = tf.TensorArray(tf.float64, size=c.MINI_BATCH_SIZE, clear_after_read=False)
        for ind, reached_ in enumerate(reached_batch):
            if reached_:
                # apply only cost of current step if at target state
                y = y.write(ind,cost_batch[ind])
            else:
                y = y.write(ind,cost_batch[ind] + c.DISCOUNT*target_values_per_input[ind])
        y= y.stack()             
        # Compute batch of Values associated to the sampled batch of states
        Q_value = deep_q.Q(xu_batch, training=True) 
        Q_value_per_input = tf.squeeze(tf.math.reduce_sum(Q_value,axis=2))
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value_per_input)) 

    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, deep_q.Q.trainable_variables)
    # Update the critic backpropagating the gradients
    deep_q.optimizer.apply_gradients(zip(Q_grad, deep_q.Q.trainable_variables))  
    return True
    
def choose_control():
    '''decides whether the control will be random or choosen by the epsilon greedy method
    * takes the current epsilon and the u list combination
    * returns the chosen control u'''
    if uniform(0,1) < deep_q.epsilon:
        u = tf.random.uniform(shape=[c.JOINT_COUNT],minval=0,maxval=c.NU,dtype=tf.dtypes.int32)
    else:
        x_rep = tf.repeat(deep_q.x.reshape(1,-1),c.NU**(c.JOINT_COUNT),axis=0)
        xu_check = tf.concat([x_rep,deep_q.u_list],axis=1).reshape(c.NU**(c.JOINT_COUNT),1,-1)
        pred = deep_q.Q.__call__(xu_check)
        u_ind = tf.squeeze(tf.math.argmin(tf.math.reduce_sum(pred,axis=2), axis=0))
        u = deep_q.u_list[u_ind]
    return u

def target_check(x, cost):
    '''checks if the state angle and velocity are below threshold, which means
    the pendulum is at target. keeps a count of steps on how long the pendulum held the target
    position after reaching it. if accomplished, it signals for dropping thresholds even further for
    next episodes
    * takes state, at_target count, and the thresholds
    * returns True if state is below threshold and at_target has stayed up for the STAY_UP step count'''
    deep_q.at_target +=1 if cost <= deep_q.threshold_c and (abs(x[deep_q.nv:])<= deep_q.threshold_v).all() else 0                  
    deep_q.reached = True if deep_q.at_target >= c.STAY_UP else False
    if(deep_q.reached):
        print(x , ',', round(cost,5),',',u)
        deep_q.dec_threshold = True

def calculate_decay(minimum, initial, decay_rate, count):
    '''calculates exponitial decay of the value after applying the decay rate to it'''
    return minimum + (initial-minimum) * np.exp(-decay_rate*count)


if __name__=='__main__':
    # Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
     
    deep_q = dq()                                       # initialize model      
    t_start = t = time.time()                           # keep count of episode time and total time
    
    try:
        for episode in range(1,c.NEPISODES+1):
            deep_q.reset()                              #reset enviroment and parameters
            for step in range(c.EPISODE_LENGTH):

                u = choose_control()                    # choose the control based on epsilon

                deep_q.step(u)
                if(c.CHECK_END_STATE): target_check(deep_q.x, deep_q.cost)     # check and update if pendulum reached target state

                deep_q.save_to_replay(u)                                       # save observation to replay buffer

                if deep_q.total_steps % c.UPDATE_TARGET_FREQ == 0:
                    deep_q.update_q_target() 
                    
                # Sampling from replay buffer and train
                if len(deep_q.replay_buffer) >= c.MIN_BUFFER_SIZE and deep_q.total_steps % c.SAMPLING_STEPS == 0:
                    mini_batch = sample(deep_q.replay_buffer, c.MINI_BATCH_SIZE)
                    update(mini_batch)

                deep_q.update_params()
            
            avg_ctg = np.average(deep_q.h_ctg[-c.NPRINT:]) if len(deep_q.h_ctg) > c.NPRINT else deep_q.ctg

            if deep_q.dec_threshold:
                deep_q.count_thresh +=1
                deep_q.threshold_c = deep_q.threshold_v = calculate_decay(c.MIN_THRESHOLD, c.THRESHOLD_C,c.THRESHOLD_DECAY,deep_q.count_thresh)

            # only start finding the best ctg after 2% of the episodes has passed
            if avg_ctg <= deep_q.best_ctg and episode > 0.02*c.NEPISODES:
                print("better ctg found: ", round(avg_ctg,2), " best ctg was: ", round(deep_q.best_ctg,2))
                deep_q.best_ctg = avg_ctg

            # Start decay only when the minimum size of the replay buffer has been reached
            if(len(deep_q.replay_buffer)>=c.MIN_BUFFER_SIZE):
                deep_q.count_epsilon +=1
                deep_q.epsilon = calculate_decay(c.MIN_EPSILON, c.EPSILON,c.EPSILON_DECAY,deep_q.count_epsilon)
   
            deep_q.h_ctg.append(deep_q.ctg)
            
            if(c.PLOT and episode % c.NPRINT == 0):
                plt.plot( np.cumsum(deep_q.h_ctg)/range(1,len(deep_q.h_ctg)+1)  )
                plt.title ("Average cost to go")
                plt.show()       
                
            if c.SAVE_MODEL and episode % c.SAVE_FREQ == 0:
                deep_q.save_model(episode)   
                
            if episode % c.NPRINT == 0:
                dt = time.time() - t
                t = time.time()
                tot_t = t - t_start
                print('Episode: #%d , cost: %.1f , buffer size: %d, epsilon: %.1f, threshold: %.5f, elapsed: %.1f s , tot. time: %.1f m' % (
                      episode, avg_ctg, len(deep_q.replay_buffer), 100*deep_q.epsilon,deep_q.threshold_c, dt, tot_t/60.0))
        
        if(c.PLOT):
            plt.plot( np.cumsum(deep_q.h_ctg)/range(1,len(deep_q.h_ctg)+1)  )
            plt.xlabel("Episode Number")
            plt.title ("Average Cost to Go")
            if(c.SAVE_MODEL):
                plt.savefig(deep_q.folder + "ctg_training.png")
            plt.show()
            
    except KeyboardInterrupt:
        if(c.SAVE_MODEL):
            print('key pressed ...stopping and saving last weights of Q')
            name = deep_q.folder + 'MODEL_'+ deep_q.model_num + '_' + str(episode) + '.h5'
            deep_q.Q.save_weights(name)
            
            plt.plot( np.cumsum(deep_q.h_ctg)/range(1,len(deep_q.h_ctg)+1)  )
            plt.xlabel("Episode Number")
            plt.title ("Average Cost to Go")
            if(c.SAVE_MODEL):
                plt.savefig(deep_q.folder + "ctg_training.png")
        else:
            print('key pressed ...stopping')
            
                
        
        
        
    
    
    