#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:51:38 2022

@author: parallels
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint
from .hybrid_pendulum import Hybrid_Pendulum
import enviroment.config_file as c
from collections import deque

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class Deep_Q():
    ''' class for the deep Q learning neural network. initializes, and keeps track of all
    the variables and parameters needed to excute the model learning'''
    def __init__(self):
        # initialize enviroment
        self.env = Hybrid_Pendulum(c.JOINT_COUNT, c.NU, dt=0.1)
        self.nx = self.env.nx
        self.nv = self.env.nv      
        # initialize keras neural networks and optimizer
        self.Q = self.get_critic("Q")
        self.Q.summary()
        self.Q_target = self.get_critic("Q_target")
        self.Q_target.set_weights(self.Q.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(c.QVALUE_LEARNING_RATE)

        # initialize the replay buffer using deque
        self.replay_buffer = deque(maxlen=c.BUFFER_SIZE)
        # initialize hyper paramters booleans and counters to calculate decay
        self.epsilon = c.EPSILON
        self.threshold_c = c.THRESHOLD_C
        self.threshold_v = c.THRESHOLD_V
        self.count_epsilon = 0
        self.count_thresh = 0
        self.dec_threshold = False              # a flag to signal that the thresholds need to be decreased
        
        self.reached = False                    # a boolean to check that pendulum reached target state
        self.at_target = 0                      # a count of the number of steps the pendulum stayed at target
        
        self.u_list = self.create_u_matrix()    # creating a matrix for descritized controls based on JOINT_COUNT
        self.h_ctg = []                         # keep a list of all cost to go per episode
        self.best_ctg = np.inf   
        self.total_steps = 0                    # keep track of all the steps taken across the episodes

        if c.SAVE_MODEL:
            # Generate a folder with a model number
            self.folder, self.model_num = self.create_folder_backup()
            print("Start training Model #", self.model_num)
        else:
            print("Start training the Model")

    def reset(self):
        '''resets the enviroment and some parameters every start of an episode'''
        self.x = self.env.reset()
        self.dec_threshold = False          # a flag to signal that the thresholds need to be decreased
        self.at_target = 0                  # a count of the number of steps the pendulum stayed at target
        self.ctg = 0.0
        self.gamma_i = 1
    
    def step(self,u): self.x_next , self.cost = self.env.step(u)
    
    def update_q_target(self): self.Q_target.set_weights(self.Q.get_weights())

    def update_params(self):
        '''update some parameters at the end of each step in the episode'''
        self.x = self.x_next
        self.ctg += self.gamma_i * self.cost
        self.gamma_i *= c.DISCOUNT
        self.total_steps += 1
        
    def create_folder_backup(self):
        '''To establish a new folder and saves the hyper parameters requested of the model in PARAM_LIST into a text file'''
        
        MODEL_NUM = str(randint(100000))
        PARAM_LIST = ['SAMPLING_STEPS','MINI_BATCH_SIZE ','BUFFER_SIZE'
                      ,'MIN_BUFFER_SIZE','EPISODE_LENGTH','UPDATE_TARGET_FREQ'
                      ,'QVALUE_LEARNING_RATE','JOINT_COUNT', 'DISCOUNT','NU']
        FOLDER = './Q_weights_backup/model_' + MODEL_NUM + '/'
        while os.path.exists(FOLDER):
            MODEL_NUM = str(randint(100000))
            FOLDER = './Q_weights_backup/model_' + MODEL_NUM + '/'
        os.makedirs(FOLDER)
        f= open(FOLDER + "parameters.txt","w+")
        parameters = list(globals().items())
        for param in parameters:
            if param[0] in PARAM_LIST:
                f.write(param[0] + "=  %f\r\n" % (param[1]))
        f.close()
        return FOLDER , MODEL_NUM

    def save_model(self,episode):
        ''' Exports the weights of Q-function NN to a .h5 file for storing'''
        eps_num = str(episode).zfill(5)
        name = self.folder + "MODEL_"+ self.model_num + '_' + eps_num + ".h5"
        self.Q.save_weights(name)
        
    def create_u_matrix(self):
        '''creates a matrix for the descritized controls based on the JOINT_COUNT
        which will generate all the possible combinations of controls on all joints to feed as input for Q
        * return a matrix of all the possible U controls'''
        u_list1 = np.array(range(0, c.NU))
        u_list2 = np.repeat(u_list1,c.NU**(c.JOINT_COUNT-1))
        u_list = u_list2
        for i in range(c.JOINT_COUNT-1):
            if(i==c.JOINT_COUNT-2):
                u_list3 = np.tile(u_list1,c.NU**(c.JOINT_COUNT-1))            
            else:
                u_list3 = np.repeat(u_list1,c.NU**(c.JOINT_COUNT-2-i))
                u_list3 = np.tile(u_list3,c.NU**(i+1))        
            u_list = np.c_[u_list,u_list3]
        return u_list
    
    def save_to_replay(self, u):
        ''' prepares xu and xu_next and saves the opservation to the replay buffer
        along with the cost'''
        xu = np.c_[self.x.reshape(1,-1),u.reshape(1,-1)]
        xu_next = np.c_[self.x_next.reshape(1,-1),u.reshape(1,-1)]
        self.replay_buffer.append([xu, self.cost, xu_next,self.reached])
              
    def get_critic(self,name):
        ''' Creates the neural network to represent the Q functions 
        * returns a neural network keras model'''
        inputs = layers.Input(shape=(1,self.nx+c.JOINT_COUNT),batch_size=c.MINI_BATCH_SIZE)
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(c.JOINT_COUNT)(state_out4)
    
        model = tf.keras.Model(inputs, outputs,name=name)
    
        return model
    
if __name__=="__main__":
    deep_q = Deep_Q()
    print('Deep Q network activated')