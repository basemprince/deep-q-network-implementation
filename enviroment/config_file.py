#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:54:46 2022

@author: parallels
"""

# =============================================================================
# Hyper paramaters
# =============================================================================
JOINT_COUNT            = 2             # Number of joints of the pendulum
NEPISODES              = 10000         # Number of training episodes
EPISODE_LENGTH         = 500           # The number of steps per episode
BUFFER_SIZE            = 100000        # The size of the replay buffer
MIN_BUFFER_SIZE        = 5000          # The minimum size for the replay buffer to start training
MINI_BATCH_SIZE        = 64            # The batch size taken randomly from buffer
SAMPLING_STEPS         = 3             # The frequncy of sampling
UPDATE_TARGET_FREQ     = 2500          # The frequncy for updating the Q-target weights
QVALUE_LEARNING_RATE   = 0.001         # The learning rate of DQN
DISCOUNT               = 0.9           # (GAMMA) Discount factor 
EPSILON                = 1             # Initial probability of epsilon
EPSILON_DECAY          = 0.001         # The exponintial decay rate for epsilon
MIN_EPSILON            = 0.001         # The Minimum for epsilon
SAVE_FREQ              = 100           # The number of steps before saving a model
NPRINT                 = 10            # Frequncy of print out
NU                     = 11            # number of discretized controls
STAY_UP                = 50            # How many steps the bendulum needs to hold standup position before considered success
THRESHOLD_Q            = 1e-1          # Threshold for angle
THRESHOLD_V            = 1e-1          # Threshold for velocity
THRESHOLD_C            = 1e-1          # Threshold for cost
MIN_THRESHOLD          = 1e-3          # Minimum value for threshold
THRESHOLD_DECAY        = 0.003         # Decay rate for threshold
PLOT                   = True          # Plot out results
SAVE_MODEL             = True          # Save the models in files
CHECK_END_STATE        = False         # Check if target state is reached
# =============================================================================
# Hyper paramaters
# =============================================================================

         