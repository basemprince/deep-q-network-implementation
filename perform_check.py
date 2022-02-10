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
import pandas as pd 
import re


#FOLDER = 'Q_weights_backup/'
FOLDER = 'model_backup/fourth_run/'
#FOLDER = 'Q_weights_backup/tr/'


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)
### --- Hyper paramaters

GAMMA                  = 0.9           # Discount factor 
nprint                 = 10
PLOT                   = True
JOINT_COUNT            = 2
NU                     = 11
SIMULATION_ITR         = 100
INNER_ITR              = 300
THRESHOLD_C            = 9e-1
THRESHOLD_V            = 9e-1
STAY_UP                = 50
RENDER                 = False

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
def reset_env():
    if JOINT_COUNT == 1:
        x0  = np.array([[np.pi], [0.]])
    elif JOINT_COUNT == 2:
        x0  = np.array([[np.pi, 0.], [0., 0.]])
    else:
        x0 = None
    return env.reset(x0) , 0.0 , 1, False

def reset_env_rand():
    return env.reset() , 0.0 , 1, False
    
def simulate_folder(sim_iter,inner_iter,simulate=RENDER):
    best_ctg = np.inf
    best_percent = -np.inf
    h_ctg = []
    model_sucess = []
    model_num_list = []
    best_model = ""    
    directory = glob.glob(FOLDER+'*')
    t_start  = time.time()
    total_output = []
    inner_output = ""
    for file in sorted(directory):
        if file.endswith(".h5"):
            model_number = re.findall(r'\d+',file)[-2]
            model_num_list.append(int(model_number))
            print('loading Model #' , model_number,end='. ')
            percent_suc, ctg = simulate_to_death(file,sim_iter,inner_iter,simulate)
            h_ctg.append(ctg)
            model_sucess.append(percent_suc)
            inner_output = 'acc: '+ str(percent_suc) + 'ctg: ' + str(round(ctg,1))
            total_output.append('loading Model: #' + model_number + ' ' + inner_output+ '\n')
            print(inner_output)         
            if percent_suc>= best_percent:
                best_ctg = ctg
                best_percent = percent_suc
                best_model = model_number
    output = "Best performance: # "+ best_model + " ctg: " + str(round(best_ctg,3)) + " acc: " + str(best_percent) if any(model_sucess) else "None of the models reached target"
    elapsed = "\ntotal time taken: " + str(round((time.time() - t_start)/60.0,1)) + " minutes"
    total_output.append(output)
    total_output.append(elapsed)
    print(output  + elapsed)
    if(PLOT):
        plt.plot(model_num_list,model_sucess)
        plt.xlabel("Episode Number")
        plt.ylabel("% Accuracy")
        plt.title ("Model Success")
        plt.savefig(FOLDER + "model_success.png")
        plt.show()
        
        plt.plot(model_num_list, np.cumsum(h_ctg)/range(1,len(h_ctg)+1)  )
        plt.xlabel("Episode Number")
        plt.title ("Average Cost to Go")
        plt.savefig(FOLDER + "ctg.png")
        plt.show()
        
    f= open(FOLDER + "total_ouput.txt","w+")
    for out in total_output:
        f.write(out)
    f.close()    
    data_frame = pd.DataFrame(model_num_list,columns=['model_number'])
    data_frame['model_success'] = pd.Series(model_sucess, index=data_frame.index)
    data_frame['ctg'] = pd.Series(h_ctg, index=data_frame.index)
    data_frame.to_csv(FOLDER+ 'results.csv')

def simulate_sp(file_name,itr,rend=True):
    rand=True
#    directory = FOLDER + 'Q_weights_'
#    file_name = directory + str(file_num) + '.h5'
#    print('loading file' , file_name)
    Q.load_weights(file_name)
    x , ctg , gamma_i, reached  = reset_env() if not rand else reset_env_rand()
    for i in range(itr):      
        x_rep = np.repeat(x.reshape(1,-1),NU**(JOINT_COUNT),axis=0)
        xu_check = np.c_[x_rep,u_list]
        pred = Q.__call__(xu_check)
        u_ind = np.argmin(tf.math.reduce_sum(pred,axis=1), axis=0)
        u = u_list[u_ind]
        x, cost = env.step(u)
#                print(cost , x[nv:])
        if cost <= THRESHOLD_C and (abs(x[nv:])<= THRESHOLD_V).all() and not reached:
            reached = True
#                    print("sucessfully reached")
        elif cost > THRESHOLD_C or (abs(x[nv:])> THRESHOLD_V).all() :
            reached = False
        ctg += gamma_i*cost
        gamma_i *= GAMMA
        if (rend):
            env.render()
#    print("Model was sucessful:" if reached else "Model failed", "with a cost to go of:",ctg)
    return ctg, reached

def simulate_to_death(file_name,sim_iter,inner_iter,simulate=RENDER):
    sucess = 0
    ctg_h = []
    for i in range(sim_iter):
        ctg, reached = simulate_sp(file_name,inner_iter,simulate)
        ctg_h.append(ctg)
        if reached: sucess+=1
    percent_suc = round(sucess/sim_iter *100.0,1)
    avg_ctg = sum(ctg_h)/len(ctg_h)
#    print("percent sucess:" ,percent_suc, "%" )
    return percent_suc, avg_ctg
    
    

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = HPendulum(JOINT_COUNT, NU, dt=0.1)
    nx = env.nx
    nv = env.nv
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
    
    simulate_folder(SIMULATION_ITR,INNER_ITR)