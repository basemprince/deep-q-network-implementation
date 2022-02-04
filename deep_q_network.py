import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from random import sample
from enviroment.hybrid_pendulum import hybrid_pendulum
import time
from collections import deque # for FIFO
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.utils import to_categorical
np_config.enable_numpy_behavior()
 

# =============================================================================
# hyper pramters
# =============================================================================
JOINT_COUNT                     = 1
QVALUE_LEARNING_RATE            = 0.001
NEPISODES                       = 5000  # Number of training episodes
NPRINT                          = 500   # print something every NPRINT episodes
MAX_EPISODE_LENGTH              = 100   # Max episode length
LEARNING_RATE                   = 0.8   # alpha coefficient of Q learning algorithm
DISCOUNT                        = 0.9   # Discount factor 
PLOT                            = True  # Plot stuff if True
exploration_prob                = 1     # initial exploration probability of eps-greedy policy
exploration_decreasing_decay    = 0.001 # exploration decay for exponential decreasing
min_exploration_prob            = 0.001 # minimum of exploration proba
SAMPLE_FREQ                     = 4
BUFFER_SIZE                     = 1000  # FIFO BUFFER SIZE for replay buffer
SAMPLE_SIZE                     = 32     # sample size from the buffer
UPDATE_WEIGHTS_FREQ             = 50    # number of steps before updating the Q-target weights
    
# =============================================================================
# Environment
# =============================================================================
nu=11   # number of discretization steps for the joint torque u


def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+nu))
    print(inputs)
    print(inputs.shape)
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(JOINT_COUNT)(state_out4) 
    model = tf.keras.Model(inputs, outputs)

    return model

def update(mini_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    xu_batch ,cost_batch ,xu_next_batch = zip(*mini_batch)
#    x_batch = np.asarray(x_batch).squeeze()
#    u_batch = np.asarray(u_batch)    
#    u_batch_i = np.eye(nu)[u_batch]
#    u_batch_i = to_categorical(u_batch,num_classes=nu)
#    xu_batch = np.concatenate((x_batch,u_batch_i),axis=1)
    
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_batch, training=True)
        print("target_value" , target_values)
        # Compute 1-step targets for the critic loss
        y = cost_batch + DISCOUNT*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))    


if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create critic and target NNs
    env = hybrid_pendulum(JOINT_COUNT, nu, dt=0.1)
    nx = env.nx
#    print(nx)
    Q = get_critic(nx, nu)
    # a separate nn to calculate the target
    Q_target = get_critic(nx, nu)
    Q.summary()
    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())
    ## Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
        
    h_ctg = [] # Learning history (for plot).
    current_step = 0
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    # to transform u to a one hot encode
    u_categorized = to_categorical(list(range(0, nu)))
    for episode in range(NEPISODES):
        x = env.reset()
#       x = x.squeeze()
        ctg = 0.0
        gamma_i = 1
        for step in range(MAX_EPISODE_LENGTH):
            if uniform(0,1) < exploration_prob:
                u = randint(env.nu)
            else:
                x_repeat  = np.repeat([x], nu, axis=0)
                xu_check = np.concatenate((x_repeat,u_categorized),axis=1)
                u = Q.predict(xu_check)
                u_enc = u_categorized[u]
#                print(u)
            u_enc = u_categorized[u].reshape(1,-1)
            x_enc = x.reshape(1,-1)
            xu = np.concatenate((x_enc,u_enc),axis=1)            
            x_next, cost = env.step(u)
            x_next_enc = x_next.reshape(1,-1)
            xu_next = np.concatenate((x_next_enc,u_enc),axis=1)  
            
            replay_buffer.append([xu, cost, xu_next])
            if len(replay_buffer) > SAMPLE_SIZE and current_step%SAMPLE_FREQ == 0:
                mini_batch = sample(replay_buffer,SAMPLE_SIZE) 
                update(mini_batch)
            x = x_next
            ctg += gamma_i*cost
            gamma_i *= DISCOUNT
            # update the Q-target weights less often
            if (current_step%UPDATE_WEIGHTS_FREQ==0):
                Q_target.set_weights(Q.get_weights())
            
        exploration_prob = max(min_exploration_prob, np.exp(-exploration_decreasing_decay*episode))
        h_ctg.append(ctg)

## Save NN weights to file (in HDF5)
#Q.save_weights("/namefile.h5")
#
## Load NN weights from file
#Q.load_weights("namefile.h5")
