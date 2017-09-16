from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 
import numpy as np
import gym                               # To train our network
import random                            # For sampling batches from the observations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'   # To supress Tensorflow warnings
env = gym.make('LunarLander-v2')         # Choose game (any in the gym should work)

# Create network. 
# Input is two consecutive game states, output is Q-values of the possible moves.
model = Sequential()
model.add(Dense(20, input_shape=(2, 8), activation="relu", kernel_initializer="uniform"))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(18, activation="relu", kernel_initializer="uniform"))
model.add(Dense(10, activation="relu", kernel_initializer="uniform"))
model.add(Dense(env.action_space.n, activation='linear', kernel_initializer="uniform"))    # Same number of outputs as possible actions
print('Compiling model...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model Compiled Successfully!')

# Parameters
D = deque()            # Register where the actions will be stored

observetime = 20000    # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7          # Probability of doing a random move
gamma = 0.9            # Discounted future reward. How much we care about steps further in time
mb_size = 50           # Learning minibatch size

# FIRST STEP: Knowing what each action does (Observing)
for t in range(observetime):
    observation = env.reset()                     # Game begins
    env.render()                    # Uncomment to see game running
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
    state = np.stack((obs, obs), axis=1)
    done = False
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, env.action_space.n, size=1)[0]
    else:
        Q = model.predict(state)          # Q-values predictions
        action = np.argmax(Q)             # Move with highest Q-value is the chosen one
    observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
    obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
    state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
    D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
    state = state_new         # Update state
    if done:
        env.reset()           # Restart game if it's finished
        obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
        state = np.stack((obs, obs), axis=1)
    string = 'observing {} / {} of observe_time'.format(t, observetime)
    print(string, end='')
    print('\b'*len(string), end='', flush=True)
print('Observing Finished')

# SECOND STEP: Learning from the observations (Experience replay)
minibatch = random.sample(D, mb_size)                              # Sample some moves
inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, env.action_space.n))
epochs = 50
for e in range(epochs):
    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)

    # Train network to output the Q function
        model.train_on_batch(inputs, targets)
    string = 'Training on {} / {} epoch'.format(e, epochs)
    print(string, end='')
    print('\b'*len(string), end='', flush=True)
print('Learning Finished')

# THIRD STEP: Play!
for t in range(10):
    observation = env.reset()
    obs = np.expand_dims(observation, axis=0)
    state = np.stack((obs, obs), axis=1)
    done = False
    tot_reward = 0.0
    while not done:
        env.render()                    # Uncomment to see game running
        Q = model.predict(state)        
        action = np.argmax(Q)         
        observation, reward, done, info = env.step(action)
        obs = np.expand_dims(observation, axis=0)
        state = np.append(np.expand_dims(obs, axis=0), state[:, :1, :], axis=1)    
        tot_reward += reward
print('Game ended! Total reward: {}'.format(reward))







