#!/usr/bin/env python
# coding: utf-8

# ## 1-qubit try 2nd

# In[1]:



import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn 
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

from datetime import datetime
import pickle

import gym
import time
import random
from collections import namedtuple
from copy import deepcopy


# In[153]:


dev = qml.device('default.qubit', wires=0)
env = gym.envs.make('CartPole-v1')
dtype = torch.DoubleTensor


# ### Variational circuit used for training

# In[151]:


@qml.qnode(dev, interface='torch')
def circuit(state):
    for i in range(3):
        qml.RY(state[0] + state[1] + state[2]+state[3], wires=0)
    return [qml.expval(qml.PauliZ(ind)) for ind in range(2)]


# ## 4 qubit try

# In[157]:


qubit4 = qml.device('default.qubit', wires=4)

def statepreparation(a):
    a[0] /= 4.8
    a[1] /= 2*4.2
    a[2] /= 0.418
    a[3] /= 2*3.1
    for ind in range(len(a)):
		#qml.RX(np.pi * a[ind], wires=ind)
		#qml.RZ(np.pi * a[ind], wires=ind)
        qml.RY(np.pi * a[ind], wires=ind)
        
def layer(W):
	""" Single layer of the variational classifier.

	Args:
		W (array[float]): 2-d array of variables for one layer
	"""

	qml.CNOT(wires=[0, 1])
	qml.CNOT(wires=[1, 2])
	qml.CNOT(wires=[2, 3])


	qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
	qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
	qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
	qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)


# In[162]:


@qml.qnode(dev, interface='torch')
def circuit(weights, states=None):
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	statepreparation(states)
	
	for W in weights:
		layer(W)

	return [qml.expval(qml.PauliZ(ind)) for ind in range(2)]


# ### Cost functions

# In[90]:


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l-p)**2
    loss = loss / len(labels)
    return loss

def cost(params, features, labels):
    predictions = [(circuit(weights, f)) for f in features]
    loss = square_loss(labels, predictions)
    return loss


# In[170]:


class QDQL():
    def __init__(self, state_dim, action_dim, learning_rate=0.05):
        # params and actions len must be equal
        num_qubits = 4
        num_layers = 2
        params = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3)))
        self.weights=params
        self.model = circuit
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([params])
    
    def update(self, state, y):
        print(f'st:{state}')
        print(f'y:{y}')
        y_pred = self.model(torch.Tensor(y, state))
        loss = self.criterion(y_pred.float(), Variable(torch.Tensor(y), requires_grad=True).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(self.weights, state))
    


# In[164]:


def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20, 
               title = 'DQL', double=False, 
               n_update=10, soft=False):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    for episode in range(episodes):
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = int(torch.argmax(q_values).item())

            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)
            
            # Update total and memory
     q       total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()
            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                    break
                if replay:
                    # Update network weights using replay memory
                    model.replay(memory, replay_size, gamma)
            else: 
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(state, q_values)
                state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)
    return final


# In[171]:


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_learning(env, QDQL(state_dim, action_dim), episodes=300)


# In[115]:


class DQN():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)



    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        print(f'st:{state}')
        print(f'y:{y}')
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))


# In[144]:


def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=False):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    episode_i=0
    sum_total_replay_time=0
    for episode in range(episodes):
        episode_i+=1
        if double and not soft:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()
        if double and soft:
            model.target_update()
        
        # Reset state
        state = env.reset()
        done = False
        total = 0
        
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()
            
            # Take action and add reward to total
            next_state, reward, done, _ = env.step(action)
            
            # Update total and memory
            total += reward
            memory.append((state, action, next_state, reward, done))
            q_values = model.predict(state).tolist()
             
            if done:
                if not replay:
                    q_values[action] = reward
                    # Update network weights
                    model.update(state, q_values)
                break

            if replay:
                t0=time.time()
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
                t1=time.time()
                sum_total_replay_time+=(t1-t0)
            else: 
                # Update network weights using the last step only
                q_values_next = model.predict(next_state)
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(state, q_values)

            state = next_state
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)
        
    return final


# In[96]:


# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n
# Number of episodes
episodes = 150
# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001


# In[98]:


def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


# In[145]:


# Get DQN results
simple_dqn = DQN(n_state, n_action, n_hidden, lr)
simple = q_learning(env, simple_dqn, episodes, gamma=.9, epsilon=0.3)

