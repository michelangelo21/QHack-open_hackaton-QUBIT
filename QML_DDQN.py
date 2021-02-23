# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Imports

# %%
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

# %% [markdown]
# ## Environment
# 
# The CartPole environment consists of a pole which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. The state space is represented by four values: cart position, cart velocity, pole angle, and the velocity of the tip of the pole. The action space consists of two actions: moving left or moving right. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center. 
# 
# Source: [https://gym.openai.com/envs/CartPole-v1/](Open AI Gym). 
# 
# The cell below plots a bunch of example frames from the environment.

# %%
env = gym.envs.make("CartPole-v1")


# %%
# Demonstration
env = gym.envs.make("CartPole-v1")


def get_screen():
    ''' Extract one step of the simulation.'''
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    return torch.from_numpy(screen)

# Speify the number of simulation steps
num_steps = 2

# Show several steps
for i in range(num_steps):
    clear_output(wait=True)
    env.reset()
    plt.figure()
    plt.imshow(get_screen().cpu().permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('CartPole-v0 Environment')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

# %% [markdown]
# ## Plotting Function
# 
# This function will make it possible to analyze how the agent learns over time. The resulting plot consists of two subplots. The first one plots the total reward the agent accumulates over time, while the other plot shows a histogram of the agent's total rewards for the last 50 episodes. 

# %%
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

# %% [markdown]
# ## Deep Q Learning
# 
# The main idea behind Q-learning is that we have a function $Q: State \times Action \rightarrow \mathbb{R}$, which can tell the agent what actions will result in what rewards. If we know the value of Q, it is possible to construct a policy that maximizes rewards:
# 
# \begin{align}\pi(s) = \arg\!\max_a \ Q(s, a)\end{align}
# 
# However, in the real world, we don't have access to full information, that's why we need to come up with ways of approximating Q. One traditional method is creating a lookup table where the values of Q are updated after each of the agent's actions. However, this approach is slow and does not scale to large action and state spaces. Since neural networks are universal function approximators, I will train a network that can approximate $Q$.
# 
# The DQL class implementation consists of a simple neural network implemented in PyTorch that has two main methods--predict and update. The network takes the agent's state as an input and returns the Q values for each of the actions. The maximum Q value is selected by the agent to perform the next action.

# %%
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
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))


# %%
def q_learning(env, model, episodes, gamma=0.9, 
               epsilon=0.3, eps_decay=0.99,
               replay=False, replay_size=20, 
               title = 'DQL', double=False, 
               n_update=10, soft=False, verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    memory = []
    episode_i=0
    sum_total_replay_time=0

    _max = [0.0, 0.0, 0.0, 0.0]

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

            for _i in range(len(_max)):
                if np.abs(state[i]) > _max[i]:
                    _max[i] = np.abs(state[i])


            state = next_state
            
        
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title)
        
        if verbose:
            print("episode: {}, total reward: {}".format(episode_i, total))
            if replay:
                print("Average replay time:", sum_total_replay_time/episode_i)
        
    print(_max)
    return final

# %% [markdown]
# ## From QML_DQN_FROZEN_LAKE
# %% [markdown]
# ### ReplayMemory

# %%

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def output_all(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

# %% [markdown]
# ### Plotting Function ##

# %%
## Plotting Function ##
"""
Note: the plotting code is origin from Yang, Chao-Han Huck, et al. "Enhanced Adversarial Strategically-Timed Attacks Against Deep Reinforcement Learning." 
## ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2020.
If you use the code in your research, please cite the original reference. 
"""

def plotTrainingResultCombined(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
    fig, ax = plt.subplots()
    # plt.yscale('log')
    ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
    ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
    leg = ax.legend();

    ax.set(xlabel='Iteration Index', 
           title=_fileTitle)
    fig.savefig(_fileTitle + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")

def plotTrainingResultReward(_iter_index, _iter_reward, _iter_total_steps, _fileTitle):
    fig, ax = plt.subplots()
    # plt.yscale('log')
    ax.plot(_iter_index, _iter_reward, '-b', label='Reward')
    # ax.plot(_iter_index, _iter_total_steps, '-r', label='Total Steps')
    leg = ax.legend();

    ax.set(xlabel='Iteration Index', 
           title=_fileTitle)
    fig.savefig(_fileTitle + "_REWARD" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".png")

# %% [markdown]
# ## PennyLane Part ##
# %% [markdown]
# ### 1 qubit try

# %%
num_wires = 1
dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(dev)
def circuit(weights, x):
    for i in range(3):
        qml.RY(weights[i,0] * x[i] + weights[i,1], wires=0)
    return qml.expval(qml.PauliZ(0))

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l-p)**2
    loss = loss / len(labels)
    return loss

def cost(weights, features, labels):
    predictions = [(circuit(weights, f)) for f in features]
    loss = square_loss(labels, predictions)
    return loss

weights = np.array([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]], dtype=np.float64) # random

np.random.seed(0)
opt = qml.GradientDescentOptimizer(stepsize=0.1)
batch_size = 30

for it in range(20):
    batch_index = np.random.randint(0,len(Y_train), (batch_size,))
    X_batch = X_train[batch_index]
    Y_batch = Y_train[batch_index]
    weights = opt.step(lambda w: cost(w, X_batch, Y_batch), weights)

    #predictions_train = [variational_classifier(var,f) for f in X_train]
    #predictions_val = [variational_classifier(var,f) for f in X_test]

    # acc_train = cost(weights, X_train, Y_train)
    # print(it, acc_train)



predictions = [circuit(weights,f) for f in X_test]

for i in range(len(X_test)):
    if predictions[i] > 0.5:
        predictions[i] = 1
    elif predictions[i] < -0.5:
        predictions[i] = -1
    else:
        predictions[i] = 0

# %%
@qml.qnode(dev)
def circuit(weights, angles):
    for i in range(4):
        qml.RY(weights[i,0] * angles[i] + weights[i,1], wires=0)
    return qml.expval(qml.PauliZ(0))

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l-p)**2
    loss = loss / len(labels)
    return loss

def cost(var_Q_circuit, var_Q_bias, features, labels):
    predictions = [(variational_classifier(var_Q_circuit, var_Q_bias, f)) for f in features]
    loss = square_loss(labels, predictions)
    return loss


def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
    """The variational classifier."""

    # Change to SoftMax???
    weights = var_Q_circuit
    # bias_1 = var_Q_bias[0]
    # bias_2 = var_Q_bias[1]
    # bias_3 = var_Q_bias[2]
    # bias_4 = var_Q_bias[3]
    # bias_5 = var_Q_bias[4]
    # bias_6 = var_Q_bias[5]

    # raw_output = circuit(weights, angles=angles) + np.array([bias_1,bias_2,bias_3,bias_4,bias_5,bias_6])
    # raw_output = circuit(weights, angles=angles) + var_Q_bias
    raw_output = circuit(weights, angles=angles) # TRY WITH NO BIAS
    # We are approximating Q Value
    # Maybe softmax is no need
    # softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

    return raw_output

def epsilon_greedy(var_Q_circuit, var_Q_bias, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """

    # Modify to incorporate with Variational Quantum Classifier
    # epsilon should change along training
    # In the beginning => More Exploration
    # In the end => More Exploitation

    # More Random
    #np.random.seed(int(datetime.now().strftime("%S%f")))

    if train or np.random.rand() < ((epsilon/n_actions)+(1-epsilon)):
        # action = np.argmax(Q[s, :])
        # variational classifier output is torch tensor
        # action = np.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = decimalToBinaryFixLength(9,s)))
        raw_output = variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=s)
        if raw_output > 0:
            action = 1
        else:
            action = 0
        # after circuit() dev.state changes so we can compute entropy
        # if(compute_entropy):
        #     S = entanglement_entropy(dev.state) #dev is global variable
        #     entropies.append(S)
        
    else:
        # need to be torch tensor
        action = np.random.randint(0, n_actions)
    return action



# %%
def deep_Q_Learning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render=False, test=False):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """

    env = gym.envs.make("CartPole-v1")
    n_actions = env.action_space.n
    print("NUMBER OF ACTIONS:" + str(n_actions))
    #print("NUMBER OF ACTIONS:"+str(n_actions))

    num_qubits = 1
    var_init_circuit = np.array([[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0]], dtype=np.float64)
    var_init_bias = np.array([0,0,0,0], dtype=np.float64)

    var_Q_circuit = var_init_circuit
    var_Q_bias = var_init_bias
    var_target_Q_circuit = var_Q_circuit.copy()
    var_target_Q_bias = var_Q_bias.copy()

    opt = qml.NesterovMomentumOptimizer(0.01)

    TARGET_UPDATE = 20
    batch_size = 5

    target_update_counter = 0

    iter_index = []
    iter_reward = []
    iter_total_steps = []

    timestep_reward = []

    memory = ReplayMemory(80)

    for episode in range(episodes):
        print(f"Episode: {episode}")

        s = env.reset()
        a = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s)
        t = 0
        total_reward = 0
        done = False

        while t < max_steps:
            if render:
                env.render()
                #render
            t += 1

            target_update_counter += 1
            s_, reward, done, info = env.step(a)
            print("State : " + str(s_))
            print("Reward : " + str(reward))
            print("Done : " + str(done))

            total_reward += reward

            a_ = epsilon_greedy(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, epsilon = epsilon, n_actions = n_actions, s = s)

            memory.push(s, a, reward, s_, done)

            if len(memory) > batch_size:
                batch_sampled = memory.sample(batch_size = batch_size)
                Q_target = [item.reward + (1 - int(item.done)) * gamma * np.max(variational_classifier(var_Q_circuit = var_target_Q_circuit, var_Q_bias = var_target_Q_bias, angles=item.next_state)) for item in batch_sampled]
                def closure():
                    #opt.zero_grad()
                    loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
                    loss.backward()
                    return loss
                opt.step(closure) ## ERROR HERE, CHANGE TO PYTORCH OPTIM ? 

                current_replay_memory = memory.output_all()

            if target_update_counter > TARGET_UPDATE:
                print("UPDATEING TARGET CIRCUIT...")

                var_target_Q_circuit = var_Q_circuit.clone().detach()
                var_target_Q_bias = var_Q_bias.clone().detach()
                
                target_update_counter = 0

            s, a = s_, a_

            if done:
                if render:
                    print("###FINAL RENDER###")
                    env.render()
                    print("###FINAL RENDER###")
                    print(f"This episode took {t} timesteps and reward: {total_reward}")
                epsilon = epsilon / ((episode/10000) + 1)
                # print("Q Circuit Params:")
                # print(var_Q_circuit)
                print(f"This episode took {t} timesteps and reward: {total_reward}")
                timestep_reward.append(total_reward)
                iter_index.append(episode)
                iter_reward.append(total_reward)
                iter_total_steps.append(t)
                break


                






deep_Q_Learning(1,1,1,10,20,1)


# %%
