import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import torch
import torch.nn as nn 
from torch.autograd import Variable

import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import gym
import time
import random
from collections import namedtuple
from copy import deepcopy

import remote_cirq

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
####



env = gym.envs.make("CartPole-v1")

dtype = torch.DoubleTensor

def square_loss(labels, predictions):
    """ Square loss function

    Args:
        labels (array[float]): 1-d array of labels
        predictions (array[float]): 1-d array of predictions
    Returns:
        float: square loss
    """
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev, interface='torch')
def circuit(weights, angles=None):
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	qml.Hadamard(wires=0)
	for i in range(4):
		qml.RY(weights[i,0] * angles[i], wires=0)
		qml.RY(weights[i,1], wires=0)

	#return [qml.expval(qml.PauliZ(ind)) for ind in range(2)]
	return qml.expval(qml.PauliZ(0))


def variational_classifier(var_Q_circuit, angles=None):
	"""The variational classifier."""

	# Change to SoftMax???

	weights = var_Q_circuit
	raw_output = circuit(weights, angles=angles)
	# We are approximating Q Value
	# Maybe softmax is no need
	# softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()
	chance_0 = (raw_output + 1)/2
	output = torch.tensor([chance_0, 1-chance_0])
	return output


def cost(var_Q_circuit, features, labels):
	"""Cost (error) function to be minimized."""

	# predictions = [variational_classifier(weights, angles=f) for f in features]
	# Torch data type??
	
	predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, angles=item.state)[item.action] for item in features]
	# predictions = torch.tensor(predictions,requires_grad=True)
	# labels = torch.tensor(labels)
	# print("PRIDICTIONS:")
	# print(predictions)
	# print("LABELS:")
	# print(labels)

	return square_loss(labels, predictions)


def deep_Q_Learning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render=False, test=False):
	env = gym.envs.make("CartPole-v1")

	memory = ReplayMemory(80)

	n_actions = env.action_space.n

	var_init_circuit = Variable(torch.tensor([[1,0],[1,0],[1,0],[1,0]]).type(dtype), requires_grad=True)
	var_Q_circuit = var_init_circuit

	opt = torch.optim.RMSprop([var_Q_circuit], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

	episode_i = 0
	for episode in range(episodes):
		episode_i += 1

		s = env.reset()
		done = False
		total_reward = 0

		while not done:
			if random.random() < epsilon:
				a = env.action_space.sample()
			else:
				q_values = variational_classifier(var_Q_circuit, angles=s)
				a = torch.argmax(q_values).item()
			
			s_, reward, done, info = env.step(a)

			total_reward += reward
			memory.push(s, a, s_, reward, done)

			q_values = variational_classifier(var_Q_circuit, s)

			if done:
				q_values[a] = reward
				#update
				y_pred = variational_classifier(var_Q_circuit, s)
				loss = square_loss(y_pred, q_values)
				opt.zero_grad()
				loss.backward()
				opt.step()
				break

			q_values_next = variational_classifier(var_Q_circuit, s_)
			q_values[a] = reward + gamma * torch.max(q_values_next).item()

			#update
			def closure():
				opt.zero_grad()
				y_pred = variational_classifier(var_Q_circuit, s)
				loss = square_loss(y_pred,Variable(torch.tensor(q_values)))
				# print(loss)
				#opt.zero_grad()
				loss.backward()
				return loss
			opt.step(closure)
			
			opt.step()
			
			s = s_


if __name__ =="__main__":
	alpha = 0.4
	gamma = 0.999
	epsilon = 1.
	episodes = 500
	max_steps = 2500
	n_tests = 2
	deep_Q_Learning(alpha, gamma, epsilon, episodes, max_steps, n_tests, test = False)

