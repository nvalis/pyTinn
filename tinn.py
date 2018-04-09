# -*- coding: utf-8 -*-

from typing import List
import math
import random
import pickle


class Tinn:

	def __init__(self, nips: int, nhid: int, nops: int):
		"""Build a new t object given number of inputs (nips), number of hidden neurons for the hidden layer (nhid), and number of outputs (nops)."""
		self.nips = nips  # number of inputs
		self.nhid = nhid
		self.nops = nops
		self.b = [random.random() - 0.5 for _ in range(2)]  # biases, Tinn only supports one hidden layer so there are two biases
		self.x1 = [[0] * nips for _ in range(nhid)]  # input to hidden layer
		self.h = [0] * nhid  # hidden layer
		self.x2 = [[random.random() - 0.5 for _ in range(nhid)] for _ in range(nops)]  # hidden to output layer weights
		self.o = [0] * nops  # output layer


def xttrain(t: Tinn, in_: float, tg: float, rate: float) -> float:
	"""Trains a t with an input and target output with a learning rate. Returns error rate of the neural network."""
	fprop(t, in_)
	bprop(t, in_, tg, rate)
	return toterr(tg, t.o)


def xtpredict(t: Tinn, in_: float) -> float:
	"""Returns an output prediction given an input."""
	fprop(t, in_)
	return t.o


def xtsave(t: Tinn, path: str) -> None:
	"""Saves the t to disk."""
	pickle.dump(t, open(path, 'wb'))


def xtload(path: str) -> Tinn:
	"""Loads a new t from disk."""
	return pickle.load(open(path, 'rb'))


def err(a: float, b: float) -> float:
	"""Error function."""
	return 0.5 * (a - b) ** 2


def pderr(a: float, b: float) -> float:
	"""Partial derivative of error function."""
	return a - b


def toterr(tg: List[float], o: List[float]) -> float:
	"""Total error."""
	return sum([err(tg[i], o[i]) for i in range(len(o))])


def act(a: float) -> float:
	"""Activation function."""
	return 1 / (1 + math.exp(-a))


def pdact(a: float) -> float:
	"""Partial derivative of activation function."""
	return a * (1 - a)


def bprop(t: Tinn, in_: List[float], tg: float, rate: float) -> None:
	"""Back propagation."""
	for i in range(t.nhid):
		s = 0
		# Calculate total error change with respect to output.
		for j in range(t.nops):
			a = pderr(t.o[j], tg[j])
			b = pdact(t.o[j])
			s += a * b * t.x2[j][i]
			# Correct weights in hidden to output layer.
			t.x2[j][i] -= rate * a * b * t.h[i]
		# Correct weights in input to hidden layer.
		for j in range(t.nips):
			t.x1[i][j] -= rate * s * pdact(t.h[i]) * in_[j]


def fprop(t: Tinn, in_: float) -> None:
	"""Forward propagation."""
	# Calculate hidden layer neuron values.
	for i in range(t.nhid):
		s = 0
		for j in range(t.nips):
			s += in_[j] * t.x1[i][j]
		t.h[i] = act(s + t.b[0])
	# Calculate output layer neuron values.
	for i in range(t.nops):
		s = 0
		for j in range(t.nhid):
			s += t.h[j] * t.x2[i][j]
		t.o[i] = act(s + t.b[1])
