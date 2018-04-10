#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import tinn


class Data:
	def __init__(self, path, nips, nops):
		self.read_data(path, nips, nops)

	def __repr__(self):
		return f'{len(self)} rows with {len(self.in_[0])} inputs and {len(self.tg[0])} outputs.'

	def read_data(self, path, nips, nops):
		self.in_, self.tg = [], []
		with open(path) as data_file:
			for line in data_file:
				row = list(map(float, line.split()))
				self.in_.append(row[:nips])
				self.tg.append(row[nips:])

	def shuffle(self):
		indexes = list(range(len(self.in_)))
		random.shuffle(indexes)
		self.in_ = [self.in_[i] for i in indexes]
		self.tg = [self.tg[i] for i in indexes]

	def __len__(self):
		return len(self.in_)


def main():
	nips = 256
	nhid = 28
	nops = 10
	rate = 1.0
	anneal = 0.99
	data = Data('semeion.data', nips, nops)
	t = tinn.Tinn(nips, nhid, nops)
	for _ in range(3):
		data.shuffle()
		error = 0
		for in_, tg in zip(data.in_, data.tg):
			error += tinn.xttrain(t, in_, tg, rate)
		print(f'error {error/len(data)} :: learning rate {rate}')
		rate *= anneal

	t.save('saved.tinn')
	loaded = tinn.xtload('saved.tinn')
	in_ = data.in_[0]
	tg = data.tg[0]
	pd = tinn.xtpredict(loaded, in_)
	print(' '.join(map(str, tg)))
	print(' '.join(map(str, pd)))


if __name__ == '__main__':
	main()
