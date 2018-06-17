import numpy as np
def random_mini_batches(X, Y, m, mini_batches, seed):

	batches = []
	np.random.seed(seed)
	num_mini_batches = int(m/mini_batches)
	permutation = list(np.random.permutation(m))
	X_shuffled = []
	for i in range(len(X)):
		X_shuffled.append(X[i][permutation])
	Y_shuffled = Y[permutation, :, :, :]

	for k in range(num_mini_batches):
		mini_batches_X = []
		for i in range(len(X)):
			mini_batches_X.append(X_shuffled[i][k*mini_batches:(k+1)*mini_batches])
		mini_batches_Y = Y_shuffled[k*mini_batches:(k+1)*mini_batches, :, :, :]
		batch = (mini_batches_X, mini_batches_Y)
		batches.append(batch)

	if m % mini_batches != 0:
		mini_batches_X = []
		for i in range(len(X)):
			mini_batches_X.append(X_shuffled[i][num_mini_batches*mini_batches:])
		mini_batches_Y = Y_shuffled[num_mini_batches*mini_batches:, :, :, :]
		batch = (mini_batches_X, mini_batches_Y)
		batches.append(batch)

	return batches
