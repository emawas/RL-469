To load data:
import numpy as np
train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)

train_data is an array, where the indices are the client_id.
Local data for each client can be accessed by:
train_data[client_id]['images']
train_data[client_id]['labels']

test_data contains the aggregated held-out test data.
The test data can be accessed by:
test_data[0]['images']
test_data[0]['labels']


This dataset is adapted from LEAF[1] by taking a subset of the users.

Reference:
[1] Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Koneˇcny`, H Brendan McMa- han, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. arXiv preprint arXiv:1812.01097, 2018.
