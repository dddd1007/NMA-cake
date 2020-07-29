import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.network import Network, nodes, topology, monitors
from bindsnet import encoding

demo_network = Network(dt=1.0)

X = nodes.Input(100)
Y = nodes.LIFNodes(100)
C = topology.Connection(source = X, target = Y, w = torch.rand(X.n, Y.n))

M1 = monitors.Monitor(obj = X, state_vars = ['s'])
M2 = monitors.Monitor(obj = Y, state_vars = ['s'])

demo_network.add_layer(layer = X, name = 'X')
demo_network.add_layer(layer = Y, name = 'Y')
demo_network.add_connection(connection = C, source = 'X', target = 'Y')

demo_network.add_monitor(monitor = M1, name = 'X')
demo_network.add_monitor(monitor = M2, name = 'Y')

data = 15 * torch.rand(100)
train = encoding.poisson(datum = data, time = 5000)
inputs = {'X' : train}
demo_network.run(inputs = inputs, time = 5000)
spikes = {'X' : M1.get('s'), 'Y' : M2.get('s') }

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for i, layer in enumerate(spikes):
    axes[i].matshow(spikes[layer].squeeze_(), cmap='binary')
    tmp = spikes[layer].squeeze_()
    axes[i].set_title('%s spikes' % layer)
    axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
    axes[i].set_xticks(()); axes[i].set_yticks(())
    axes[i].set_aspect('auto')
