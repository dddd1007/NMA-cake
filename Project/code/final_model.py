####### Setup #######

# Importing Packages

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bindsnet.network import Network, nodes, topology, monitors
from bindsnet import encoding

# Importing Data

alldat = np.array([])
for i in range(3):
  alldat = np.hstack((alldat, np.load('Project/steinmetz_part%d.npz'%i, allow_pickle=True)['dat']))

data = alldat[7]

# Function Defination

def location_index_extractor(location_index, area_name:str):
    """
    Args:
        location_index ([numpy.ndarry]): The dict of location
        area_name (str): The location you want to search

    Returns:
        np.array: The index of area name in location_index
    """

    loc = np.argwhere(location_index == area_name)
    return loc

# Parameters of network

bins = data['bin_size']
spikes = data['spks']
location = data['brain_area']

nodes_counts = len(np.unique(location))

mice_cnn = Network(dt = 1, learning = "Hebbian")



