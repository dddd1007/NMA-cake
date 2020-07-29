####### Setup #######

# Importing Packages

import os
from os import name
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

def calculate_nodes_count(location_index, brain_groups, group_name):
    each_count_areas = []
    waiting_to_count = brain_groups[group_name]
    
    for area in waiting_to_count:
        each_count_areas.append(len(location_index_extractor(location_index, area)))
        
    all_nodes_counts = np.sum(each_count_areas)
    
    return all_nodes_counts 

# Parameters of network

bins = data['bin_size']
spikes = data['spks']
location = data['brain_area']

nodes_counts = len(np.unique(location))
regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
brain_groups = {"visual_cortex": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                "thalamus": ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                "hippocampal": ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # 
                "non_visual_cortex": ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                "midbrain": ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                "basal_ganglia": ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                "cortical_subplate": ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                }
brain_groups_keys = brain_groups.keys()

###### Building the Network ######

mice_snn = Network(dt = 1, learning = "Hebbian")

visual_input = nodes.Input(n = calculate_nodes_count(location, brain_groups, "visual_cortex"))
action = nodes.Input(n = 1)

mice_snn.add_layer(
    layer = visual_input, name = "visual_cortex",
    layer = action, name = "action"
)

