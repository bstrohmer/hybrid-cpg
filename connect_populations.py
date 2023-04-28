#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.signal import find_peaks, peak_widths, peak_prominences
import time
import numpy as np
import copy
from set_network_params import neural_network
netparams = neural_network()

class connect():
    def __init__(self):
        self.total_weight_exc = 0
        self.total_weight_inh = 0
        self.balance_pct = 0
        
    def create_connections(self,pop1,pop2,syn_type):
        #Connect populations
        if syn_type=='exc':
            self.coupling_populations_exc = nest.Connect(pop1,pop2,netparams.conn_dict_custom_rg,netparams.exc_syn_params) 
            #print('Excitatory connections created')
        if syn_type=='inh':
            self.coupling_populations_inh = nest.Connect(pop1,pop2,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
            #print('Inhibitory connections created')
        if syn_type=='inh_strong':
            self.coupling_populations_strong_inh = nest.Connect(pop1,pop2,netparams.conn_dict_custom_cpg,netparams.strong_inh_syn_params)
            #print('Strong inhibitory connections created')

    def sum_weights_per_source(self,population):
        synapse_data = nest.GetConnections(population).get(['source', 'weight'])
        weights_per_source = {}
        for connection in synapse_data:
            source_neuron = synapse_data['source']
            weights = synapse_data['weight']
            for s in set(source_neuron):
                if s not in weights_per_source:
                    weights_per_source[s] = sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
                else:
                    weights_per_source[s] += sum([w for i, w in enumerate(weights) if source_neuron[i] == s])
        return weights_per_source
    
    def count_spikes_per_source(self,spike_detector):
        sender_counts = {}
        spike_data = spike_detector.get('events', 'senders')
        #print('Sender data: ',spike_data)
        for sender_list in spike_data:
            for sender in sender_list:
                if sender not in sender_counts:
                    sender_counts[sender] = 1
                else:
                    sender_counts[sender] += 1
        return sender_counts
    
    def calculate_weighted_balance(self, pop1,spike_detector):
        self.total_weight = 0 
        self.weights_by_source = self.sum_weights_per_source(pop1)
        self.sender_counts = self.count_spikes_per_source(spike_detector)
        #print('Count per neuron ID: ',self.sender_counts)        
        #print('Weights by source: ',self.weights_by_source)
        for source in self.weights_by_source:
            #print('Neuron ID: ',source)
            if source in self.sender_counts:
                weighted_weight = self.weights_by_source[source] * self.sender_counts[source]
            else:
                weighted_weight = 0
            self.total_weight += weighted_weight
        self.total_weight = self.total_weight*2 if self.total_weight < 0 else self.total_weight*.2
        return round(self.total_weight,2)
        
conn = connect()
