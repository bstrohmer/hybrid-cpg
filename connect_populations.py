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
    
    def calculate_balance(self,pop1,pop2,synapse_type):
        self.synapse_data = nest.GetConnections(pop1,pop2).get(['source', 'target', 'weight'])
        self.synapse_weight = sum(self.synapse_data['weight'])
        if synapse_type == 'exc':
            self.total_weight_exc += self.synapse_weight
            #print(self.total_weight_exc)
        if synapse_type == 'inh' or synapse_type == 'inh_strong':
            self.total_weight_inh += self.synapse_weight
            #print(self.total_weight_inh)
        if self.total_weight_inh != 0:
            #self.balance_pct = round((round(self.total_weight_exc)/-round(self.total_weight_inh))*100)
            self.balance_pct = round(((round(self.total_weight_exc)+round(self.total_weight_inh))/(round(self.total_weight_exc)-round(self.total_weight_inh)))*100)
        return self.balance_pct 

    def reset_balance(self):
        self.total_weight_exc = 0
        self.total_weight_inh = 0
        print('Balance reset')    
        
conn = connect()
