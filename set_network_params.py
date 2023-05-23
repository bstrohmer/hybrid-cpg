#!/usr/bin/env python

import nest
import numpy as np
import pathlib, sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import pickle, yaml
import time, datetime

#Import parameters for network
file = open(r'configuration_run_nest.yaml')
args = yaml.load(file, Loader=yaml.FullLoader)
print(f'\nLoading parameters from configuration file:\n')

class neural_network():
    def __init__(self):
        #Import parameters for network
        #file = open(r'configuration_run_nest.yaml')
        #args = yaml.load(file, Loader=yaml.FullLoader)
        #print(f'\nLoading parameters from configuration file:\n')
        self.args = args

        #Set parameters for network
        self.rng_seed = np.random.randint(10**7) if args['seed'] is 0 else args['seed'] 	#RUN WITH RANDOM SEED	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"
        self.inh_pop_neurons = args['inh_pop_size']
        self.rg_pop_neurons = args['rg_pop_size']
        self.exc_neurons_count = int(np.round(args['rg_pop_size'] * (args['ratio_exc_inh'] / (args['ratio_exc_inh'] + 1)))) # N_E = N*(r / (r+1))
        self.inh_neurons_count = int(np.round(args['rg_pop_size'] * ( 1 / (args['ratio_exc_inh'] + 1))))         # N_I = N*(1 / (r+1))
        self.exc_tonic_count = round(self.exc_neurons_count*args['exc_pct_tonic'])
        self.exc_bursting_count = self.exc_neurons_count-self.exc_tonic_count
        self.inh_tonic_count = round(self.inh_neurons_count*args['inh_pct_tonic'])
        self.inh_bursting_count = self.inh_neurons_count-self.inh_tonic_count
        self.sim_time = args['t_steps']         #time in ms
        
        #Set neuronal and synaptic parameters
        self.V_th_mean = -52.0 #mV
        self.V_th_std = 1.0 #mV
        self.V_m_mean = -60.0 #mV 
        self.V_m_std = 10.0 #mV
        self.C_m_bursting_mean = 500.0 #pF
        self.C_m_bursting_std = 80.0 #pF 
        self.C_m_tonic_mean = 200.0 #pF
        self.C_m_tonic_std =40.0 #pF
        self.t_ref_mean = 1.0 #ms
        self.t_ref_std = 0.2 #ms
        self.coupling_strength = args['coupling']
        self.w_exc_mean = args['coupling']/args['ratio_exc_inh']+args['w_exc_multiplier']*args['coupling'] #nS
        self.w_exc_std = args['coupling_std'] #nS
        self.w_exc_multiplier = args['w_exc_multiplier']
        self.w_inh_mean = -1*args['coupling'] if args['remove_inhibition'] == 0 else args['coupling']
        self.w_inh_std = args['coupling_std'] #nS        
        self.w_strong_inh_mean = -2*args['coupling'] #nS
        self.w_strong_inh_std = args['coupling_std'] #nS
        self.synaptic_delay = args['synaptic_delay']
        self.I_e_bursting_mean = 160.0 #pA Control = 160
        self.I_e_bursting_std = 40.0 #pA Control = 40
        self.I_e_tonic_mean = 320.0 #pA Control = 320
        self.I_e_tonic_std = 80.0 #pA Control = 0		
        self.noise_std_dev_tonic = args['noise_amplitude_tonic'] #pA
        self.noise_std_dev_bursting = args['noise_amplitude_bursting'] #pA
        self.freezing_enabled = args['freezing_enabled']
        self.rgs_connected = args['rgs_connected']

        #Set data evaluation parameters
        self.convstd = args['convstd']
        self.chop_edges_amount = args['chop_edges_amount']
        self.remove_mean = args['remove_mean']
        self.high_pass_filtered = args['high_pass_filtered']
        self.downsampling_convolved = args['downsampling_convolved']
        self.remove_silent = args['remove_silent']
        self.PCA_components = args['PCA_components']
        self.calculate_balance = args['calculate_balance']
        self.raster_plot = args['raster_plot']
        self.rate_coded_plot = args['rate_coded_plot']
        self.spike_distribution_plot = args['spike_distribution_plot']
        self.pca_plot = args['pca_plot']
        self.phase_ordered_plot = args['phase_ordered_plot']
        self.membrane_potential_plot = args['membrane_potential_plot']
        self.time_window = args['smoothing_window']
        self.normalized_rate_coded_plot = args['normalized_rate_coded_plot']
        self.excitation_file_generation_leg = args['excitation_file_generation_leg']
        self.excitation_file_generation_arm = args['excitation_file_generation_arm']
        self.excitation_gain = args['excitation_gain']

        #Set spike detector parameters 
        self.sd_params = {"withtime" : True, "withgid" : True, 'to_file' : False, 'flush_after_simulate' : False, 'flush_records' : True}

        #Set connection parameters
        self.conn_dict_custom_rg = {'rule': 'pairwise_bernoulli', 'p': args['sparsity_rg']}		#connection probability between any 2 neurons
        self.conn_dict_custom_cpg = {'rule': 'pairwise_bernoulli', 'p': args['sparsity_cpg']}

        #Set multimeter parameters
        self.mm_params = {'interval': 1., 'record_from': ['V_m']}#, 'g_ex', 'g_in']}

        #Set noise parameters
        self.noise_params_tonic = {"dt": self.time_resolution, "std":self.noise_std_dev_tonic}
        self.noise_params_bursting = {"dt": self.time_resolution, "std":self.noise_std_dev_bursting}
        
    ################
    # Save results #
    ################
    if args['save_results']:
        id_ = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #simulation_config['date'] = datetime.date.today()
        #simulation_config['id'] = id_
        path = 'saved_simulations' + '/' + id_ 
        pathFigures = 'saved_simulations' + '/' + id_ + '/Figures'
        pathlib.Path(path).mkdir(parents=True, exist_ok=False)
        pathlib.Path(pathFigures).mkdir(parents=True, exist_ok=False)
        with open(path + '/args_' + id_ + '.yaml', 'w') as yamlfile:
            #args['seed'] = simulation_config['seed']
            yaml.dump(args, yamlfile)

