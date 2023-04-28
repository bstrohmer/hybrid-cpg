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
        self.rng_seed = np.random.randint(10**7) #if args['seed'] is None else args['seed'] 	#RUN WITH RANDOM SEED	
        self.time_resolution = args['delta_clock'] 		#equivalent to "delta_clock"
        self.inh_pop_neurons = args['inh_pop_size']
        self.rg_pop_neurons = args['rg_pop_size']
        self.exc_neurons_count = int(np.round(args['rg_pop_size'] * (args['ratio_exc_inh'] / (args['ratio_exc_inh'] + 1)))) # N_E = N*(r / (r+1))
        self.inh_neurons_count = int(np.round(args['rg_pop_size'] * ( 1 / (args['ratio_exc_inh'] + 1))))         # N_I = N*(1 / (r+1))
        self.exc_tonic_count = round(self.exc_neurons_count*args['exc_pct_tonic'])
        self.exc_irregular_count = self.exc_neurons_count-self.exc_tonic_count
        self.inh_tonic_count = round(self.inh_neurons_count*args['inh_pct_tonic'])
        self.inh_irregular_count = self.inh_neurons_count-self.inh_tonic_count
        self.sim_time = args['t_steps']         #time in ms
        
        #Initialize neuronal parameters
        self.V_th_initial = nest.random.normal(mean=-52.0, std=1.0) #mV
        self.V_m_initial = nest.random.normal(mean=-60.0, std=10.0) #mV
        self.C_m_initial_irregular = nest.random.normal(mean=500.0, std=80.0) #nF - UPDATED FROM 500/80
        self.C_m_initial_tonic = nest.random.normal(mean=200.0, std=40.0) #pF 500/80
        self.t_ref_initial = nest.random.normal(mean=1.0, std=0.2) #ms
        self.w_exc_initial = nest.random.normal(mean=args['coupling']/args['ratio_exc_inh'], std=args['coupling_std'])+(args['w_exc_multiplier']*args['coupling']) #nS, divide mean weight by exc/inh ratio to keep network balance (control is 2x)
        #self.w_exc_initial = nest.random.normal(mean=args['coupling'], std=args['coupling_std'])
        self.w_inh_initial = -1*nest.random.normal(mean=args['coupling'], std=args['coupling_std']) #nS        
        self.w_strong_inh_initial = -2*nest.random.normal(mean=args['coupling'], std=args['coupling_std']) #nS
        self.I_e_irregular = nest.random.normal(mean=160.0, std=40.0) #pA Control = 160/40
        self.I_e_tonic = nest.random.normal(mean=320.0, std=80.0) #pA Control = 320/80		
        self.noise_std_dev_tonic = args['noise_amplitude_tonic'] #pA
        self.noise_std_dev_irregular = args['noise_amplitude_irregular'] #pA
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


        #Set neuron parameters
        self.irregular_neuronparams = {'C_m':self.C_m_initial_irregular, 'g_L':26.,'E_L':-60.,'V_th':self.V_th_initial,'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':self.I_e_irregular,'t_ref':self.t_ref_initial,'V_m':self.V_m_initial} #irregular spiking, Naud et al. 2008, C = pF; g_L = nS
        self.irregular_neuronparams_wo_Ie = {'C_m':self.C_m_initial_irregular, 'g_L':26.,'E_L':-60.,'V_th':self.V_th_initial,'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':0,'t_ref':self.t_ref_initial,'V_m':self.V_m_initial} #irregular spiking, Naud et al. 2008, C = pF; g_L = nS
        self.tonic_neuronparams = {'C_m':self.C_m_initial_tonic, 'g_L':10.,'E_L':-70.,'V_th':-50.,'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':self.I_e_tonic,'t_ref':self.t_ref_initial,'V_m':self.V_m_initial} #tonic firing, Naud et al. 2008, Fig 4a

        #Set spike detector parameters 
        self.sd_params = {"withtime" : True, "withgid" : True, 'to_file' : False, 'flush_after_simulate' : False, 'flush_records' : True}

        #Set synapse parameters
        self.inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : self.w_inh_initial, #nS            
            "delay" : args['synaptic_delay']}	#ms
        self.strong_inh_syn_params = {"synapse_model":"static_synapse",
            "weight" : self.w_strong_inh_initial, #nS            
            "delay" : args['synaptic_delay']}	#ms
        self.exc_syn_params = {"synapse_model":"static_synapse",
            "weight" : self.w_exc_initial, #nS
            "delay" : args['synaptic_delay']}	#ms
        
        self.conn_dict_custom_rg = {'rule': 'pairwise_bernoulli', 'p': args['sparsity_rg']}		#connection probability between any 2 neurons
        self.conn_dict_custom_cpg = {'rule': 'pairwise_bernoulli', 'p': args['sparsity_cpg']}

        #Set multimeter parameters
        self.mm_params = {'interval': 1., 'record_from': ['V_m', 'g_ex', 'g_in']}

        #Set noise parameters
        self.noise_params_tonic = {"dt": self.time_resolution, "std":self.noise_std_dev_tonic}
        self.noise_params_irregular = {"dt": self.time_resolution, "std":self.noise_std_dev_irregular}
        
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

