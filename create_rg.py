#!/usr/bin/env python

#include <static_connection.h>
import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as pyplot
import pickle, yaml
import random
import scipy
import scipy.fftpack
from scipy.signal import find_peaks, peak_widths, peak_prominences
import time
import copy
from set_network_params import neural_network
netparams = neural_network()

class create_rg_population():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50.		#50= 5ms (0.1ms*50 = 5ms), this is based on the delta_clock
        self.count = 0
        
        #Create populations for rg
        self.rg_exc_irregular = nest.Create("aeif_cond_alpha",netparams.exc_irregular_count,netparams.irregular_neuronparams)		
        self.rg_inh_irregular = nest.Create("aeif_cond_alpha",netparams.inh_irregular_count,netparams.irregular_neuronparams)	
        if netparams.exc_tonic_count != 0: self.rg_exc_tonic = nest.Create("aeif_cond_alpha",netparams.exc_tonic_count,netparams.tonic_neuronparams) 	
        if netparams.inh_tonic_count != 0: self.rg_inh_tonic = nest.Create("aeif_cond_alpha",netparams.inh_tonic_count,netparams.tonic_neuronparams) 
        
        #Create noise
        self.white_noise_tonic = nest.Create("noise_generator",netparams.noise_params_tonic) 
        self.white_noise_irregular = nest.Create("noise_generator",netparams.noise_params_irregular)     
        
        #Create spike detectors (for recording spikes)
        self.spike_detector_rg_exc_irregular = nest.Create("spike_recorder",netparams.exc_irregular_count)
        self.spike_detector_rg_inh_irregular = nest.Create("spike_recorder",netparams.inh_irregular_count)
        if netparams.exc_tonic_count != 0: 
            self.spike_detector_rg_exc_tonic = nest.Create("spike_recorder",netparams.exc_tonic_count)
        if netparams.inh_tonic_count != 0: 
            self.spike_detector_rg_inh_tonic = nest.Create("spike_recorder",netparams.inh_tonic_count)
                
        #Create multimeters (for recording membrane potential)
        self.mm_rg_exc_irregular = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_inh_irregular = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_exc_tonic = nest.Create("multimeter",netparams.mm_params)
        self.mm_rg_inh_tonic = nest.Create("multimeter",netparams.mm_params)
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise_irregular,self.rg_exc_irregular,"all_to_all")
        nest.Connect(self.white_noise_irregular,self.rg_inh_irregular,"all_to_all")
        if netparams.exc_tonic_count != 0: nest.Connect(self.white_noise_tonic,self.rg_exc_tonic,"all_to_all") 
        if netparams.inh_tonic_count != 0: nest.Connect(self.white_noise_tonic,self.rg_inh_tonic,"all_to_all") 
	
        #Connect neurons within rg
        self.coupling_exc_inh = nest.Connect(self.rg_exc_irregular,self.rg_inh_irregular,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
        self.coupling_exc_exc = nest.Connect(self.rg_exc_irregular,self.rg_exc_irregular,netparams.conn_dict_custom_rg,netparams.exc_syn_params)  	  
        self.coupling_inh_exc = nest.Connect(self.rg_inh_irregular,self.rg_exc_irregular,netparams.conn_dict_custom_rg,netparams.inh_syn_params)  
        self.coupling_inh_inh = nest.Connect(self.rg_inh_irregular,self.rg_inh_irregular,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
        if netparams.exc_tonic_count != 0: 
            self.coupling_exc_tonic_inh = nest.Connect(self.rg_exc_tonic,self.rg_inh_irregular,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
            self.coupling_exc_tonic_exc = nest.Connect(self.rg_exc_tonic,self.rg_exc_irregular,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
            self.coupling_exc_inh_tonic = nest.Connect(self.rg_exc_irregular,self.rg_inh_tonic,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
            self.coupling_exc_exc_tonic = nest.Connect(self.rg_exc_irregular,self.rg_exc_tonic,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
            self.coupling_exc_tonic_exc_tonic = nest.Connect(self.rg_exc_tonic,self.rg_exc_tonic,netparams.conn_dict_custom_rg,netparams.exc_syn_params)
            self.coupling_exc_tonic_inh_tonic = nest.Connect(self.rg_exc_tonic,self.rg_inh_tonic,netparams.conn_dict_custom_rg,netparams.exc_syn_params)            
        if netparams.inh_tonic_count != 0: 
            self.coupling_inh_tonic_inh = nest.Connect(self.rg_inh_tonic,self.rg_exc_irregular,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
            self.coupling_inh_tonic_exc = nest.Connect(self.rg_inh_tonic,self.rg_inh_irregular,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
            self.coupling_inh_exc_tonic = nest.Connect(self.rg_inh_irregular,self.rg_exc_tonic,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
            self.coupling_inh_inh_tonic = nest.Connect(self.rg_inh_irregular,self.rg_inh_tonic,netparams.conn_dict_custom_rg,netparams.inh_syn_params)
            self.coupling_inh_tonic_inh_tonic = nest.Connect(self.rg_inh_tonic,self.rg_inh_tonic,netparams.conn_dict_custom_rg,netparams.inh_syn_params)           
            self.coupling_exc_tonic_inh_tonic = nest.Connect(self.rg_inh_tonic,self.rg_exc_tonic,netparams.conn_dict_custom_rg,netparams.inh_syn_params) 

        #Connect spike detectors to neuron populations
        nest.Connect(self.rg_exc_irregular,self.spike_detector_rg_exc_irregular,"one_to_one")
        nest.Connect(self.rg_inh_irregular,self.spike_detector_rg_inh_irregular,"one_to_one")
        self.spike_detector_rg_exc_irregular.n_events = 0		#ensure no spikes left from previous simulations
        self.spike_detector_rg_inh_irregular.n_events = 0		#ensure no spikes left from previous simulations
        if netparams.exc_tonic_count != 0: 
            nest.Connect(self.rg_exc_tonic,self.spike_detector_rg_exc_tonic,"one_to_one")
            self.spike_detector_rg_exc_tonic.n_events = 0	#ensure no spikes left from previous simulations
        if netparams.inh_tonic_count != 0: 
            nest.Connect(self.rg_inh_tonic,self.spike_detector_rg_inh_tonic,"one_to_one")
            self.spike_detector_rg_inh_tonic.n_events = 0	#ensure no spikes left from previous simulations
                    
        #Connect multimeters to neuron populations
        nest.Connect(self.mm_rg_exc_irregular,self.rg_exc_irregular)
        nest.Connect(self.mm_rg_inh_irregular,self.rg_inh_irregular)
        if netparams.exc_tonic_count != 0: 
            nest.Connect(self.mm_rg_exc_tonic,self.rg_exc_tonic)
        if netparams.inh_tonic_count != 0: 
            nest.Connect(self.mm_rg_inh_tonic,self.rg_inh_tonic)
            
    def update_neuronal_characteristic(self,update_charac,neuron_population,leakage_value):
        self.neuron_charac = update_charac
        for neuron in neuron_population:
            nest.SetStatus(neuron, {self.neuron_charac: leakage_value})
        self.new_val = nest.GetStatus(neuron_population, keys=self.neuron_charac)[0]
        return self.new_val
        
    def read_spike_data(self,spike_detector):
        self.senders = []
        self.spiketimes = []
        self.spike_detector = spike_detector
        self.senders += [self.spike_detector.get('events', 'senders')]
        self.spiketimes += [self.spike_detector.get('events', 'times')]
        return self.senders,self.spiketimes

    def read_membrane_potential(self,multimeter,pop_size,neuron_num):
        mm = nest.GetStatus(multimeter,keys="events")[0]
        vm =  mm['V_m']
        t_vm = mm['times']
        vm = vm[neuron_num::pop_size]
        t_vm = t_vm[neuron_num::pop_size]
        return vm,t_vm

    def count_indiv_spikes(self,total_neurons,neuron_id_data):
        self.spike_count_array = []
        for i in range(total_neurons):
            self.neuron_id = neuron_id_data[0][i]
            self.spike_count_array.append(len(self.neuron_id))
        return self.spike_count_array

    def save_spike_data(self,num_neurons,population,neuron_num_offset):
        spike_time = []
        all_spikes = []
        for i in range(num_neurons):
            spike_data = population[0][i]
            neuron_num = [i+neuron_num_offset]*spike_data.shape[0]
            for j in range(spike_data.shape[0]):
                spike_time.append(spike_data[j])    
            indiv_spikes = list(zip(neuron_num,spike_time))
            all_spikes.extend(indiv_spikes)  
            spike_time = []     
        return all_spikes

    def single_neuron_spikes(self,neuron_number,population):
        spike_time = [0]*int(netparams.sim_time/netparams.time_resolution)
        spike_data = population[0][neuron_number]
        for j in range(spike_data.shape[0]):
            #print(spike_data[j])
            #print(spike_data[j]*10)
            spike_time_index = int(spike_data[j]*10)-1
            spike_time[spike_time_index]=spike_data[j]        
        return spike_time

    def single_neuron_spikes_binary(self,neuron_number,population):
        spike_time = [0]*int(netparams.sim_time/netparams.time_resolution)
        spike_data = population[0][neuron_number]
        for j in range(spike_data.shape[0]):
            spike_time_index = int(spike_data[j]*10)-1
            spike_time[spike_time_index]=1        
        return spike_time
            
    def rate_code_spikes(self,neuron_count,output_spiketimes):
        for i in range(neuron_count):
            spike_total_current = []
            t_spikes = output_spiketimes[0][i]
            step = self.time_window
            for n in range(int(netparams.sim_time/netparams.time_resolution)):
                spike_total_current.append(len(list(x for x in t_spikes if step-self.time_window <= x <= step)))
                step = step + netparams.time_resolution
            if i == 0:
                spike_bins_current = spike_total_current
            else:
                spike_bins_current = np.add(spike_bins_current,spike_total_current)
        return spike_bins_current
        
    def smooth(self, data, sd):
        data = copy.copy(data)       
        from scipy.signal import gaussian
        from scipy.signal import convolve
        n_bins = data.shape[1]
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=sd)
        for j in range(data.shape[0]):
            data[j,:] = convolve(data[j,:], window, mode='same', method='auto') 
        return data
    
    def convolve_spiking_activity(self,population_size,population):
        time_steps = int(netparams.sim_time/netparams.time_resolution)
        self.binary_spikes = self.single_neuron_spikes_binary(0,population)
        for i in range(population_size-1):
            self.binary_spikes = np.vstack([self.binary_spikes,self.single_neuron_spikes_binary(i+1,population)])
        smoothed_spikes = self.smooth(self.binary_spikes, netparams.convstd)
        if netparams.chop_edges_amount > 0.0:
            smoothed_spikes = smoothed_spikes[:, int(netparams.chop_edges_amount*self.binary_spikes.shape[-1]) : int(self.binary_spikes.shape[-1] - netparams.chop_edges_amount*self.binary_spikes.shape[-1])] # chop edges to remove artifiacts induce by convoing over the gaussian
        if netparams.remove_mean:
            smoothed_spikes = (smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T
        if netparams.high_pass_filtered:            
            from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
            # Same used as in Linden et al, 2022 paper
            b, a = butter(3, .1, 'highpass', fs=1000)		#high pass freq was previously 0.3Hz
            smoothed_spikes = filtfilt(b, a, smoothed_spikes)
            smoothed_spikes = smoothed_spikes[:, int(netparams.chop_edges_amount*smoothed_spikes.shape[-1]) : int(smoothed_spikes.shape[-1] - netparams.chop_edges_amount*smoothed_spikes.shape[-1])]
        if netparams.downsampling_convolved:
            from scipy.signal import decimate
            smoothed_spikes = decimate(smoothed_spikes, int(1/netparams.time_resolution), n=2, ftype='iir', zero_phase=True)
        return smoothed_spikes
        
    def inject_current(self,neuron_population,current):
        for neuron in neuron_population:
            nest.SetStatus([neuron],{"I_e": current})
        updated_current = nest.GetStatus(neuron_population, keys="I_e")[0]
        return updated_current
   

        
rg = create_rg_population()       	        
