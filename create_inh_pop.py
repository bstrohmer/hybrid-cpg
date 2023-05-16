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

class create_inh_population():
    def __init__(self):
        self.senders = []
        self.spiketimes = []
        self.saved_spiketimes = []
        self.saved_senders = []
        self.time_window = 50		#50*0.1=5ms time window, based on time resolution of 0.1
        self.count = 0
        
        #Create populations for rg
        self.irregular_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_irregular_mean, std=netparams.C_m_irregular_std), 'g_L':26.,'E_L':-60.,'V_th':nest.random.normal(mean=netparams.V_th_mean, std=netparams.V_th_std),'Delta_T':2.,'tau_w':130., 'a':-11., 'b':30., 'V_reset':-48., 'I_e':nest.random.normal(mean=netparams.I_e_irregular_mean, std=netparams.I_e_irregular_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std)} #irregular spiking, Naud et al. 2008, C = pF; g_L = nS
        self.tonic_neuronparams = {'C_m':nest.random.normal(mean=netparams.C_m_tonic_mean, std=netparams.C_m_tonic_std), 'g_L':10.,'E_L':-70.,'V_th':-50.,'Delta_T':2.,'tau_w':30., 'a':3., 'b':0., 'V_reset':-58., 'I_e':nest.random.normal(mean=netparams.I_e_tonic_mean, std=netparams.I_e_tonic_std),'t_ref':nest.random.normal(mean=netparams.t_ref_mean, std=netparams.t_ref_std),'V_m':nest.random.normal(mean=netparams.V_m_mean, std=netparams.V_m_std)}
        	
        #self.inh_pop = nest.Create("aeif_cond_alpha",netparams.inh_pop_neurons,self.tonic_neuronparams)
        self.inh_pop = nest.Create("aeif_cond_alpha",netparams.inh_pop_neurons,self.irregular_neuronparams)	

        #Create noise
        self.white_noise = nest.Create("noise_generator",netparams.noise_params_irregular)
        
        #Create spike detectors (for recording spikes)
        self.spike_detector_inh = nest.Create("spike_recorder",netparams.inh_pop_neurons)
                
        #Create multimeters (for recording membrane potential)
        self.mm_inh = nest.Create("multimeter",netparams.mm_params)
	
        #Connect white noise to neurons
        nest.Connect(self.white_noise,self.inh_pop,"all_to_all")

        #Connect spike detectors to neuron populations
        nest.Connect(self.inh_pop,self.spike_detector_inh,"one_to_one")
        self.spike_detector_inh.n_events = 0		#ensure no spikes left from previous simulations

        #Connect multimeters to neuron populations
        nest.Connect(self.mm_inh,self.inh_pop)
            
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

    def count_indiv_spikes(self,total_neurons,neuron_id_data):
        self.spike_count_array = [len(neuron_id_data[0][i]) for i in range(total_neurons)]
        self.less_than_10_indices = [i for i, count in enumerate(self.spike_count_array) if count>=1 and count<10]
        self.silent_neuron_count = [i for i, count in enumerate(self.spike_count_array) if count==0]
        self.neuron_to_sample = self.less_than_10_indices[1] if len(self.less_than_10_indices) > 1 else 0
        return self.spike_count_array,self.neuron_to_sample,len(self.less_than_10_indices),len(self.silent_neuron_count)      
        
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
            spike_time_index = int(spike_data[j]*(1/netparams.time_resolution))-1
            spike_time[spike_time_index]=spike_data[j]        
        return spike_time

    def single_neuron_spikes_binary(self,neuron_number,population):
        spike_time = [0]*int(netparams.sim_time/netparams.time_resolution)
        spike_data = population[0][neuron_number]
        for j in range(spike_data.shape[0]):
            spike_time_index = int(spike_data[j]*(1/netparams.time_resolution))-1
            spike_time[spike_time_index]=1        
        return spike_time

    def rate_code_spikes(self, neuron_count, output_spiketimes):
        # Initialize the spike bins array as a 2D array
        bins=np.arange(0, netparams.sim_time+netparams.time_resolution,netparams.time_resolution)
        # Loop over each neuron
        for i in range(neuron_count):
            t_spikes = output_spiketimes[0][i]
            # Use numpy's histogram function to assign each spike to its corresponding time bin index
            spikes_per_bin,bin_edges=np.histogram(t_spikes, bins)
            # Add the spike counts to the `spike_bins_current` array
            if i == 0:
                spike_bins_current = spikes_per_bin
            else:
                spike_bins_current += spikes_per_bin
        spike_bins_current = self.sliding_time_window(spike_bins_current,self.time_window)
        from scipy.ndimage import gaussian_filter
        smoothed_spike_bins = gaussian_filter(spike_bins_current, netparams.convstd)
        return smoothed_spike_bins
        
    def sliding_time_window(self,signal, window_size):
        windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
        return np.sum(windows, axis=1)    
   
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
        self.binary_spikes = np.vstack([self.single_neuron_spikes_binary(i, population) for i in range(population_size)])
        smoothed_spikes = self.smooth(self.binary_spikes, netparams.convstd)
        if netparams.chop_edges_amount > 0.0:
            smoothed_spikes = smoothed_spikes[:, int(netparams.chop_edges_amount*self.binary_spikes.shape[-1]) : int(self.binary_spikes.shape[-1] - netparams.chop_edges_amount*self.binary_spikes.shape[-1])] # chop edges to remove artifiacts induce by convoing over the gaussian
        if netparams.remove_mean:
            smoothed_spikes = (smoothed_spikes.T - np.mean(smoothed_spikes, axis=1)).T
        if netparams.high_pass_filtered:            
            from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
            # Same used as in Linden et al, 2022 paper
            b, a = butter(3, .1, 'highpass', fs=1000)   #high pass freq was previously 0.3Hz
            smoothed_spikes = filtfilt(b, a, smoothed_spikes)
            smoothed_spikes = smoothed_spikes[:, int(netparams.chop_edges_amount*smoothed_spikes.shape[-1]) : int(smoothed_spikes.shape[-1] - netparams.chop_edges_amount*smoothed_spikes.shape[-1])]
        if netparams.downsampling_convolved:
            from scipy.signal import decimate
            smoothed_spikes = decimate(smoothed_spikes, int(1/netparams.time_resolution), n=2, ftype='iir', zero_phase=True)
        smoothed_spikes = smoothed_spikes[:, :-self.time_window+1]
        return smoothed_spikes
        
inh = create_inh_population()       	        
