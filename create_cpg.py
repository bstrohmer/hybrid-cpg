#!/usr/bin/env python

import nest
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import time
import start_simulation as ss
import set_network_params as netparams
import create_rg as rg
import create_inh_pop as inh
from connect_populations import connect
import pickle, yaml
import pandas as pd
from phase_ordering import order_by_phase
from pca import run_PCA
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

ss.nest_start()
conn=connect() 
nn=netparams.neural_network()

#Create neuron populations - NEST
rg1 = rg.create_rg_population()
rg2 = rg.create_rg_population()

if nn.rgs_connected==1:
	inh1 = inh.create_inh_population()
	inh2 = inh.create_inh_population()

	#Connect excitatory rg neurons to inhibitory populations
	conn.create_connections(rg1.rg_exc_irregular,inh1.inh_pop,'exc')  #EXCITATORY CONNECTIONS UPDATED TO SAME SPARSITY AS POST-SYNAPTIC FROM INH POPS
	conn.create_connections(rg1.rg_exc_tonic,inh1.inh_pop,'exc')
	conn.create_connections(rg2.rg_exc_irregular,inh2.inh_pop,'exc')
	conn.create_connections(rg2.rg_exc_tonic,inh2.inh_pop,'exc')

	#Connect inhibitory populations to all rg neurons
	conn.create_connections(inh1.inh_pop,rg2.rg_exc_irregular,'inh_strong')
	conn.create_connections(inh1.inh_pop,rg2.rg_exc_tonic,'inh_strong')
	conn.create_connections(inh1.inh_pop,rg2.rg_inh_irregular,'inh_strong')
	conn.create_connections(inh1.inh_pop,rg2.rg_inh_tonic,'inh_strong')

	conn.create_connections(inh2.inh_pop,rg1.rg_exc_irregular,'inh_strong')
	conn.create_connections(inh2.inh_pop,rg1.rg_exc_tonic,'inh_strong')
	conn.create_connections(inh2.inh_pop,rg1.rg_inh_irregular,'inh_strong')
	conn.create_connections(inh2.inh_pop,rg1.rg_inh_tonic,'inh_strong')

print("Seed#: ",nn.rng_seed)
print("# exc (bursting, tonic): ",nn.exc_irregular_count,nn.exc_tonic_count,"; # inh(bursting, tonic): ",nn.inh_irregular_count,nn.inh_tonic_count,"; # inh buffer: ",nn.inh_pop_neurons)

t_start = time.perf_counter()
nest.Simulate(nn.sim_time)
t_stop = time.perf_counter()    
print('Simulation completed. It took ',round(t_stop-t_start,2),' seconds.')

spike_count_array = []
#Read spike data - rg populations
senders_exc1,spiketimes_exc1 = rg1.read_spike_data(rg1.spike_detector_rg_exc_irregular)
senders_inh1,spiketimes_inh1 = rg1.read_spike_data(rg1.spike_detector_rg_inh_irregular)
senders_exc_tonic1,spiketimes_exc_tonic1 = rg1.read_spike_data(rg1.spike_detector_rg_exc_tonic)
senders_inh_tonic1,spiketimes_inh_tonic1 = rg1.read_spike_data(rg1.spike_detector_rg_inh_tonic)

senders_exc2,spiketimes_exc2 = rg2.read_spike_data(rg2.spike_detector_rg_exc_irregular)
senders_inh2,spiketimes_inh2 = rg2.read_spike_data(rg2.spike_detector_rg_inh_irregular)
senders_exc_tonic2,spiketimes_exc_tonic2 = rg2.read_spike_data(rg2.spike_detector_rg_exc_tonic)
senders_inh_tonic2,spiketimes_inh_tonic2 = rg2.read_spike_data(rg2.spike_detector_rg_inh_tonic)

#Read spike data - inhibitory populations
if nn.rgs_connected==1:
	senders_inhpop1,spiketimes_inhpop1 = inh1.read_spike_data(inh1.spike_detector_inh)
	senders_inhpop2,spiketimes_inhpop2 = inh2.read_spike_data(inh2.spike_detector_inh)

#Calculate synaptic balance of rg populations and total CPG network
if nn.calculate_balance==1:
		
	rg1_exc_irr_weight = conn.calculate_weighted_balance(rg1.rg_exc_irregular,rg1.spike_detector_rg_exc_irregular)
	rg1_inh_irr_weight = conn.calculate_weighted_balance(rg1.rg_inh_irregular,rg1.spike_detector_rg_inh_irregular)
	rg1_exc_tonic_weight = conn.calculate_weighted_balance(rg1.rg_exc_tonic,rg1.spike_detector_rg_exc_tonic)
	rg1_inh_tonic_weight = conn.calculate_weighted_balance(rg1.rg_inh_tonic,rg1.spike_detector_rg_inh_tonic)
	weights_per_pop1 = [rg1_exc_irr_weight,rg1_inh_irr_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight]
	absolute_weights_per_pop1 = [rg1_exc_irr_weight,abs(rg1_inh_irr_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight)]
	rg1_balance_pct = (sum(weights_per_pop1)/sum(absolute_weights_per_pop1))*100
	print('RG1 balance %: ',rg1_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
	
	rg2_exc_irr_weight = conn.calculate_weighted_balance(rg2.rg_exc_irregular,rg2.spike_detector_rg_exc_irregular)
	rg2_inh_irr_weight = conn.calculate_weighted_balance(rg2.rg_inh_irregular,rg2.spike_detector_rg_inh_irregular)
	rg2_exc_tonic_weight = conn.calculate_weighted_balance(rg2.rg_exc_tonic,rg2.spike_detector_rg_exc_tonic)
	rg2_inh_tonic_weight = conn.calculate_weighted_balance(rg2.rg_inh_tonic,rg2.spike_detector_rg_inh_tonic)
	weights_per_pop2 = [rg2_exc_irr_weight,rg2_inh_irr_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight]
	absolute_weights_per_pop2 = [rg2_exc_irr_weight,abs(rg2_inh_irr_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight)]
	rg2_balance_pct = (sum(weights_per_pop2)/sum(absolute_weights_per_pop2))*100
	print('RG2 balance %: ',rg2_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
	
	if nn.rgs_connected==1:
		inh1_weight = conn.calculate_weighted_balance(inh1.inh_pop,inh1.spike_detector_inh)
		inh2_weight = conn.calculate_weighted_balance(inh2.inh_pop,inh2.spike_detector_inh)
		weights_per_pop = [rg1_exc_irr_weight,rg1_inh_irr_weight,rg1_exc_tonic_weight,rg1_inh_tonic_weight,rg2_exc_irr_weight,rg2_inh_irr_weight,rg2_exc_tonic_weight,rg2_inh_tonic_weight,inh1_weight,inh1_weight]
		absolute_weights_per_pop = [rg1_exc_irr_weight,abs(rg1_inh_irr_weight),rg1_exc_tonic_weight,abs(rg1_inh_tonic_weight),rg2_exc_irr_weight,abs(rg2_inh_irr_weight),rg2_exc_tonic_weight,abs(rg2_inh_tonic_weight),abs(inh1_weight),abs(inh1_weight)]
		cpg_balance_pct = (sum(weights_per_pop)/sum(absolute_weights_per_pop))*100
		print('CPG balance %: ',cpg_balance_pct,' >0 skew excitatory; <0 skew inhibitory')

if nn.phase_ordered_plot==1:
	t_start = time.perf_counter()
	#Convolve spike data - rg populations
	rg_exc_convolved1 = rg1.convolve_spiking_activity(nn.exc_irregular_count,spiketimes_exc1)
	rg_exc_tonic_convolved1 = rg1.convolve_spiking_activity(nn.exc_tonic_count,spiketimes_exc_tonic1)
	rg_inh_convolved1 = rg1.convolve_spiking_activity(nn.inh_irregular_count,spiketimes_inh1)
	rg_inh_tonic_convolved1 = rg1.convolve_spiking_activity(nn.inh_tonic_count,spiketimes_inh_tonic1)
	spikes_convolved_all1 = np.vstack([rg_exc_convolved1,rg_inh_convolved1])
	spikes_convolved_all1 = np.vstack([spikes_convolved_all1,rg_exc_tonic_convolved1])
	spikes_convolved_all1 = np.vstack([spikes_convolved_all1,rg_inh_tonic_convolved1])

	rg_exc_convolved2 = rg2.convolve_spiking_activity(nn.exc_irregular_count,spiketimes_exc2)
	rg_exc_tonic_convolved2 = rg2.convolve_spiking_activity(nn.exc_tonic_count,spiketimes_exc_tonic2)
	rg_inh_convolved2 = rg2.convolve_spiking_activity(nn.inh_irregular_count,spiketimes_inh2)
	rg_inh_tonic_convolved2 = rg2.convolve_spiking_activity(nn.inh_tonic_count,spiketimes_inh_tonic2)
	spikes_convolved_all2 = np.vstack([rg_exc_convolved2,rg_inh_convolved2])
	spikes_convolved_all2 = np.vstack([spikes_convolved_all2,rg_exc_tonic_convolved2])
	spikes_convolved_all2 = np.vstack([spikes_convolved_all2,rg_inh_tonic_convolved2])
	spikes_convolved_rgs = np.vstack([spikes_convolved_all1,spikes_convolved_all2])
	spikes_convolved_complete_network = spikes_convolved_rgs
	
	#Convolve spike data - inh populations
	if nn.rgs_connected==1:
		inh1_convolved = inh1.convolve_spiking_activity(nn.inh_pop_neurons,spiketimes_inhpop1)
		inh2_convolved = inh2.convolve_spiking_activity(nn.inh_pop_neurons,spiketimes_inhpop2)

		#spikes_convolved_complete_network = np.vstack([spikes_convolved_all1,spikes_convolved_all2])
		spikes_convolved_complete_network = np.vstack([spikes_convolved_rgs,inh1_convolved])
		spikes_convolved_complete_network = np.vstack([spikes_convolved_complete_network,inh2_convolved])

	if nn.remove_silent:
	    print('Removing silent neurons')
	    spikes_convolved_all1 = spikes_convolved_all1[~np.all(spikes_convolved_all1 == 0, axis=1)]
	    spikes_convolved_all2 = spikes_convolved_all2[~np.all(spikes_convolved_all2 == 0, axis=1)]
	    spikes_convolved_rgs = spikes_convolved_rgs[~np.all(spikes_convolved_rgs == 0, axis=1)]
	    if nn.rgs_connected==1:
	        spikes_convolved_complete_network = spikes_convolved_complete_network[~np.all(spikes_convolved_complete_network == 0, axis=1)]
	t_stop = time.perf_counter()
	print('Convolved spiking activity complete, taking ',int(t_stop-t_start),' seconds.') #Originally 9 seconds for a 500ms simulation

#Run PCA - rg populations
if nn.pca_plot==1 and nn.phase_ordered_plot==1:
	run_PCA(spikes_convolved_all1,'rg1')
	run_PCA(spikes_convolved_all2,'rg2')
	run_PCA(spikes_convolved_complete_network,'all_pops')
	print('PCA complete')
if nn.pca_plot==1 and nn.phase_ordered_plot==0:
	print('The convolved spiking activity is required to run a PCA, ensure "phase_ordered_plot" is selected.')

#Create Rate Coded Output
if nn.rate_coded_plot==1:
	t_start = time.perf_counter()
	spike_bins_rg_exc1 = rg1.rate_code_spikes(nn.exc_irregular_count,spiketimes_exc1)
	spike_bins_rg_inh1 = rg1.rate_code_spikes(nn.inh_irregular_count,spiketimes_inh1)
	spike_bins_rg_exc_tonic1 = rg1.rate_code_spikes(nn.exc_tonic_count,spiketimes_exc_tonic1)
	spike_bins_rg_inh_tonic1 = rg1.rate_code_spikes(nn.inh_tonic_count,spiketimes_inh_tonic1)
	spike_bins_rg1 = spike_bins_rg_exc1+spike_bins_rg_exc_tonic1+spike_bins_rg_inh1+spike_bins_rg_inh_tonic1

	spike_bins_rg_exc2 = rg2.rate_code_spikes(nn.exc_irregular_count,spiketimes_exc2)
	spike_bins_rg_inh2 = rg2.rate_code_spikes(nn.inh_irregular_count,spiketimes_inh2)
	spike_bins_rg_exc_tonic2 = rg2.rate_code_spikes(nn.exc_tonic_count,spiketimes_exc_tonic2)
	spike_bins_rg_inh_tonic2 = rg2.rate_code_spikes(nn.inh_tonic_count,spiketimes_inh_tonic2)
	spike_bins_rg2=spike_bins_rg_exc2+spike_bins_rg_exc_tonic2+spike_bins_rg_inh2+spike_bins_rg_inh_tonic2
	spike_bins_rgs = spike_bins_rg1+spike_bins_rg2
	
	if nn.rgs_connected==1:
		spike_bins_inh1 = inh1.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop1)
		spike_bins_inh2 = inh2.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop2)
		spike_bins_inh = spike_bins_inh1+spike_bins_inh2
		spike_bins_all_pops = spike_bins_rgs+spike_bins_inh
	t_stop = time.perf_counter()
	print('rg1 peaks ',find_peaks(spike_bins_rg1,height=150,prominence=70)[0])
	print('rg2 peaks ',find_peaks(spike_bins_rg2,height=150,prominence=70)[0])
	print('Rate coded activity complete, taking ',int(t_stop-t_start),' seconds.')

#Plot phase sorted activity
if nn.phase_ordered_plot==1 and nn.rate_coded_plot==1:
	order_by_phase(spikes_convolved_all1, spike_bins_rg1, 'rg1', remove_mean = True, high_pass_filtered = True, generate_plot = True)
	order_by_phase(spikes_convolved_all2, spike_bins_rg2, 'rg2', remove_mean = True, high_pass_filtered = True, generate_plot = True)
	order_by_phase(spikes_convolved_rgs, spike_bins_rgs, 'rgs', remove_mean = True, high_pass_filtered = True, generate_plot = True)
	order_by_phase(spikes_convolved_complete_network, spike_bins_rgs, 'all_pops', remove_mean = True, high_pass_filtered = True, generate_plot = True) #UPDATED - compare summed output from rgs to all spikes in network (inh pops output is "absorbed" into rg so it does not directly contribute to the rate-coded output of the network)
if nn.phase_ordered_plot==1 and nn.rate_coded_plot==0:
    print('The rate-coded output must be calculated in order to produce a phase-ordered plot, ensure "rate_coded_plot" is selected.')

#Plot raster plot of individual spikes
if nn.raster_plot==1:
	pylab.figure()
	pylab.subplot(211)
	for i in range(nn.exc_irregular_count-1): 
	    pylab.plot(spiketimes_exc1[0][i],senders_exc1[0][i],'.',label='Exc')
	for i in range(nn.exc_tonic_count-1):
	    if nn.exc_tonic_count != 0: pylab.plot(spiketimes_exc_tonic1[0][i],senders_exc_tonic1[0][i],'.',label='Exc tonic')
	for i in range(nn.inh_irregular_count-1):
	    pylab.plot(spiketimes_inh1[0][i],senders_inh1[0][i],'.',label='Inh') #offset neuron number by total # of excitatory neurons
	for i in range(nn.inh_tonic_count-1):
	    if nn.inh_tonic_count != 0: pylab.plot(spiketimes_inh_tonic1[0][i],senders_inh_tonic1[0][i],'.',label='Inh tonic')   
	pylab.xlabel('Time (ms)')
	pylab.ylabel('Neuron #')
	pylab.title('Spike Output RGs')
	pylab.subplot(212)
	for i in range(nn.exc_irregular_count-1): 
	    pylab.plot(spiketimes_exc2[0][i],senders_exc2[0][i],'.',label='Exc')
	for i in range(nn.exc_tonic_count-1):
	    if nn.exc_tonic_count != 0: pylab.plot(spiketimes_exc_tonic2[0][i],senders_exc_tonic2[0][i],'.',label='Exc tonic')
	for i in range(nn.inh_irregular_count-1):
	    pylab.plot(spiketimes_inh2[0][i],senders_inh2[0][i],'.',label='Inh') #offset neuron number by total # of excitatory neurons
	for i in range(nn.inh_tonic_count-1):
	    if nn.inh_tonic_count != 0: pylab.plot(spiketimes_inh_tonic2[0][i],senders_inh_tonic2[0][i],'.',label='Inh tonic')
	pylab.xlabel('Time (ms)')
	pylab.ylabel('Neuron #')
	#pylab.title('Spike Output rg2')
	pylab.subplots_adjust(bottom=0.15)
	if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spikes_rg.png',bbox_inches="tight")

	if nn.rgs_connected==1:
		pylab.figure()
		pylab.subplot(211)
		for i in range(nn.inh_pop_neurons-1): 
		    pylab.plot(spiketimes_inhpop1[0][i],senders_inhpop1[0][i],'.',label='Inh Pop1')
		pylab.xlabel('Time (ms)')
		pylab.ylabel('Neuron #')
		pylab.title('Spike Output Inh')
		pylab.subplot(212)
		for i in range(nn.inh_pop_neurons-1): 
		    pylab.plot(spiketimes_inhpop2[0][i],senders_inhpop2[0][i],'.',label='Inh Pop2')
		pylab.xlabel('Time (ms)')
		pylab.ylabel('Neuron #')
		#pylab.title('Spike Output Inh2')
		pylab.subplots_adjust(bottom=0.15)
		if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spikes_inh.png',bbox_inches="tight")

#Plot rate-coded output
if nn.rate_coded_plot==1:
	t = np.arange(0,len(spike_bins_rg1),1)
	pylab.figure()
	pylab.plot(t[200:],spike_bins_rg1[200:],label='RG1')		
	pylab.plot(t[200:],spike_bins_rg2[200:],label='RG2')
	#pylab.plot(t[200:],spike_bins_rgs[200:],label='Sum RGs')
	plt.legend( bbox_to_anchor=(1.1,1.05))		
	pylab.xlabel('Time (ms)')
	pylab.ylabel('Spike Count')
	pylab.title('Rate-coded Output per RG')
	if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.png',bbox_inches="tight")

if nn.spike_distribution_plot==1:
	#Count spikes per neuron
	indiv_spikes_exc1,neuron_to_sample_rg1_irr = rg1.count_indiv_spikes(nn.exc_irregular_count,senders_exc1)
	indiv_spikes_inh1,neuron_to_sample_rg1_irr_inh = rg1.count_indiv_spikes(nn.inh_irregular_count,senders_inh1)
	indiv_spikes_exc_tonic1,neuron_to_sample_rg1_ton = rg1.count_indiv_spikes(nn.exc_tonic_count,senders_exc_tonic1)
	indiv_spikes_inh_tonic1,neuron_to_sample_rg1_ton_inh = rg1.count_indiv_spikes(nn.inh_tonic_count,senders_inh_tonic1)

	indiv_spikes_exc2,neuron_to_sample_rg2_irr = rg2.count_indiv_spikes(nn.exc_irregular_count,senders_exc2)
	indiv_spikes_inh2,neuron_to_sample_rg2_irr_inh = rg2.count_indiv_spikes(nn.inh_irregular_count,senders_inh2)
	indiv_spikes_exc_tonic2,neuron_to_sample_rg2_ton = rg2.count_indiv_spikes(nn.exc_tonic_count,senders_exc_tonic2)
	indiv_spikes_inh_tonic2,neuron_to_sample_rg2_ton_inh = rg2.count_indiv_spikes(nn.inh_tonic_count,senders_inh_tonic2)
	all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2
	if nn.rgs_connected==1:
		indiv_spikes_inhpop1,neuron_to_sample_inh1 = inh1.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop1)
		indiv_spikes_inhpop2,neuron_to_sample_inh2 = inh2.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop2)
		all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+ indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2+indiv_spikes_inhpop1+indiv_spikes_inhpop2
	print('Length of spike count array (all) ',len(all_indiv_spike_counts))
	spike_distribution = [all_indiv_spike_counts.count(i) for i in range(max(all_indiv_spike_counts))]

	#print('Original spike counts: ',spike_distribution)
	pylab.figure()
	pylab.plot(spike_distribution[2:])
	pylab.xscale('log')
	pylab.xlabel('Total Spike Count')
	pylab.ylabel('Number of Neurons')
	pylab.title('Spike Distribution')
	if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.png',bbox_inches="tight")

if nn.membrane_potential_plot==1:
	#Read membrane potential of an irregularly firing neuron - neuron number <= population size
	v_m1_irr,t_m1_irr = rg1.read_membrane_potential(rg1.mm_rg_exc_irregular,nn.exc_irregular_count,neuron_to_sample_rg1_irr)
	v_m2_irr,t_m2_irr = rg2.read_membrane_potential(rg2.mm_rg_exc_irregular,nn.exc_irregular_count,neuron_to_sample_rg2_irr)

	#Read membrane potential of a tonically firing neuron - neuron number <= population size
	v_m1,t_m1 = rg1.read_membrane_potential(rg1.mm_rg_exc_tonic,nn.exc_tonic_count,neuron_to_sample_rg1_ton)
	v_m2,t_m2 = rg2.read_membrane_potential(rg2.mm_rg_exc_tonic,nn.exc_tonic_count,neuron_to_sample_rg2_ton)
	
	pylab.figure()
	pylab.subplot(211)
	pylab.plot(t_m1_irr,v_m1_irr)
	pylab.title('Individual Neuron Membrane Potential')
	pylab.subplot(212)
	pylab.plot(t_m2_irr,v_m2_irr)
	pylab.xlabel('Time (ms)')
	pylab.ylabel('Membrane potential (mV)')
	if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential_bursting.png',bbox_inches="tight")

	pylab.figure()
	pylab.subplot(211)
	pylab.plot(t_m1,v_m1)
	pylab.title('Individual Neuron Membrane Potential')
	pylab.subplot(212)
	pylab.plot(t_m2,v_m2)
	pylab.xlabel('Time (ms)')
	pylab.ylabel('Membrane potential (mV)')
	if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential_tonic.png',bbox_inches="tight")
#pylab.show()
