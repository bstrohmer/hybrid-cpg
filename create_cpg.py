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
from scipy.signal import find_peaks

conn=connect() 
ss.nest_start()
nn=netparams.neural_network()

#Create neuron populations - NEST
rg1 = rg.create_rg_population()
rg2 = rg.create_rg_population()

inh1 = inh.create_inh_population()
inh2 = inh.create_inh_population()

#Connect excitatory rg neruons to inhibitory populations
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
'''
#Test by connecting RG populations directly using excitation and inhibition
conn.create_connections(rg1.rg_exc_irregular,rg2.rg_inh_irregular,'exc')
conn.create_connections(rg1.rg_exc_irregular,rg2.rg_inh_tonic,'exc')
conn.create_connections(rg1.rg_exc_tonic,rg2.rg_inh_irregular,'exc')
conn.create_connections(rg1.rg_exc_tonic,rg2.rg_inh_tonic,'exc')

conn.create_connections(rg1.rg_inh_irregular,rg2.rg_exc_irregular,'inh')
conn.create_connections(rg1.rg_inh_irregular,rg2.rg_exc_tonic,'inh')
conn.create_connections(rg1.rg_inh_tonic,rg2.rg_exc_irregular,'inh')
conn.create_connections(rg1.rg_inh_tonic,rg2.rg_exc_tonic,'inh')

conn.create_connections(rg2.rg_exc_irregular,rg1.rg_inh_irregular,'exc')
conn.create_connections(rg2.rg_exc_irregular,rg1.rg_inh_tonic,'exc')
conn.create_connections(rg2.rg_exc_tonic,rg1.rg_inh_irregular,'exc')
conn.create_connections(rg2.rg_exc_tonic,rg1.rg_inh_tonic,'exc')

conn.create_connections(rg2.rg_inh_irregular,rg1.rg_exc_irregular,'inh')
conn.create_connections(rg2.rg_inh_irregular,rg1.rg_exc_tonic,'inh')
conn.create_connections(rg2.rg_inh_tonic,rg1.rg_exc_irregular,'inh')
conn.create_connections(rg2.rg_inh_tonic,rg1.rg_exc_tonic,'inh')
'''

print("Seed#: ",nn.rng_seed)
print("# exc (irregular, tonic): ",nn.exc_irregular_count,nn.exc_tonic_count,"; # inh(irregular, tonic): ",nn.inh_irregular_count,nn.inh_tonic_count,"; # inh buffer: ",nn.inh_pop_neurons)

#Calculate synaptic balance of rg populations
conn.calculate_balance(rg1.rg_exc_irregular,rg1.rg_exc_irregular,'exc')
conn.calculate_balance(rg1.rg_exc_irregular,rg1.rg_exc_tonic,'exc')
conn.calculate_balance(rg1.rg_exc_irregular,rg1.rg_inh_irregular,'exc')
conn.calculate_balance(rg1.rg_exc_irregular,rg1.rg_inh_tonic,'exc')
conn.calculate_balance(rg1.rg_inh_irregular,rg1.rg_inh_irregular,'inh')
conn.calculate_balance(rg1.rg_inh_irregular,rg1.rg_inh_tonic,'inh')
conn.calculate_balance(rg1.rg_inh_irregular,rg1.rg_exc_irregular,'inh')
conn.calculate_balance(rg1.rg_inh_irregular,rg1.rg_exc_tonic,'inh')
conn.calculate_balance(rg1.rg_exc_tonic,rg1.rg_exc_irregular,'exc')
conn.calculate_balance(rg1.rg_exc_tonic,rg1.rg_exc_tonic,'exc')
conn.calculate_balance(rg1.rg_exc_tonic,rg1.rg_inh_irregular,'exc')
conn.calculate_balance(rg1.rg_exc_tonic,rg1.rg_inh_tonic,'exc')
conn.calculate_balance(rg1.rg_inh_tonic,rg1.rg_inh_irregular,'inh')
conn.calculate_balance(rg1.rg_inh_tonic,rg1.rg_inh_tonic,'inh')
conn.calculate_balance(rg1.rg_inh_tonic,rg1.rg_exc_irregular,'inh')
rg1_balance_pct = conn.calculate_balance(rg1.rg_inh_tonic,rg1.rg_exc_tonic,'inh')

#print('RG1 balance %: ',rg1_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
#conn.reset_balance()

conn.calculate_balance(rg2.rg_exc_irregular,rg2.rg_exc_irregular,'exc')
conn.calculate_balance(rg2.rg_exc_irregular,rg2.rg_exc_tonic,'exc')
conn.calculate_balance(rg2.rg_exc_irregular,rg2.rg_inh_irregular,'exc')
conn.calculate_balance(rg2.rg_exc_irregular,rg2.rg_inh_tonic,'exc')
conn.calculate_balance(rg2.rg_inh_irregular,rg2.rg_inh_irregular,'inh')
conn.calculate_balance(rg2.rg_inh_irregular,rg2.rg_inh_tonic,'inh')
conn.calculate_balance(rg2.rg_inh_irregular,rg2.rg_exc_irregular,'inh')
conn.calculate_balance(rg2.rg_inh_irregular,rg2.rg_exc_tonic,'inh')
conn.calculate_balance(rg2.rg_exc_tonic,rg2.rg_exc_irregular,'exc')
conn.calculate_balance(rg2.rg_exc_tonic,rg2.rg_exc_tonic,'exc')
conn.calculate_balance(rg2.rg_exc_tonic,rg2.rg_inh_irregular,'exc')
conn.calculate_balance(rg2.rg_exc_tonic,rg2.rg_inh_tonic,'exc')
conn.calculate_balance(rg2.rg_inh_tonic,rg2.rg_inh_irregular,'inh')
conn.calculate_balance(rg2.rg_inh_tonic,rg2.rg_inh_tonic,'inh')
conn.calculate_balance(rg2.rg_inh_tonic,rg2.rg_exc_irregular,'inh')
rg2_balance_pct = conn.calculate_balance(rg2.rg_inh_tonic,rg2.rg_exc_tonic,'inh')

conn.calculate_balance(inh1.inh_pop,rg2.rg_exc_irregular,'inh_strong')
conn.calculate_balance(inh1.inh_pop,rg2.rg_exc_tonic,'inh_strong')
conn.calculate_balance(inh1.inh_pop,rg2.rg_inh_irregular,'inh_strong')
conn.calculate_balance(inh1.inh_pop,rg2.rg_inh_tonic,'inh_strong')

conn.calculate_balance(inh2.inh_pop,rg1.rg_exc_irregular,'inh_strong')
conn.calculate_balance(inh2.inh_pop,rg1.rg_exc_tonic,'inh_strong')
conn.calculate_balance(inh2.inh_pop,rg1.rg_inh_irregular,'inh_strong')
rg_balance_pct = conn.calculate_balance(inh2.inh_pop,rg1.rg_inh_tonic,'inh_strong')


#print('RG2 balance %: ',rg2_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
print('CPG balance %: ',rg_balance_pct,' >0 skew excitatory; <0 skew inhibitory')

nest.Simulate(nn.sim_time)

#Read membrane potential of an irregularly firing neuron - neuron number <= population size
neuron_to_sample = 863
v_m1_irr,t_m1_irr = rg1.read_membrane_potential(rg1.mm_rg_exc_irregular,nn.exc_irregular_count,neuron_to_sample)
v_m2_irr,t_m2_irr = rg2.read_membrane_potential(rg2.mm_rg_exc_irregular,nn.exc_irregular_count,neuron_to_sample)

#Read membrane potential of a tonically firing neuron - neuron number <= population size
neuron_to_sample = 150
v_m1,t_m1 = rg1.read_membrane_potential(rg1.mm_rg_exc_tonic,nn.exc_tonic_count,neuron_to_sample)
v_m2,t_m2 = rg2.read_membrane_potential(rg2.mm_rg_exc_tonic,nn.exc_tonic_count,neuron_to_sample)

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
senders_inhpop1,spiketimes_inhpop1 = inh1.read_spike_data(inh1.spike_detector_inh)
senders_inhpop2,spiketimes_inhpop2 = inh2.read_spike_data(inh2.spike_detector_inh)

#Count spikes per neuron
indiv_spikes_exc1 = rg1.count_indiv_spikes(nn.exc_irregular_count,senders_exc1)
indiv_spikes_inh1 = rg1.count_indiv_spikes(nn.inh_irregular_count,senders_inh1)
indiv_spikes_exc_tonic1 = rg1.count_indiv_spikes(nn.exc_tonic_count,senders_exc_tonic1)
indiv_spikes_inh_tonic1 = rg1.count_indiv_spikes(nn.inh_tonic_count,senders_inh_tonic1)

indiv_spikes_exc2 = rg2.count_indiv_spikes(nn.exc_irregular_count,senders_exc2)
indiv_spikes_inh2 = rg2.count_indiv_spikes(nn.inh_irregular_count,senders_inh2)
indiv_spikes_exc_tonic2 = rg2.count_indiv_spikes(nn.exc_tonic_count,senders_exc_tonic2)
indiv_spikes_inh_tonic2 = rg2.count_indiv_spikes(nn.inh_tonic_count,senders_inh_tonic2)

indiv_spikes_inhpop1 = inh1.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop1)
indiv_spikes_inhpop2 = inh2.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop2)

all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_tonic1+indiv_spikes_inh_tonic1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_tonic2+indiv_spikes_inh_tonic2+indiv_spikes_inhpop1+indiv_spikes_inhpop2
print('The maximum amount of spikes by a single neuron is: ',max(all_indiv_spike_counts))
spike_distribution=[]
for i in range(max(all_indiv_spike_counts)):
    spike_distribution.append(all_indiv_spike_counts.count(i))
print('The spike distribution peak is at ',max(spike_distribution[2:]),' neurons, having ', np.argmax(spike_distribution[2:]),' spikes.')

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

#Convolve spike data - rg populations
inh1_convolved = inh1.convolve_spiking_activity(nn.inh_pop_neurons,spiketimes_inhpop1)
inh2_convolved = inh2.convolve_spiking_activity(nn.inh_pop_neurons,spiketimes_inhpop2)

spikes_convolved_complete_network = np.vstack([spikes_convolved_all1,spikes_convolved_all2])
spikes_convolved_complete_network = np.vstack([spikes_convolved_complete_network,inh1_convolved])
spikes_convolved_complete_network = np.vstack([spikes_convolved_complete_network,inh2_convolved])

if nn.remove_silent:
    print('Removing silent neurons')
    spikes_convolved_all1 = spikes_convolved_all1[~np.all(spikes_convolved_all1 == 0, axis=1)]
    spikes_convolved_all2 = spikes_convolved_all2[~np.all(spikes_convolved_all2 == 0, axis=1)]
    spikes_convolved_complete_network = spikes_convolved_complete_network[~np.all(spikes_convolved_complete_network == 0, axis=1)]

print('Convolved spiking activity complete')

#Create Rate Coded Output - rg populations
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

spike_bins_inh1 = inh1.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop1)
spike_bins_inh2 = inh2.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop2)
spike_bins_inh = spike_bins_inh1+spike_bins_inh2

spike_bins_rgs = spike_bins_rg1+spike_bins_rg2
spike_bins_all_pops = spike_bins_rg1+spike_bins_rg2+spike_bins_inh
print('rg1 peaks ',find_peaks(spike_bins_rg1,height=1500,prominence=70)[0])
print('rg2 peaks ',find_peaks(spike_bins_rg2,height=1500,prominence=70)[0])
print('Rate coded activity complete')

#Plot phase sorted activity
order_by_phase(spikes_convolved_all1, spike_bins_rg1, 'rg1', remove_mean = True, high_pass_filtered = True, generate_plot = True)
order_by_phase(spikes_convolved_all2, spike_bins_rg2, 'rg2', remove_mean = True, high_pass_filtered = True, generate_plot = True)
order_by_phase(spikes_convolved_complete_network, spike_bins_rgs, 'all_pops', remove_mean = True, high_pass_filtered = True, generate_plot = True) #UPDATED - compare summed output from rgs to all spikes in network (inh pops output is "absorbed" into rg so it does not directly contribute to the rate-coded output of the network)
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
pylab.title('Spike Output rg1')
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

#Plot spiking output
t = np.arange(0,nn.sim_time,nn.time_resolution)

pylab.figure()
pylab.subplot(211)
for i in range(nn.inh_pop_neurons-1): 
    pylab.plot(spiketimes_inhpop1[0][i],senders_inhpop1[0][i],'.',label='Inh Pop1')
pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron #')
pylab.title('Spike Output Inh1')
pylab.subplot(212)
for i in range(nn.inh_pop_neurons-1): 
    pylab.plot(spiketimes_inhpop2[0][i],senders_inhpop2[0][i],'.',label='Inh Pop2')
pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron #')
#pylab.title('Spike Output Inh2')
pylab.subplots_adjust(bottom=0.15)
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spikes_inh.png',bbox_inches="tight")

pylab.figure()
pylab.subplot(211)
pylab.plot(t[100:-100],spike_bins_rg1[100:-100],label='RG1')
pylab.plot(t[100:-100],spike_bins_rg2[100:-100],label='RG2')
plt.legend( bbox_to_anchor=(1.1,1.05))
pylab.xlabel('Time (ms)')
pylab.ylabel('Spike Count')
pylab.title('Rate-coded Output per RG')
pylab.subplot(212)
pylab.plot(t[100:-100],spike_bins_rgs[100:-100],label='CPG')	#UPDATED - only showing sum of rate-coded output of rgs
plt.legend( bbox_to_anchor=(1.1,1.05))
pylab.xlabel('Time (ms)')
pylab.ylabel('Spike Count')
#pylab.title('Rate-coded Output CPG')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.png',bbox_inches="tight")

counts, bins = np.histogram(spike_distribution)
pylab.figure()
pylab.stairs(counts,bins)  #UPDATED - PLOT A HISTOGRAM
pylab.xticks(bins,rotation=30, horizontalalignment='right')
pylab.xlabel('Spike Count')
pylab.ylabel('Number of Neurons')
pylab.title('Spike Distribution')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution_hist.png',bbox_inches="tight")
pylab.figure()
pylab.plot(spike_distribution[2:])
pylab.xscale('log')
pylab.xlabel('Total Spike Count')
pylab.ylabel('Number of Neurons')
pylab.title('Spike Distribution')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spike_distribution.png',bbox_inches="tight")

pylab.figure()
pylab.subplot(211)
pylab.plot(t_m1_irr,v_m1_irr)
#pylab.xlabel('Time (ms)')
#pylab.ylabel('Membrane potential (mV)')
pylab.title('Individual Neuron Membrane Potential')
pylab.subplot(212)
pylab.plot(t_m2_irr,v_m2_irr)
pylab.xlabel('Time (ms)')
pylab.ylabel('Membrane potential (mV)')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential_irregular.png',bbox_inches="tight")

pylab.figure()
pylab.subplot(211)
pylab.plot(t_m1,v_m1)
#pylab.xlabel('Time (ms)')
#pylab.ylabel('Membrane potential (mV)')
pylab.title('Individual Neuron Membrane Potential')
pylab.subplot(212)
pylab.plot(t_m2,v_m2)
pylab.xlabel('Time (ms)')
pylab.ylabel('Membrane potential (mV)')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential_tonic.png',bbox_inches="tight")
#pylab.show()

