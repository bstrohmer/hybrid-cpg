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
import create_bsg as bsg
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
bsg1 = bsg.create_bsg_population()
bsg2 = bsg.create_bsg_population()

inh1 = inh.create_inh_population()
inh2 = inh.create_inh_population()
'''
#Connect excitatory BSG neruons to inhibitory populations
conn.create_connections(bsg1.bsg_exc,inh1.inh_pop,'exc')  #EXCITATORY CONNECTIONS UPDATED TO SAME SPARSITY AS POST-SYNAPTIC FROM INH POPS
conn.create_connections(bsg1.bsg_exc_burst,inh1.inh_pop,'exc')
conn.create_connections(bsg2.bsg_exc,inh2.inh_pop,'exc')
conn.create_connections(bsg2.bsg_exc_burst,inh2.inh_pop,'exc')

#Connect inhibitory populations to all BSG neurons
conn.create_connections(inh1.inh_pop,bsg2.bsg_exc,'inh_strong')
conn.create_connections(inh1.inh_pop,bsg2.bsg_exc_burst,'inh_strong')
conn.create_connections(inh1.inh_pop,bsg2.bsg_inh,'inh_strong')
conn.create_connections(inh1.inh_pop,bsg2.bsg_inh_burst,'inh_strong')

conn.create_connections(inh2.inh_pop,bsg1.bsg_exc,'inh_strong')
conn.create_connections(inh2.inh_pop,bsg1.bsg_exc_burst,'inh_strong')
conn.create_connections(inh2.inh_pop,bsg1.bsg_inh,'inh_strong')
conn.create_connections(inh2.inh_pop,bsg1.bsg_inh_burst,'inh_strong')
'''
#Test by connecting RG populations directly using excitation and inhibition
conn.create_connections(bsg1.bsg_exc,bsg2.bsg_inh,'exc')
conn.create_connections(bsg1.bsg_exc,bsg2.bsg_inh_burst,'exc')
conn.create_connections(bsg1.bsg_exc_burst,bsg2.bsg_inh,'exc')
conn.create_connections(bsg1.bsg_exc_burst,bsg2.bsg_inh_burst,'exc')

conn.create_connections(bsg1.bsg_inh,bsg2.bsg_exc,'inh')
conn.create_connections(bsg1.bsg_inh,bsg2.bsg_exc_burst,'inh')
conn.create_connections(bsg1.bsg_inh_burst,bsg2.bsg_exc,'inh')
conn.create_connections(bsg1.bsg_inh_burst,bsg2.bsg_exc_burst,'inh')

conn.create_connections(bsg2.bsg_exc,bsg1.bsg_inh,'exc')
conn.create_connections(bsg2.bsg_exc,bsg1.bsg_inh_burst,'exc')
conn.create_connections(bsg2.bsg_exc_burst,bsg1.bsg_inh,'exc')
conn.create_connections(bsg2.bsg_exc_burst,bsg1.bsg_inh_burst,'exc')

conn.create_connections(bsg2.bsg_inh,bsg1.bsg_exc,'inh')
conn.create_connections(bsg2.bsg_inh,bsg1.bsg_exc_burst,'inh')
conn.create_connections(bsg2.bsg_inh_burst,bsg1.bsg_exc,'inh')
conn.create_connections(bsg2.bsg_inh_burst,bsg1.bsg_exc_burst,'inh')


print("Seed#: ",nn.rng_seed)
print("# exc (tonic, bursting): ",nn.exc_tonic_count,nn.exc_burst_count,"; # inh(tonic, bursting): ",nn.inh_tonic_count,nn.inh_burst_count,"; # inh buffer: ",nn.inh_pop_neurons)
'''
#Calculate synaptic balance of BSG populations
conn.calculate_balance(bsg1.bsg_exc,bsg1.bsg_exc,'exc')
conn.calculate_balance(bsg1.bsg_exc,bsg1.bsg_exc_burst,'exc')
conn.calculate_balance(bsg1.bsg_exc,bsg1.bsg_inh,'exc')
conn.calculate_balance(bsg1.bsg_exc,bsg1.bsg_inh_burst,'exc')
conn.calculate_balance(bsg1.bsg_inh,bsg1.bsg_inh,'inh')
conn.calculate_balance(bsg1.bsg_inh,bsg1.bsg_inh_burst,'inh')
conn.calculate_balance(bsg1.bsg_inh,bsg1.bsg_exc,'inh')
conn.calculate_balance(bsg1.bsg_inh,bsg1.bsg_exc_burst,'inh')
conn.calculate_balance(bsg1.bsg_exc_burst,bsg1.bsg_exc,'exc')
conn.calculate_balance(bsg1.bsg_exc_burst,bsg1.bsg_exc_burst,'exc')
conn.calculate_balance(bsg1.bsg_exc_burst,bsg1.bsg_inh,'exc')
conn.calculate_balance(bsg1.bsg_exc_burst,bsg1.bsg_inh_burst,'exc')
conn.calculate_balance(bsg1.bsg_inh_burst,bsg1.bsg_inh,'inh')
conn.calculate_balance(bsg1.bsg_inh_burst,bsg1.bsg_inh_burst,'inh')
conn.calculate_balance(bsg1.bsg_inh_burst,bsg1.bsg_exc,'inh')
bsg1_balance_pct = conn.calculate_balance(bsg1.bsg_inh_burst,bsg1.bsg_exc_burst,'inh')

#print('RG1 balance %: ',bsg1_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
#conn.reset_balance()

conn.calculate_balance(bsg2.bsg_exc,bsg2.bsg_exc,'exc')
conn.calculate_balance(bsg2.bsg_exc,bsg2.bsg_exc_burst,'exc')
conn.calculate_balance(bsg2.bsg_exc,bsg2.bsg_inh,'exc')
conn.calculate_balance(bsg2.bsg_exc,bsg2.bsg_inh_burst,'exc')
conn.calculate_balance(bsg2.bsg_inh,bsg2.bsg_inh,'inh')
conn.calculate_balance(bsg2.bsg_inh,bsg2.bsg_inh_burst,'inh')
conn.calculate_balance(bsg2.bsg_inh,bsg2.bsg_exc,'inh')
conn.calculate_balance(bsg2.bsg_inh,bsg2.bsg_exc_burst,'inh')
conn.calculate_balance(bsg2.bsg_exc_burst,bsg2.bsg_exc,'exc')
conn.calculate_balance(bsg2.bsg_exc_burst,bsg2.bsg_exc_burst,'exc')
conn.calculate_balance(bsg2.bsg_exc_burst,bsg2.bsg_inh,'exc')
conn.calculate_balance(bsg2.bsg_exc_burst,bsg2.bsg_inh_burst,'exc')
conn.calculate_balance(bsg2.bsg_inh_burst,bsg2.bsg_inh,'inh')
conn.calculate_balance(bsg2.bsg_inh_burst,bsg2.bsg_inh_burst,'inh')
conn.calculate_balance(bsg2.bsg_inh_burst,bsg2.bsg_exc,'inh')
bsg2_balance_pct = conn.calculate_balance(bsg2.bsg_inh_burst,bsg2.bsg_exc_burst,'inh')

conn.calculate_balance(inh1.inh_pop,bsg2.bsg_exc,'inh_strong')
conn.calculate_balance(inh1.inh_pop,bsg2.bsg_exc_burst,'inh_strong')
conn.calculate_balance(inh1.inh_pop,bsg2.bsg_inh,'inh_strong')
conn.calculate_balance(inh1.inh_pop,bsg2.bsg_inh_burst,'inh_strong')

conn.calculate_balance(inh2.inh_pop,bsg1.bsg_exc,'inh_strong')
conn.calculate_balance(inh2.inh_pop,bsg1.bsg_exc_burst,'inh_strong')
conn.calculate_balance(inh2.inh_pop,bsg1.bsg_inh,'inh_strong')
bsg_balance_pct = conn.calculate_balance(inh2.inh_pop,bsg1.bsg_inh_burst,'inh_strong')


#print('RG2 balance %: ',bsg2_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
print('CPG balance %: ',bsg_balance_pct,' >0 skew excitatory; <0 skew inhibitory')
'''
nest.Simulate(nn.sim_time)

#Read membrane potential
v_m,t_m = bsg1.read_membrane_potential(bsg1.mm_bsg_exc_burst,nn.exc_burst_count)
df = pd.DataFrame({"Time (ms)" : t_m, "Membrane potential (mV)" : v_m})
df.to_csv(nn.pathFigures + '/' + 'v_m_rg1.csv', index=False)
#df.to_csv("/home/bs/git_repos/BSG_network/CPG_network_files/v_m_rg1.csv", index=False)
v_m,t_m = bsg2.read_membrane_potential(bsg2.mm_bsg_exc_burst,nn.exc_burst_count)
df = pd.DataFrame({"Time (ms)" : t_m, "Membrane potential (mV)" : v_m})
df.to_csv(nn.pathFigures + '/' + 'v_m_rg2.csv', index=False)

spike_count_array = []
#Read spike data - BSG populations
senders_exc1,spiketimes_exc1 = bsg1.read_spike_data(bsg1.spike_detector_bsg_exc)
senders_inh1,spiketimes_inh1 = bsg1.read_spike_data(bsg1.spike_detector_bsg_inh)
senders_exc_burst1,spiketimes_exc_burst1 = bsg1.read_spike_data(bsg1.spike_detector_bsg_exc_burst)
senders_inh_burst1,spiketimes_inh_burst1 = bsg1.read_spike_data(bsg1.spike_detector_bsg_inh_burst)

senders_exc2,spiketimes_exc2 = bsg2.read_spike_data(bsg2.spike_detector_bsg_exc)
senders_inh2,spiketimes_inh2 = bsg2.read_spike_data(bsg2.spike_detector_bsg_inh)
senders_exc_burst2,spiketimes_exc_burst2 = bsg2.read_spike_data(bsg2.spike_detector_bsg_exc_burst)
senders_inh_burst2,spiketimes_inh_burst2 = bsg2.read_spike_data(bsg2.spike_detector_bsg_inh_burst)

#Read spike data - inhibitory populations
senders_inhpop1,spiketimes_inhpop1 = inh1.read_spike_data(inh1.spike_detector_inh)
senders_inhpop2,spiketimes_inhpop2 = inh2.read_spike_data(inh2.spike_detector_inh)

#Count spikes per neuron
indiv_spikes_exc1 = bsg1.count_indiv_spikes(nn.exc_tonic_count,senders_exc1)
indiv_spikes_inh1 = bsg1.count_indiv_spikes(nn.inh_tonic_count,senders_inh1)
indiv_spikes_exc_burst1 = bsg1.count_indiv_spikes(nn.exc_burst_count,senders_exc_burst1)
indiv_spikes_inh_burst1 = bsg1.count_indiv_spikes(nn.inh_burst_count,senders_inh_burst1)

indiv_spikes_exc2 = bsg2.count_indiv_spikes(nn.exc_tonic_count,senders_exc2)
indiv_spikes_inh2 = bsg2.count_indiv_spikes(nn.inh_tonic_count,senders_inh2)
indiv_spikes_exc_burst2 = bsg2.count_indiv_spikes(nn.exc_burst_count,senders_exc_burst2)
indiv_spikes_inh_burst2 = bsg2.count_indiv_spikes(nn.inh_burst_count,senders_inh_burst2)

indiv_spikes_inhpop1 = inh1.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop1)
indiv_spikes_inhpop2 = inh2.count_indiv_spikes(nn.inh_pop_neurons,senders_inhpop2)

all_indiv_spike_counts=indiv_spikes_exc1+indiv_spikes_inh1+indiv_spikes_exc_burst1+indiv_spikes_inh_burst1+indiv_spikes_exc2+indiv_spikes_inh2+indiv_spikes_exc_burst2+indiv_spikes_inh_burst2+indiv_spikes_inhpop1+indiv_spikes_inhpop2
print('The maximum amount of spikes by a single neuron is: ',max(all_indiv_spike_counts))
spike_distribution=[]
for i in range(max(all_indiv_spike_counts)):
    spike_distribution.append(all_indiv_spike_counts.count(i))
print('The spike distribution peak is at ',max(spike_distribution[2:]),' neurons, having ', np.argmax(spike_distribution[2:]),' spikes.')

#Convolve spike data - BSG populations
bsg_exc_convolved1 = bsg1.convolve_spiking_activity(nn.exc_tonic_count,spiketimes_exc1)
bsg_exc_burst_convolved1 = bsg1.convolve_spiking_activity(nn.exc_burst_count,spiketimes_exc_burst1)
bsg_inh_convolved1 = bsg1.convolve_spiking_activity(nn.inh_tonic_count,spiketimes_inh1)
bsg_inh_burst_convolved1 = bsg1.convolve_spiking_activity(nn.inh_burst_count,spiketimes_inh_burst1)
spikes_convolved_all1 = np.vstack([bsg_exc_convolved1,bsg_inh_convolved1])
spikes_convolved_all1 = np.vstack([spikes_convolved_all1,bsg_exc_burst_convolved1])
spikes_convolved_all1 = np.vstack([spikes_convolved_all1,bsg_inh_burst_convolved1])

bsg_exc_convolved2 = bsg2.convolve_spiking_activity(nn.exc_tonic_count,spiketimes_exc2)
bsg_exc_burst_convolved2 = bsg2.convolve_spiking_activity(nn.exc_burst_count,spiketimes_exc_burst2)
bsg_inh_convolved2 = bsg2.convolve_spiking_activity(nn.inh_tonic_count,spiketimes_inh2)
bsg_inh_burst_convolved2 = bsg2.convolve_spiking_activity(nn.inh_burst_count,spiketimes_inh_burst2)
spikes_convolved_all2 = np.vstack([bsg_exc_convolved2,bsg_inh_convolved2])
spikes_convolved_all2 = np.vstack([spikes_convolved_all2,bsg_exc_burst_convolved2])
spikes_convolved_all2 = np.vstack([spikes_convolved_all2,bsg_inh_burst_convolved2])

#Convolve spike data - BSG populations
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

#Create Rate Coded Output - BSG populations
spike_bins_bsg_exc1 = bsg1.rate_code_spikes(nn.exc_tonic_count,spiketimes_exc1)
spike_bins_bsg_inh1 = bsg1.rate_code_spikes(nn.inh_tonic_count,spiketimes_inh1)
spike_bins_bsg_exc_burst1 = bsg1.rate_code_spikes(nn.exc_burst_count,spiketimes_exc_burst1)
spike_bins_bsg_inh_burst1 = bsg1.rate_code_spikes(nn.inh_burst_count,spiketimes_inh_burst1)
spike_bins_bsg1 = spike_bins_bsg_exc1+spike_bins_bsg_exc_burst1+spike_bins_bsg_inh1+spike_bins_bsg_inh_burst1

spike_bins_bsg_exc2 = bsg2.rate_code_spikes(nn.exc_tonic_count,spiketimes_exc2)
spike_bins_bsg_inh2 = bsg2.rate_code_spikes(nn.inh_tonic_count,spiketimes_inh2)
spike_bins_bsg_exc_burst2 = bsg2.rate_code_spikes(nn.exc_burst_count,spiketimes_exc_burst2)
spike_bins_bsg_inh_burst2 = bsg2.rate_code_spikes(nn.inh_burst_count,spiketimes_inh_burst2)
spike_bins_bsg2=spike_bins_bsg_exc2+spike_bins_bsg_exc_burst2+spike_bins_bsg_inh2+spike_bins_bsg_inh_burst2

spike_bins_inh1 = inh1.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop1)
spike_bins_inh2 = inh2.rate_code_spikes(nn.inh_pop_neurons,spiketimes_inhpop2)
spike_bins_inh = spike_bins_inh1+spike_bins_inh2

spike_bins_bsgs = spike_bins_bsg1+spike_bins_bsg2
spike_bins_all_pops = spike_bins_bsg1+spike_bins_bsg2+spike_bins_inh
print('BSG1 peaks ',find_peaks(spike_bins_bsg1,height=1500,prominence=70)[0])
print('BSG2 peaks ',find_peaks(spike_bins_bsg2,height=1500,prominence=70)[0])
print('Rate coded activity complete')

#Plot phase sorted activity
order_by_phase(spikes_convolved_all1, spike_bins_bsg1, 'bsg1', remove_mean = True, high_pass_filtered = True, generate_plot = True)
order_by_phase(spikes_convolved_all2, spike_bins_bsg2, 'bsg2', remove_mean = True, high_pass_filtered = True, generate_plot = True)
order_by_phase(spikes_convolved_complete_network, spike_bins_bsgs, 'all_pops', remove_mean = True, high_pass_filtered = True, generate_plot = True) #UPDATED - compare summed output from BSGs to all spikes in network (inh pops output is "absorbed" into BSG so it does not directly contribute to the rate-coded output of the network)

pylab.figure()
pylab.subplot(211)
for i in range(nn.exc_tonic_count-1): 
    pylab.plot(spiketimes_exc1[0][i],senders_exc1[0][i],'.',label='Exc')
for i in range(nn.exc_burst_count-1):
    if nn.exc_burst_count != 0: pylab.plot(spiketimes_exc_burst1[0][i],senders_exc_burst1[0][i],'.',label='Exc bursting')
for i in range(nn.inh_tonic_count-1):
    pylab.plot(spiketimes_inh1[0][i],senders_inh1[0][i],'.',label='Inh') #offset neuron number by total # of excitatory neurons
for i in range(nn.inh_burst_count-1):
    if nn.inh_burst_count != 0: pylab.plot(spiketimes_inh_burst1[0][i],senders_inh_burst1[0][i],'.',label='Inh bursting')
pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron #')
pylab.title('Spike Output BSG1')
pylab.subplot(212)
for i in range(nn.exc_tonic_count-1): 
    pylab.plot(spiketimes_exc2[0][i],senders_exc2[0][i],'.',label='Exc')
for i in range(nn.exc_burst_count-1):
    if nn.exc_burst_count != 0: pylab.plot(spiketimes_exc_burst2[0][i],senders_exc_burst2[0][i],'.',label='Exc bursting')
for i in range(nn.inh_tonic_count-1):
    pylab.plot(spiketimes_inh2[0][i],senders_inh2[0][i],'.',label='Inh') #offset neuron number by total # of excitatory neurons
for i in range(nn.inh_burst_count-1):
    if nn.inh_burst_count != 0: pylab.plot(spiketimes_inh_burst2[0][i],senders_inh_burst2[0][i],'.',label='Inh bursting')
pylab.xlabel('Time (ms)')
pylab.ylabel('Neuron #')
#pylab.title('Spike Output BSG2')
pylab.subplots_adjust(bottom=0.15)
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'spikes_bsg.png',bbox_inches="tight")

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
pylab.plot(t[100:-100],spike_bins_bsg1[100:-100],label='RG1')
pylab.plot(t[100:-100],spike_bins_bsg2[100:-100],label='RG2')
plt.legend( bbox_to_anchor=(1.1,1.05))
pylab.xlabel('Time (ms)')
pylab.ylabel('Spike Count')
pylab.title('Rate-coded Output per RG')
pylab.subplot(212)
pylab.plot(t[100:-100],spike_bins_bsgs[100:-100],label='CPG')	#UPDATED - only showing sum of rate-coded output of BSGs
plt.legend( bbox_to_anchor=(1.1,1.05))
pylab.xlabel('Time (ms)')
pylab.ylabel('Spike Count')
#pylab.title('Rate-coded Output CPG')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'rate_coded_output.png',bbox_inches="tight")

print('Spike distribution # of bins: ',len(spike_distribution[2:])/10)
pylab.figure()
pylab.hist(spike_distribution[2:],bins=int(len(spike_distribution[2:])/10))  #UPDATED - PLOT A HISTOGRAM
pylab.xlabel('Spike (Bins of 10)')
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
pylab.plot(t_m,v_m)
pylab.xlabel('Time (ms)')
pylab.ylabel('Membrane potential (mV)')
pylab.title('Individual Neuron Membrane Potential')
if nn.args['save_results']: plt.savefig(nn.pathFigures + '/' + 'membrane_potential.png',bbox_inches="tight")

#pylab.show()

