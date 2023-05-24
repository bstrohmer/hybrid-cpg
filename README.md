# hybrid-cpg
NEST simulation of a sparse firing CPG

To run this simulation, all of the files must be in the same folder and NEST version 3.4 must be installed. Run the simulation using the command:

python3 create_cpg.py

<b>File overview:</b><br>
configuration_run_nest.yaml - file to quickly access and change network characteristics for testing, the inidividual analyses can also be selected from this file by turning on the relevant plot(s)<br>
membrane_potential_plot: This will produce a membrane potential plot of a tonic and burstig neuron from each of the RG populations.<br>
pca_plot: This creates both 2D and 3D PCA plots with accompanying analysis data printed to the screen.<br>
phase_ordered_plot: These are the firing rate plots shown as panels of the corresponding unsorted and sorted plots.<br>
raster_plot: These plots show the spikes produced per neuron as individual points.<br>
rate_coded_plot: This plot using a sliding time window and smoothing to produce an analog output from each of the RGs.<br>
spike_distribution_plot: This is a line graph to show how many neurons spike 'x' amount of times, for example, the number of neurons spiking 2 times.<br>

loop_script.sh - bash script to loop the python script to run specified number of simulations in a row using the same parameters<br>

create_cpg.py - the main python script which calls the other scripts to build a CPG<br>
connect_populations.py - connects specified populations in order to create different architectures, contains balance calculation functions<br>
create_rg.py - creates a single rhythm-generating population<br>
create_inh_pop.py - creates a single inhibitory population<br>
phase_ordering.py - compares single neuron activity to mean population activity and orders the output by phase<br>
pca.py - runs a PCA on the convolved spiking activity from the individual RGs and the overall network output<br>
excitations_file_gen.py - creates the file for import to OpenSim for feed-forward control of a musculoskeletal model<br>
set_network_params.py - sets the general network parameters, some of these parameters are taken from the yaml file, others must be set from within this script<br>
start_simulation.py - starts the NEST simulation, select number of cores to run the simulation on - this must less than or equal to the number of CPU cores available on your computer<br>

<b>Testing the parameters from the paper:</b><br>
P1 - Ratio of excitatory / inhibitory neurons in the RG - update the value for "ratio_exc_inh" in the yaml file. Ex. 5 means 5:1 ratio or 20% inhibitory<br>
P2 - Connectivity (RG) - update the value for "sparsity_rg" in the yaml file. Ex. 0.03 means 3% connectivity within and from each RG population<br>
P2 - Connectivity (Inh) - update the value for "sparsity_cpg" in the yaml file. Ex. 0.09 means 9% connectivity from each Inhibitory population<br>
P3 - Neuronal sub-type (bursting) - update the value for "exc_pct_tonic" and "inh_pct_tonic" in the yaml file. Ex. 0.7 means 70% of the neurons within a sub-population are tonically firing and 30% are bursting<br>
P4 - Network balance - update the value for "w_exc_multiplier" in the yaml file. Decreasing the default value will bias the balance more inhibitory and increasing will bias it to be more excitatory.<br>
P5 - Noise (to tonic firing neurons) - update the value for "noise_amplitude_tonic" in the yaml file. Ex. 320 means that the current noise injected into each neuron has a standard deviation of 320pA<br>
P5 - Noise (to bursting neurons) - update the value for "noise_amplitude_irregular" in the yaml file. Ex. 160 means that the current noise injected into each neuron has a standard deviation of 160pA<br>

All other network parameters can be updated in the set_network_params.py file if the network needs to be tuned to a specific application.
