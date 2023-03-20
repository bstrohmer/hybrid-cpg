# hybrid-cpg
NEST simulation of a sparse firing CPG

To run this simulation, all of the files must be in the same folder and NEST version 3.4 must be installed. Run the simulation using the command:

python3 create_cpg.py

File overview:<br>
configuration_run_nest.yaml - file to quickly access and change network characteristics for testing<br>
loop_script.sh - bash script to loop the python script to run specified number of simulations in a row using the same parameters<br>

create_cpg.py - the main python script which calls the other scripts to build a CPG<br>
connect_populations.py - connects specified populations in order to create different architectures<br>
create_rg.py - creates a single rhythm-generating population<br>
create_inh_pop.py - creates a single inhibitory population<br>
phase_ordering.py - compares single neuron activity to mean population activity and orders the output by phase<br>
pca.py - runs a PCA on the convolved spiking activity from the individual RGs and the overall network output<br>
set_network_params.py - sets the general network parameters, some of these parameters are taken from the yaml file, others must be set from within this script<br>
start_simulation.py - starts the NEST simulation<br>

Testing the parameters from the paper:
P1 - Ratio of excitatory / inhibitory neurons in the RG - update the value for "ratio_exc_inh" in the yaml file. Ex. 5 means 5:1 ratio or 20% inhibitory<br>
P2 - Connectivity (RG) - update the value for "sparsity_rg" in the yaml file. Ex. 0.03 means 3% connectivity within and from each RG population<br>
P2 - Connectivity (Inh) - update the value for "sparsity_cpg" in the yaml file. Ex. 0.09 means 9% connectivity from each Inhibitory population<br>
P3 - Neuronal sub-type (bursting) - update the value for "exc_pct_tonic" and "inh_pct_tonic" in the yaml file. Ex. 0.7 means 70% of the neurons within a sub-population are tonically firing and 30% are bursting<br>
P4 - Network balance - update the weight initialization for the synapses, this can be altered using the parameters "w_exc_initial" or "w_inh_initial" in the set_network_params.py file<br>
P5 - Noise (to tonic firing neurons) - update the value for "noise_amplitude_tonic" in the yaml file. Ex. 320 means that the current noise injected into each neuron has a standard deviation of 320pA<br>
P5 - Noise (to bursting neurons) - update the value for "noise_amplitude_irregular" in the yaml file. Ex. 160 means that the current noise injected into each neuron has a standard deviation of 160pA<br>
