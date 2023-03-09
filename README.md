# hybrid-cpg
NEST simulation of a sparse firing CPG

To run this simulation, all of the files must be in the same folder and NEST version 3.3 must be installed. Run the simulation using the command:

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
start_simulation.py - starts the NEST simulation
