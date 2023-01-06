# hybrid-cpg
NEST simulation of a sparse firing CPG, NEST version 3.3

To run this simulation, all of the files must be in the same folder and NEST 3.3 must be installed. Run the simulation using the command:

python3 create_cpg.py

File overview:
configuration_run_nest.yaml - file to quickly access and change network characteristics for testing
loop_script.sh - bash script to loop the python script to run specified number of simulations in a row using the same parameters

create_cpg.py - the main python script which calls the other scripts to build a CPG
connect_populations.py - connects specified populations in order to create different architectures
create_bsg.py - creates a single rhythm-generating population
create_inh_pop.py - creates a single inhibitory population
phase_ordering.py - compares single neuron activity to mean population activity and orders the output by phase
set_network_params.py - sets the general network parameters, some of these parameters are taken from the yaml file, others must be set from within this script
start_simulation.py - starts the NEST simulation
