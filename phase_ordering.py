from scipy import signal
from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable #REQUIRED FOR COLORBAR
from set_network_params import neural_network
netparams = neural_network()

def order_by_phase(convolved_activity, population_mean_activity, pop_name, remove_mean = True, high_pass_filtered = True, generate_plot = True):
    """Compute phase ordering as described in "Movement is governed by rotational population dynamics in spinal motor networks, Linden et al. 2022"

    Args:
        convolved_activity (np.array): continuos firing activity resulting from convolving spikes (nb_neurons x simulation_time)
        population_mean_activity (np.array): mean population activity (1 x simulation_time)
        remove_mean (bool): if True, it removes the mean from each convolved activity. Defaults 
        high_pass_filtered (bool): if true, the activity of each neuron is high-pass filtered. Defaults to True.
        generate_plot (bool, optional): if True, generates plots. Defaults to True.

    Returns:
        sorted_convolved_activity (np.array): sorted continuos firing activity resulting from convolving spikes (nb_neurons x simulation_time)
        sorted_idx (list): index describing the phase ordering, it's return in case it's need to order something else like the raster plot.
    """
    #print('Lengths conv,ratecoded',np.shape(convolved_activity)[1],len(population_mean_activity))

    if remove_mean:
        population_mean_activity = (population_mean_activity - np.mean(population_mean_activity, axis=0))
    if high_pass_filtered:            
        b, a = butter(3, .1, 'highpass', fs=1000)		#high pass freq was previously 0.3Hz
        population_mean_activity = filtfilt(b, a, population_mean_activity)		#high pass filter the mean activity

    freqs_pop, psd_pop = signal.welch(population_mean_activity)
    peak_population_mean_activitypsd_freq = freqs_pop[np.where(psd_pop == psd_pop.max())[0][0]]

    phase_i = []
    psd_indiv = []
    count_min_sca = 0
    for convolved_activity_neuron_i in convolved_activity:
        # Cross spectral density or cross power spectrum of x,y.
        f, Pxy = signal.csd(convolved_activity_neuron_i,population_mean_activity)
        
        index_ = np.where(f == peak_population_mean_activitypsd_freq)[0][0]
        phase_activity = np.angle(Pxy)
        
        #freqs_indiv_neuron, psd_indiv_neuron = signal.welch(convolved_activity_neuron_i)
        
        phase_i.append(phase_activity[index_])
        #psd_indiv.append(psd_indiv_neuron)
    
    #min_sca = np.min(convolved_activity)   
    #count_min_sca = np.count_nonzero(convolved_activity == min_sca)
    #print('Convolved activity min, occurences', min_sca,count_min_sca) 
    #print('Sample psd output',psd_indiv_neuron)
    sorted_idx = sorted(range(len(phase_i)), key=lambda k:phase_i[k])
    sorted_convolved_activity = convolved_activity[sorted_idx]

    if generate_plot:
        figConv, axsConv = pyplot.subplots(2, figsize=(10,8))
        figConv.suptitle('Convolved Activity')
        im = axsConv[0].imshow(convolved_activity, aspect='auto', vmin=0, vmax=1)
        axsConv[0].set(title='Unsorted', ylabel='Neuron idx #')
        #********************ADD COLORBAR TO PLOT*****************************
        divider = make_axes_locatable(axsConv[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(im,cax=cax)
        
        im = axsConv[1].imshow(sorted_convolved_activity, aspect='auto', vmin=0, vmax=1)
        #axsConv[1].set(title='Phase sorted', xlabel='Time [ms]', ylabel='Neuron idx #')
        axsConv[1].set(xlabel='Time [ms]', ylabel='Neuron idx #')
        #********************ADD COLORBAR TO PLOT*****************************
        divider = make_axes_locatable(axsConv[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(im,cax=cax)
        if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'phase_sorted'+ pop_name +'.png',bbox_inches="tight")
        
        pyplot.figure()
        count, bins, ignored = pyplot.hist(phase_i,bins=25,density=True)
        average_bin_height = (sum(count)/len(count)) 
        diff_bin_height = [abs(x - average_bin_height) for x in count]
        average_diff_bin_height = (sum(diff_bin_height)/len(diff_bin_height))
        #print('The avg probability density is: ',average_bin_height)
        #print('The avg difference is: ',average_diff_bin_height)
        pyplot.axhline(y=sum(count)/len(count), linewidth=2, color='r')
        pyplot.xlim(-np.pi,np.pi)
        pyplot.xlabel('Phase (rad)')
        pyplot.ylabel('Probability density')
        pyplot.title('Phase distribution')
        if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'phase_distribution'+ pop_name +'.png',bbox_inches="tight")
        
    return sorted_convolved_activity, sorted_idx
        
