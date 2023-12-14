from scipy import signal
from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt, find_peaks, correlate
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable #REQUIRED FOR COLORBAR
from set_network_params import neural_network
netparams = neural_network()

def order_by_phase(convolved_activity, population_mean_activity, pop_name, avg_period_mean_activity):
    remove_mean = netparams.remove_mean
    high_pass_filtered = netparams.high_pass_filtered
    generate_plot = netparams.phase_ordered_plot
    """Compute phase ordering as described in "Movement is governed by rotational population dynamics in spinal motor networks, Linden et al. 2022"

    Args:
        convolved_activity (np.array): continuos firing activity resulting from convolving spikes (nb_neurons x simulation_time)
        population_mean_activity (np.array): mean population activity (1 x simulation_time)
        remove_mean (bool): if True, it removes the mean from each convolved activity. Defaults to True.
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
    #Remove last time window from population mean activity
    population_mean_activity = population_mean_activity[:-netparams.time_window+1]

    freqs_pop, psd_pop = signal.welch(population_mean_activity,fs=10) #sampling frequency is 1 by default
    #print('Freq mean activity: ',freqs_pop)
    peak_population_mean_activitypsd_freq = freqs_pop[np.where(psd_pop == psd_pop.max())[0][0]]
    #print('Peak pop mean activity freq: ',peak_population_mean_activitypsd_freq)
    
    phase_i = []
    psd_indiv = []
    phase_dist = []
    for convolved_activity_neuron_i in convolved_activity:
        # Cross spectral density or cross power spectrum of x,y.
        f, Pxy = signal.csd(convolved_activity_neuron_i,population_mean_activity,fs=10) #sampling frequency is 1 by default
        #print('Freq csd: ',f)
        index_ = np.where(f == peak_population_mean_activitypsd_freq)[0][0]
        phase_activity = np.angle(Pxy)     
        phase_i.append(phase_activity[index_])
        #psd_indiv.append(psd_indiv_neuron)
        
        #Use cross-correlation to find phase difference for phase distribution plot	
        corr = correlate(population_mean_activity, convolved_activity_neuron_i, mode='same')	
        max_index = int(np.argmax(corr)) # Find the index of the maximum value in the correlation        
        t2 = np.arange(0,len(corr),1)
        time_at_max = t2[max_index]
        abs_t2_max = abs(t2[max_index])
        #Reduce the time difference to be within one period
        max_multiplier = 1
        while abs_t2_max >= max_multiplier * avg_period_mean_activity:
            max_multiplier += 1
        if max_multiplier > 1:
            if t2[max_index] > 0:
                time_at_max -= max_multiplier * avg_period_mean_activity
            else:
                time_at_max += max_multiplier * avg_period_mean_activity
            #print(f"Max is over {max_multiplier}x")
        #Calculate the phase    
        phase_diff = (time_at_max * 2 * np.pi) / avg_period_mean_activity
        #print('Length pop mean act, indiv act, correlation',len(population_mean_activity),len(convolved_activity_neuron_i),len(t2))
        #print("Absolute time, average period ",abs(t2[max_index]),avg_period_mean_activity)
        #print("Time at max index ",time_at_max," Phase diff: ",phase_diff)
        phase_dist.append(abs(phase_diff))
    
    #Normalize activity and sort
    max_activity = np.max(convolved_activity)
    #convolved_activity = (convolved_activity-np.min(convolved_activity))/(np.max(convolved_activity)-np.min(convolved_activity))
    #print('Sample psd output',psd_indiv_neuron)
    sorted_idx = sorted(range(len(phase_i)), key=lambda k:phase_i[k])
    sorted_convolved_activity = convolved_activity[sorted_idx]

    if generate_plot:
        figConv, axsConv = pyplot.subplots(2, figsize=(10,8))
        figConv.suptitle('Individual Neuronal Firing Rate')
        im = axsConv[0].imshow(convolved_activity, aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
        axsConv[0].set(title='Unsorted', ylabel='Neuron idx #')
        axsConv[0].set(xticks=[0,5000,10000,15000,20000,25000,30000])
        axsConv[0].set(xticklabels=[0,500,1000,1500,2000,2500,3000])
        #********************ADD COLORBAR TO PLOT*****************************
        divider = make_axes_locatable(axsConv[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(im,cax=cax)
        
        im = axsConv[1].imshow(sorted_convolved_activity, aspect='auto', interpolation='nearest')#, vmin=0, vmax=1)
        #axsConv[1].set(title='Phase sorted', xlabel='Time [ms]', ylabel='Neuron idx #')
        axsConv[1].set(xlabel='Time (ms)', ylabel='Neuron idx #')
        axsConv[1].set(xticks=[0,5000,10000,15000,20000,25000,30000])
        axsConv[1].set(xticklabels=[0,500,1000,1500,2000,2500,3000])
        #********************ADD COLORBAR TO PLOT*****************************
        divider = make_axes_locatable(axsConv[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(im,cax=cax)
        if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'phase_sorted'+ pop_name +'.pdf',bbox_inches="tight")
        
        pyplot.figure()
        #count, bins, ignored = pyplot.hist(phase_i,bins=25,density=True,stacked=True)
        count, bins = np.histogram(phase_dist, bins=25,density=True)
        count_normalized = count / float(np.max(count))
        pyplot.bar(bins[:-1], count_normalized, width=np.diff(bins))
        print('The bin with the smallest count in',pop_name,' is: ',np.min(count),np.min(count_normalized))
        #average_bin_height = (sum(count_normalized)/len(count_normalized)) 
        #diff_bin_height = [abs(x - average_bin_height) for x in count_normalized]
        #average_diff_bin_height = (sum(diff_bin_height)/len(diff_bin_height))
        #print('The avg probability density is: ',average_bin_height)
        #print('The avg difference is: ',average_diff_bin_height)
        #pyplot.axhline(y=sum(count_normalized)/len(count_normalized), linewidth=2, color='r')
        pyplot.xlim(0,2*np.pi)
        pyplot.xlabel('Phase (rad)')
        pyplot.ylabel('Probability density')
        pyplot.title('Phase distribution')
        if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'phase_distribution'+ pop_name +'.pdf',bbox_inches="tight")
        #np.savetxt(netparams.pathFigures + '/counts_per_phase_bin.csv',count,delimiter=',')
        #np.savetxt(netparams.pathFigures + '/normalized_counts_per_phase_bin.csv',count_normalized,delimiter=',')
        
    return sorted_convolved_activity, sorted_idx
        
