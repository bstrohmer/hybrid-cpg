from scipy import signal
from scipy.signal import butter, sosfilt, filtfilt, sosfiltfilt
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable #REQUIRED FOR COLORBAR
from set_network_params import neural_network
from sklearn.cluster import KMeans
netparams = neural_network()

pyplot.rcParams.update({'font.size': 20})

def run_PCA(smoothed_spikes,pop_name):
        
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from mpl_toolkits.mplot3d import Axes3D
        
    data = smoothed_spikes.T

    figPCA = pyplot.figure()
    figPCA.suptitle('PCA 3D of convolved spiking activity', fontsize=14)
    ax = figPCA.add_subplot(111, projection="3d")

    # scaler = StandardScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)

    pca = PCA(n_components=netparams.PCA_components)
    pca.fit(data)
    x_pca_s = pca.transform(data)
    ax.scatter(x_pca_s[0,0], x_pca_s[0,1], x_pca_s[0,2], color='xkcd:orange', s=50, alpha=1, marker='*' )
    ax.scatter(x_pca_s[-1,0], x_pca_s[-1,1], x_pca_s[-1,2], color='xkcd:blue', s=50, alpha=1, marker='v' )
    # ax.scatter(x_pca_s[:,0], x_pca_s[:,1], x_pca_s[:,2], color='xkcd:black', s=1, alpha=1, marker='.')
    ax.scatter(x_pca_s[:,0], x_pca_s[:,1], x_pca_s[:,2], color='xkcd:black', s=0.01, marker='.', alpha=0.8)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_zlabel("PC 3", fontsize=12)
    if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'PCA_3D_'+ pop_name +'.png')
    
    figPCA2D = pyplot.figure()
    figPCA2D.suptitle('PCA 2D of convolved spiking activity', fontsize=14)
    ax2D = figPCA2D.add_subplot(111)

    ax2D.scatter(x_pca_s[0,0], x_pca_s[0,1], color='xkcd:orange', s=50, alpha=1, marker='*' )
    ax2D.scatter(x_pca_s[-1,0], x_pca_s[-1,1], color='xkcd:blue', s=50, alpha=1, marker='v' )
    # ax2D.scatter(x_pca_s[:,0], x_pca_s[:,1], color='xkcd:black', s=1, alpha=1, marker='.')
    ax2D.scatter(x_pca_s[:,0], x_pca_s[:,1], color='xkcd:black', s=0.01, marker='.', alpha=0.8)
    ax2D.set_xlabel("PC 1", fontsize=12)
    ax2D.set_ylabel("PC 2", fontsize=12)
    if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'PCA_2D_'+ pop_name +'.png')

    # Calculate duration of events
    pca_data = pca.fit_transform(data)
    # Cluster the data using KMeans
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(pca_data)
    # Calculate the duration of each event (assume fixed duration of 1 second)
    durations = np.ones(len(data))
    # Calculate the total time spent in each cluster
    cluster_times = np.zeros(len(labels))
    for i, label in enumerate(labels):
        cluster_times[label] += durations[i]
    '''    
    # Create a color map based on the time values
    cmap = pyplot.get_cmap('viridis')
    time = np.arange(0,np.shape(smoothed_spikes)[1],1)
    colors = cmap(time)
    # Plot the PCA data with colors based on time
    pyplot.figure()
    pyplot.scatter(pca_data[:, 0], pca_data[:, 1],s=1)
    # Add a color bar to the plot
    cb = pyplot.colorbar()
    cb.set_label('Time')
    # Add axis labels and a title to the plot
    pyplot.xlabel('PCA Component 1')
    pyplot.ylabel('PCA Component 2')
    pyplot.title('PCA with Time Dimension')
    if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'PCA_2D_time_'+ pop_name +'.png')

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(data)   
    pyplot.figure()
    pyplot.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels)
    pyplot.xlabel('t-SNE 1')
    pyplot.ylabel('t-SNE 2')
    if netparams.args['save_results']: pyplot.savefig(netparams.pathFigures + '/' + 'tsne_2D'+ pop_name +'.png')
    '''
    print(f'\nPCA analysis:')
    print("Cluster times:", cluster_times[:5])
    print(f'    Explained variance: {np.round(pca.explained_variance_ratio_,2)}')
    #population_config['pca explained variance'] = pca.explained_variance_ratio_
    print(f'    n_samples_: {pca.n_samples_}')
    print(f'    n_features_in_: {pca.n_features_in_}')
    print(f'    singular_values_: {pca.singular_values_}')
    print(f'    New basis shape: {pca.components_.shape}') # New basis "Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_"
    print(f'    Shape of data in new basis: {x_pca_s.shape}') # Data in the new basis (only up to the number of components selected)
        
