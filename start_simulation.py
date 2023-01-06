#!/usr/bin/env python

import nest
import nest.raster_plot
import numpy as np
import sys
import pylab
import math
import matplotlib.pyplot as plt
import random
import scipy
import scipy.fftpack
import time
import numpy as np
import set_network_params as netparams

nn=netparams.neural_network()

class nest_start():
    def __init__(self):
        np.set_printoptions(precision=1,threshold=sys.maxsize)
        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()
        nest.SetKernelStatus({"local_num_threads":1,"resolution":nn.time_resolution})
        nest.SetKernelStatus({'rng_seed': nn.rng_seed})
        plt.rcParams.update({'font.size': 20})
        nest.GetKernelStatus("recording_backends")




