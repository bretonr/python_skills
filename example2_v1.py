## v1 (pedestrian version)
## This version is highly innefficient in terms of execution speed.

## This example creates a Gaussian burst in a time series which is copied in
## different frequency channels and shifted according to the dispersion law.
## This produces a dynamic spectrum with a swepped up burst. Many of these
## dynamic spectra are generated with random pulse parameters in order to obtain
## a set of fake bursts. The data are generate at high resolution, and then
## downsampled to a lower resolution.


## Module import
## NOTE: Never import modules in the following way: from numpy import *
import numpy as np
import os
import matplotlib.pyplot as plt
from icecream import ic
ic.enable()

## Profiler
import yappi
RUN_YAPPI = True

## Setting random number generator
rng = np.random.default_rng(12345)



#------------------------------------------------------------------------------
## Define important functions

## NOTE: It is a good habit to gather functions at the top or in a separate file
## which is imported so that the actual script part is easier to follow

## NOTE: The following function examplifies how to implement a good docstring for a function.
def Delay(dm, frequencies):
    '''
    Calculates the delay at a given frequency according to the DM formula.

    Parameters
    ----------
    dm : float
        Dispersion measure.
    frequencies : float, ndarray
        Frequencies in GHz at which to calculate the delay.

    Returns
    -------
    t_delays : float, ndarray
        Time delay in ms at each frequency.
    '''
    t_delays = np.zeros(frequencies.shape)
    
    ## NOTE: Any potential issue here?
    for i in np.arange(frequencies.size):
        delay = 4.15*dm*( (frequencies[i]**-2) - (frequencies.max()**-2) )
        t_delays[i] = delay

    return t_delays

def Spectrum(flux0, freq0, alpha, freqs):
    flux = flux0*(freqs/freq0)**alpha
    return flux

def Calc_model(amp, t, t0, delays, duration):
    model = amp * np.exp( -0.5*(t-(t0+delays))**2 / duration**2 )
    return model

def Downsample(arr, fact0, fact1):
    new_shape = (arr.shape[0]//fact0, fact0, arr.shape[1]//fact1, fact1)
    new_arr = arr.reshape(new_shape)
    new_arr = new_arr.sum(axis=(1,3))
    return new_arr



#------------------------------------------------------------------------------
## Parameters definition

## NOTE: Also a good habit to gather the variables that can be modified at the
## top. This is much better than hardcording values as you might forget to
## modify a value somewhere if a change takes place.

## Final data size
nchannels_final = 128
nsamp_final = 128
nchannels_downsamp = 4
nsamp_downsamp = 150

## Input parameters for the observing setup
bottom_freq = 1.200 # in GHz
bandwidth = 0.4     # in GHz
obs_length = 1000.  # in ms

## Other input parameters
DMmax = 300
burst_minwidth = 1.
burst_maxwidth = 50.
burst_minsnr = 5.
burst_maxsnr = 20.
avg_index = -1.4
std_index = 0.5
avg_chan_noise = 1.
std_chan_noise = 1.0

## Number of mock bursts
nbursts = 100

## Directory where to store file
data_dir = '.'

## Derived quantities
nchannels = nchannels_final * nchannels_downsamp
nsamp = nsamp_final * nsamp_downsamp
dt_sampling = obs_length / nsamp
chanwidth = bandwidth / nchannels
top_freq = bottom_freq + (nchannels-1)*chanwidth



#------------------------------------------------------------------------------
## Script section

## Profiler
if RUN_YAPPI:
    yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
    yappi.start()


## Initialising various arrays
## NOTE: the following commented are not so good as prone to not yield the right size
#freqs = np.arange(bottom_freq, bottom_freq+bandwidth, chanwidth)
#tstamp = np.arange(0., obs_length, dt_sampling)
freqs = np.arange(nchannels)*chanwidth + bottom_freq
tstamp = np.arange(nsamp)*dt_sampling

## Double checking that the array sizes are right
assert freqs.size == nchannels
assert tstamp.size == nsamp

noise_arr = np.zeros((nbursts, nchannels_final, nsamp_final))
model_arr = np.zeros((nbursts, nchannels_final, nsamp_final))
data_arr = np.zeros((nbursts, nchannels_final, nsamp_final))

## Loop through the number of bursts to generate
for i in np.arange(nbursts):
    ## Temporary arrays to store parameters
    noise_arr_tmp = np.zeros((nchannels, nsamp))
    model_arr_tmp = np.zeros((nchannels, nsamp))
    data_arr_tmp = np.zeros((nchannels, nsamp))

    ## Generate burst parameters
    DM = rng.uniform(0., DMmax)
    t_burst = rng.uniform(0, obs_length)
    duration_burst = rng.uniform(burst_minwidth, burst_maxwidth)
    amp_burst = rng.uniform(burst_minwidth, burst_maxwidth)
    alpha = rng.normal(loc=avg_index, scale=std_index)

    ## Loop through the channels, draw a noise level from a lognormal distribution,
    ## then generate white noise according to this noise level.
    for j in np.arange(nchannels):
        noise_std_per_channel = rng.lognormal(mean=avg_chan_noise, sigma=std_chan_noise)
        noise = rng.normal(loc=0., scale=noise_std_per_channel, size=nsamp)
        # noise = 1.
        noise_arr_tmp[j] = noise


    ## Calculate the time delay at each frequency
    t_at_freqs = Delay(DM, freqs)

    ## Loop through the channels, generate the model
    for j in np.arange(nchannels):        
        ## Calculate the flux amplitude at each frequency
        flux_at_freq = Spectrum(amp_burst, bottom_freq, alpha, freqs[j])

        model = Calc_model(flux_at_freq, tstamp, t_burst, t_at_freqs[j], duration_burst)
        model_arr_tmp[j] = model

    ## Combining the data
    data_arr_tmp = noise_arr_tmp + model_arr_tmp

    noise_arr[i] = Downsample(noise_arr_tmp, nchannels_downsamp, nsamp_downsamp)
    model_arr[i] = Downsample(model_arr_tmp, nchannels_downsamp, nsamp_downsamp)
    data_arr[i] = Downsample(data_arr_tmp, nchannels_downsamp, nsamp_downsamp)


## Profiler
if RUN_YAPPI:
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()



#------------------------------------------------------------------------------
## Saving the results
np.save(os.path.join(data_dir,'noise'), noise_arr)
np.save(os.path.join(data_dir,'model'), model_arr)
np.save(os.path.join(data_dir,'data'), data_arr)



#------------------------------------------------------------------------------
## Plotting the last burst so we can visualise

if 0:

    ## Plot cavas for the original data
    fig1, ax1 = plt.subplots(ncols=3, figsize=(12,4))
    ax1[0].imshow(noise_arr_tmp, origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax1[1].imshow(model_arr_tmp, origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax1[2].imshow(data_arr_tmp, origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax1[0].set_title('Noise Original')
    ax1[1].set_title('Model Original')
    ax1[2].set_title('Data Original')
    ax1[0].set_xlabel('Time (ms)')
    ax1[1].set_xlabel('Time (ms)')
    ax1[2].set_xlabel('Time (ms)')
    ax1[0].set_ylabel('Frequency (GHz)')
    ax1[1].set_ylabel('Frequency (GHz)')
    ax1[2].set_ylabel('Frequency (GHz)')
    fig1.subplots_adjust(left=0.06, right=0.98, wspace=0.3)

    ## Plot cavas for the downsampled data
    fig2, ax2 = plt.subplots(ncols=3, figsize=(12,4))
    ax2[0].imshow(noise_arr[-1], origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax2[1].imshow(model_arr[-1], origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax2[2].imshow(data_arr[-1], origin='lower', aspect='auto', extent=[0, obs_length, bottom_freq, top_freq])
    ax2[0].set_title('Noise Downsampled')
    ax2[1].set_title('Model Downsampled')
    ax2[2].set_title('Data Downsampled')
    ax2[0].set_xlabel('Time (ms)')
    ax2[1].set_xlabel('Time (ms)')
    ax2[2].set_xlabel('Time (ms)')
    ax2[0].set_ylabel('Frequency (GHz)')
    ax2[1].set_ylabel('Frequency (GHz)')
    ax2[2].set_ylabel('Frequency (GHz)')
    fig2.subplots_adjust(left=0.06, right=0.98, wspace=0.3)








