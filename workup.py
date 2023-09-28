"""
Created on Tue Sep 19 13:03:40 2023

@author: achri

Input is raw VNA (frequency domain) data in the form of magnitude and phase
This decomposes the data into real and imaginary components, appends zeros, performs an IFFT to the time domain
The TD data is then time-gated to remove other contributions (only SAW remains)
The time-gated TD data is then FFTed back to FD and coefficients from padded zeros are removed
This final FD data is compared to the original FD data

Parameters that need to be edited for each dataset 

Time-gating bounds
    make a new function that timegates based on bound input

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math

# allows for calling of array.index() - must have defined array as x = myarray(np.array(data))
class myarray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self == value)

def read_dat(file_name):
    datContent = [i.strip().split() for i in open(file_name).readlines()]
    
    # first two rows are headers (column title and units)
    datContent = np.asarray(datContent)
    
    # converts the data in the array to float
    datContent = datContent.astype(float)      

    # dataframe with columns for frequency, magnitude, and phase
    d = {'Frequency': datContent[:, 0], 'Magnitude': datContent[:, 1], 'Phase': datContent[:, 2]}
    df = pd.DataFrame(data=d)
    
    return df, file_name

def plot_data(df):
    df.plot(x='Frequency', y='Magnitude', title=f'Frequency vs. Magnitude ({file_name})', xlabel='Frequency (MHz)', ylabel='|S21| (dB)', figsize=(15,10), xlim=(100, 6000))
    df.plot(x='Frequency', y='Phase', title=f'Frequency vs. Magnitude ({file_name})', xlabel='Frequency (MHz)', ylabel='Phase', figsize=(15,10), xlim=(100, 6000))

def complex_components(magnitude, phase):
    # Take the magnitude and phase data from VNA and decompose into real and imaginary components
    # Initializing correct length zero arrays to replace with values
    vector = np.zeros((len(magnitude)), dtype=complex)
    coeff_real = np.zeros((len(magnitude)))
    coeff_imag = np.zeros((len(magnitude)))
    
    for i in range(0, len(magnitude)):
        magnitude[i] = 10**(magnitude[i]/20) # Converting back to S21 from logarithmic scale (dB)
        # real component
        coeff_real[i] = (magnitude[i])*np.cos((phase[i])*np.pi/180)
        # imaginary component
        coeff_imag[i] = (magnitude[i])*np.sin((phase[i])*np.pi/180)
        # combine components into complex vector
        vector[i] = complex(coeff_real[i], coeff_imag[i])
          
    return vector, coeff_real, coeff_imag

def window(array):
    # Crude window function, just setting first and last values to 0
    array[0] = 0
    array[-1] = 0
    
    return array

def zero_padding(array, freq):
    # Uses start frequency to find how many zeros and corresponding freq values need to be appended to get to v = 0
    num_zeros = math.trunc(freq[0]/(freq[1] - freq[0])) + 1
    
    # Pads number of zeros/values found above
    array = np.pad(array, (num_zeros, 0), 'constant', constant_values = 0)
    freq = np.pad(freq, (num_zeros, 0), 'linear_ramp', end_values = (100 - ((freq[1] - freq[0])*num_zeros)))
    
    return array, freq, num_zeros
    
def time_domain(array, freq):
    # Perform real-data inverse fourier transform
    array = np.fft.irfft(array)
    
    # Find Nyquist frequency for given measurement
    nyquist_frequency = max(freq)
    
    # Calculate time interval from Nyquist frequency and make time axis to display time domain data
    time_interval = 1/(2*nyquist_frequency*1e6)
    time_axis = np.arange(0, time_interval*len(array), step=time_interval)

    return array, time_axis, time_interval

def time_gating(array, time_axis):
    # Making time_axis compatible with the index function defined by myarray class
    time_axis = myarray(time_axis)
    
    # Defining endpoints for rectangular timegate
    j = int(time_axis.index(1.8e-7)[0])
    k = int(time_axis.index(3.0e-7)[0])

    # Applying timegate
    for i in range(0, len(time_axis)):
        if i < j or i > k:
            array[i] = 0
        else:
            continue
    
    return array

def frequency_domain(array, time_interval, num_zeros):
    # Performing Fourier transform to return data to frequency domain
    freq = np.fft.rfftfreq(len(array), d=time_interval)
    array = np.fft.rfft(array)

    # Converting raw S21 back to S21 (dB)
    array = 20*np.log10(array)

    # Now need to delete vector values where zeros were padded and the corresponding frequencies
    array = np.delete(array, np.array(np.arange(0, num_zeros)))
    freq = np.delete(freq, np.array(np.arange(0, num_zeros)))
    
    return array, freq

if __name__ == '__main__':
    
    data, file_name = read_dat("0906_log_phase_2.dat")
    
    plot_data(data)
    
    fourier_coefficients, real, imag = complex_components(data['Magnitude'], data['Phase'])
    
    freq = np.array(data['Frequency'])
    magnitude = np.array(data['Magnitude'])
    
    fourier_coefficients = window(fourier_coefficients)
    
    fourier_coefficients, freq, N = zero_padding(fourier_coefficients, freq)
    
    td_array, time_axis, time_interval = time_domain(fourier_coefficients, freq)
    
    gated_td_array = time_gating(td_array, time_axis)
    
    gated_fd_array, freq = frequency_domain(gated_td_array, time_interval, N)
