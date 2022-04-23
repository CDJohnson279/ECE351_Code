# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 12                                                        #
# Final Lab                                                     #
# 4/12/2022                                                     #
#                                                               #        
#################################################################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig
import scipy.fftpack
import control as con

### variables ###
fs = 1000000
steps = 1
hz = np.arange(0, 10e6, steps)

### functions ###
### Fast fourier Transform ###

def fftfunc(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi


def make_stem ( ax ,x ,y , color ='k', style ='solid ', label ='', linewidths=2.5 ,**kwargs) :
    ax.axhline(x[0], x[-1] ,0 , color='r')
    ax.vlines(x, 0, y, color=color, linestyles=style, label=label, linewidths=linewidths)
    ax.set_ylim([1.05*y.min(), 1.05*y.max()])

### Task 1 ###

 # load input signal
df = pd . read_csv ('NoisySignal.csv ')

t = df ['0']. values
sensor_sig = df ['1']. values

## Noisy Signal Plot ###
plt.figure ( figsize = (10 , 7) )
plt.plot (t , sensor_sig )
plt.grid ()
plt.title ('Noisy Input Signal ')
plt.xlabel ('Time [s]')
plt.ylabel ('Amplitude [V]')
plt.show ()


###Task 2 - creating the filter ####


#### Filtered Signal ####

L = 5.2613
R = 10000
C = 5.263e-8

# L = 1.59154943
# R = 10000
# C = 17.63489674e-9
### Basic Bandpass Filter ####
num = [R/L, 0]
den = [1, R/L, 1/(L*C)]


 ### Task 3 - Bode Plots ###
z, p = sig.bilinear(num, den, 100000)

outputSig = sig.lfilter(z, p, sensor_sig)

plt.figure(figsize = (10,7))
plt.ylim(-10,10)
plt.plot (t, outputSig)
plt.grid()
plt.title('Filtered Output Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()

sys = sig.TransferFunction(num,den)
w, mag, phase = sig.bode(sys, hz)


fig, ax = plt.subplots(2)
fig.suptitle("Bode Plot")
plt.tight_layout()
ax[0].semilogx(hz, mag)
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title("Magnitude")
ax[0].grid()

ax[1].semilogx(hz,phase)
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].grid()


# ### Verify Step 2 ###

### Position Measurement attenuated less than -.3dB ###
fig, ax = plt.subplots(2)
fig.suptitle("Verify Postion Measurement")
plt.tight_layout()
ax[0].set_ylim(-1 ,0)
ax[0].semilogx(hz, mag)
ax[0].set_xlim([1750,2050])
ax[0].set_ylabel("Decibel (dB)")
ax[0].set_title('Magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase)
ax[1].set_xlim([1750,2050])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].grid()

### Low frequency by at least -30db ###
fig, ax = plt.subplots(2)
fig.suptitle("Low-Frequency Vibration Noise")
plt.tight_layout()

ax[0].semilogx(hz, mag)
ax[0].set_xlim([1,1750])
ax[0].set_ylabel("Decibel (dB)")

ax[0].set_title('Magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase)
ax[1].set_xlim([1,1750])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].grid()

# ### Low frequency by at least -30db ###
fig, ax = plt.subplots(2)
fig.suptitle("High Frequency (Switching Amplifier) Noise")
plt.tight_layout()

ax[0].semilogx(hz, mag)
ax[0].set_xlim([1800,10e7])
ax[0].set_ylabel("Decibel (dB)")

ax[0].set_title('Magnitude')
ax[0].grid()

ax[1].semilogx(hz,phase)
ax[1].set_xlim([1800,10e7])
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Phase (degrees)")
ax[1].grid()

# ### Fast Fourier Transform Plots ###

fullFreq, fullXMag, fullXPhi = fftfunc(sensor_sig, fs)
freqFiltered, X_magFiltered, X_phiFiltered = fftfunc(outputSig, fs)

fig, ax = plt.subplots(figsize=(10,7))
make_stem(ax, fullFreq, fullXMag)
plt.title('Fast Fourier Transform Unfiltered')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
fig, ax = plt.subplots(figsize=(10,7))
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Fast Fourier Transform Filtered')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()



fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([2050, 100000])
make_stem(ax, fullFreq, fullXMag)
plt.title('Unfiltered High Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([2050, 100000])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Filtered High Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([1750, 2050])
make_stem(ax, fullFreq, fullXMag)
plt.title('Unfiltered Position Measurement')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([1750, 2050])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Filtered Position Measurement')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([0, 1780])
make_stem(ax, freqFiltered, X_magFiltered)
plt.title('Filtered Low Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
fig, ax = plt.subplots(figsize=(10,7))
ax.set_xlim([0,1780])
make_stem(ax, fullFreq, fullXMag)
plt.title('Unfiltered Low Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()