# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 9                                                      #
# 3/22/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal  as sig
import scipy.fftpack

#Step Function
steps = 1e-2

t = np.arange(0,2,steps)

def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

def u(t):
    if t < 0:
        return 0
    if t >= 0:
        return 1
# Ramp funtion
def r(t):
    y = np.zeros(t.shape) #t.shape of whatever is inputted in
    for i in range(len(t)): # run the loop once for each index of t 
        if t[i] >= 0: 
            y[i] = t[i] 
        else:
            y[i] = 0
    return 
  
 ### Part 1 ###  
###Task 1###
x = np.cos(np.pi * 2 * t)
fs = 100


def fftfunc(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    return freq, X_mag, X_phi
    
    
freq, X_mag, X_phi = fftfunc(x, fs)


plot = plt.figure()

# plot1 = plt.subplot2grid((3,2),(0,0), colspan = 2, rowspan = 1)
# plot1.plot(t,x)
# plot1.grid()
# plot1.title.set_text("Part 1 Task 1")

# plot2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
# plot2.stem(freq, X_mag)
# plot2.grid()

# plot3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
# plot3.set_xlim([-2,2])
# plot3.stem(freq, X_mag)
# plot3.grid()

# plot4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
# plot4.stem(freq, X_phi)
# plot4.grid()

# plot5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
# plot5.stem(freq, X_phi)
# plot5.set_xlim([-2,2])
# plot5.grid()

# plot.tight_layout()

# ###Task 2 ###
# x = 5*np.sin(2*t*np.pi)
# fs = 100
  
# freq, X_mag, X_phi = fftfunc(x, fs)

# plt.stem(freq,X_mag)
# plt.stem(freq, X_phi)

# plot = plt.figure()

# plot1 = plt.subplot2grid((3,2),(0,0), colspan = 2, rowspan = 1)
# plot1.plot(t,x)
# plot1.grid()
# plot1.title.set_text("Part 1 Task 2")

# plot2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
# plot2.stem(freq, X_mag)
# plot2.grid()

# plot3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
# plot3.set_xlim([-2,2])
# plot3.stem(freq, X_mag)
# plot3.grid()

# plot4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
# plot4.stem(freq, X_phi)
# plot4.grid()

# plot5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
# plot5.stem(freq, X_phi)
# plot5.set_xlim([-2,2])
# plot5.grid()

# plot.tight_layout()

# ### Task 3 ###
# x = 2*np.cos((2*np.pi*2*t)-2) + (np.sin((2*np.pi*6*t)+3))**2
        
# freq, X_mag, X_phi = fftfunc(x, fs)

# plt.stem(freq,X_mag)
# plt.stem(freq, X_phi)

# plot = plt.figure()

# plot1 = plt.subplot2grid((3,2),(0,0), colspan = 2, rowspan = 1)
# plot1.plot(t,x)
# plot1.grid()
# plot1.title.set_text("Part 1 Task 3")

# plot2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
# plot2.stem(freq, X_mag)
# plot2.grid()


# plot3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
# plot3.set_xlim([-15,15])
# plot3.stem(freq, X_mag)
# plot3.grid()

# plot4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
# plot4.stem(freq, X_phi)
# plot4.grid()

# plot5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
# plot5.stem(freq, X_phi)
# plot5.set_xlim([-2,2])
# plot5.grid()

# plot.tight_layout()

### Task 4 ###
def fftfunc2(x, fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_mag)):
        if X_mag[i] < 1e-10:
            X_phi[i] = 0
            
    return freq, X_mag, X_phi

### Redone Task1  ###

x = np.cos(np.pi * 2 * t)   
    
freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 4 - Task 1")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

### Redone task 2 ###

x = 5*np.sin(2*np.pi*t)

freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 4 - Task 2")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-2,2])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-2,2])
fig5.grid()

fig.tight_layout()

###Redone Task 3 ###

x = 2*np.cos((2*np.pi*2*t) - 2) + np.sin((2*np.pi*6*t)+3)**2

freq, X_mag, X_phi = fftfunc2(x, fs)

fig = plt.figure()

fig1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
fig1.plot(t,x)
fig1.title.set_text("Task 4 -Task 3")
fig1.grid()

fig2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
fig2.stem(freq, X_mag)
fig2.grid()

fig3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
fig3.set_xlim([-15,15])
fig3.stem(freq, X_mag)
fig3.grid()

fig4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
fig4.stem(freq, X_phi)
fig4.grid()

fig5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
fig5.stem(freq, X_phi)
fig5.set_xlim([-15,15])
fig5.grid()

fig.tight_layout()

### Task 5 ###

### N=15 case from Lab 8 ###
T = 8
w = (2*np.pi)/T
n15=0
t = np.arange(0, 16, steps)

for k in range(1,15+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n15 += bk*np.sin(k*w*t)
    
freq, X_mag, X_phi = fftfunc2(n15, fs)
plot = plt.figure()

plot1 = plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
plot1.plot(t,n15)
plot1.title.set_text("Task 5")
plot1.grid()

plot2 = plt.subplot2grid((3,2), (1,0), colspan=1, rowspan=1)
plot2.stem(freq, X_mag)
plot2.grid()

plot3 = plt.subplot2grid((3,2), (1,1), colspan=1, rowspan=1)
plot3.set_xlim([-2,2])
plot3.stem(freq, X_mag)
plot3.grid()

plot4 = plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
plot4.stem(freq, X_phi)
plot4.grid()

plot5 = plt.subplot2grid((3,2), (2,1), colspan=1, rowspan=1)
plot5.stem(freq, X_phi)
plot5.set_xlim([-2,2])
plot5.grid()

plot.tight_layout()