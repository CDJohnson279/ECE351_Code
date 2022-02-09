# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 4                                                         #
# 2//2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


#Step Function

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
    
f = 0.25
w = f*2*np.pi
 
def h1(t): 
    a = np.zeros((len(t), 1))
    for i in range(len(t)):
        a[i] = np.exp(2*t[i])*u(1-t[i])
    return a

def h2(t): 
    b = np.zeros((len(t), 1))
    for i in range(len(t)):
        b[i] = u(t[i]-2)-u(t[i]-6)
    return b

def h3(t): 
    f = 0.25
    w = f*2*np.pi
    c = np.zeros((len(t), 1))
    for i in range(len(t)):
        c[i] = math.cos(w*t[i])*u(t[i])
    return c

#Task 2 plot of the h(t) functions
steps = .1
t = np.arange(-10,10+steps,steps)

a = h1(t)
b = h2(t)
c = h3(t)

#Graphs for each h(t) function 
myFigSize = (20,6)
plt.figure(figsize=myFigSize)

plt.subplot(1,3,1)
plt.plot(t,a)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('h1(t)')
plt.title('h1(t) = exp(2*t)*u(1-t)')

plt.subplot(1,3,2)
plt.plot(t,b,'r')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('h2(t)')
plt.title('h2 = u(t-2)-u(t-6)')

plt.subplot(1,3,3)
plt.plot(t,c, 'm')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('h3(t)')
plt.title('h3 = cos(w*t)*u(t)')
plt.show()


def convolve(f,h):
    len_f = np.size(f) #size of forcing function
    len_h = np.size(h) #size of impulse response
    
    sum = np.zeros(len_f+len_h - 1) #since functions share a commmon start (assuming at 0) the total length is 1 less than the sum
    
    for i in np.arange(len_f):         #at given time in forcing function
        for j in np.arange(len_h):     #at same time in impulse function
            sum[i+j] = sum[i+j] + f[i]*h[j]   #the total sum of previous time with the current time
    return sum

#unit step to array
def array_u(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        y[i] = u(t[i])
    return y

### Part 2 ###
### Task 1 ### 
#convolution of step with user defined function
steps = .1
t = np.arange(-10,10+steps,steps)
NN = len(t)
tExtended = np.arange(-20, 2*t[NN-1]+steps, steps)

a = h1(t)
b = h2(t)
c = h3(t)
h = array_u(t)

conv_1 = convolve(a,h)
conv_2= convolve(b,h)
conv_3= convolve(c,h)

myFigSize = (20,5)
plt.figure(figsize=myFigSize)

plt.subplot(1,3,1)
plt.plot(tExtended,conv_1*steps)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('h1(t)*u(t)')

plt.subplot(1,3,2)
plt.plot(tExtended,conv_2*steps,'r')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('h2(t)*u(t)')

plt.subplot(1,3,3)
plt.plot(tExtended,conv_3*steps, 'm')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('h1(t)*u(t)')
plt.show()

#verification of user defined convolve

verify1 = sig.convolve(a,h)
verify2 = sig.convolve(b,h)
verify3 = sig.convolve(c,h)

myFigSize = (20,5)
plt.figure(figsize=myFigSize)

plt.subplot(1,3,1)
plt.plot(tExtended,verify1*steps)
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Verification of h1(t)*u(t)')

plt.subplot(1,3,2)
plt.plot(tExtended,verify2*steps,'r')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Verification of h2(t)*u(t)')

plt.subplot(1,3,3)
plt.plot(tExtended,verify3*steps, 'm')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Verification of h3(t)*u(t)')
plt.show()


#hand calculated convolution.
handconv1 = (.5*(1-np.exp(-2*tExtended))*step(tExtended)) - (.5*(1-np.exp(-2*(tExtended-3)))*step(tExtended-3))
handconv2 = (tExtended-2)*step(tExtended-2)-(tExtended-6)*step(tExtended-6)
handconv3 = (np.sin(w*tExtended)/w)*step(tExtended)
                                                                                             
myFigSize = (20,5)
plt.figure(figsize=myFigSize)
                                                                                                                                                                                       
plt.subplot(1,3,1)
plt.plot(tExtended,handconv1/steps,'b')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Hand Calculated h1(t)*u(t)')

plt.subplot(1,3,2)
plt.plot(tExtended,handconv2,'r')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Hand Calculated of h2(t)*u(t)')

plt.subplot(1,3,3)
plt.plot(tExtended,handconv3, 'm')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Hand Calculated of h3(t)*u(t)')
plt.show()                                                                                             
                                                                                             
                                                                                             
                                                                                             
                                                                                             
                                                                                             
                                                                            