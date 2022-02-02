# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 3                                                         #
# 2/8/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
steps = 1e-2 # Define step size
t = np . arange (0 , 20 + steps , steps ) # Add a step size to make sure the
                                         # plot includes 5.0. Since np. arange () only
                                         # goes up to , but doesn ’t include the
                                         # value of the second argument
#Ramp function
t = np.arange(0, 10 + .1, .1)
def ramp(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
    return y

y = ramp(t)

#Step Function
t = np.arange(0, 10 + steps, steps)
def step(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

y = step(t)
######## Part 1 #########

#Task 1 

#f1(t)=u(t-2)-u(t-9)
t = np . arange (0 , 20 + steps , steps )
def f1(t):
    y=step(t-2)-step(t-9)
    return y 
a = f1(t)


#f2(t) =(e^-t)*u(t)
def f2(t):
    y=np.exp(-t)*step(t)
    return y 
b=f2(t) 



#f3(t) = r(t-2)[u(t-2)-u(t-3)]+r(4-t)[u(t-3)-u(t-4)]
def f3(t):
    y = ramp(t-2)*(step(t-2)-step(t-2))+ramp(4-t)*(step(t-3)-step(t-4)) 
    return y 
c=f3(t) 

steps = .1
t = np.arange(0,20+steps,steps)

a = f1(t)
b = f2(t)
c = f3(t)

myFigSize = (20,6)
plt.figure(figsize=myFigSize)

plt.subplot(1,3,1)
plt.plot(t,a,'b')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('f1(t)')

plt.subplot(1,3,2)
plt.plot(t,b,'r')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('f2(t)')

plt.subplot(1,3,3)
plt.plot(t,c, 'g')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('f(t)')
plt.show()

###### Part 2 ####
#Task 1 - user defined convolution function

def convolve(f,h):
    len_f = np.size(f) #size of forcing function
    len_h = np.size(h) #size of impulse response
    
    #since functions share a commmon start the total length is 1 les than the sum

    sum = np.zeros(len_f+len_h - 1)
    
    for i in np.arange(len_f):         #at given time in forcing function
        for j in np.arange(len_h):     #at same time in impulse function
            sum[i+j] = sum[i+j] + f[i]*h[j]   #the total sum of previous time with the current time
    return sum

#setting the range
steps = .1
t = np.arange(0,20+steps,steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN-1]+steps, steps)

a = f1(t)
b = f2(t)
c = f3(t)

#Task 2 - convolution of f1 and f2
conv1 = convolve(a,b)
#verification using scipy.signal.convolve()
verify1 = sig.convolve(a,b)

myFigSize = (20,5)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.plot(tExtended,conv1,'b')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Task 2 - Convolution of f1 and f2.')

plt.subplot(1,2,2)
plt.plot(tExtended,verify1,'r')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Verification of f1 and f2.')

#Task 3
conv2 = convolve(b,c)
#verification using scipy.signal.convolve()
verify2 = sig.convolve(b,c)

myFigSize = (20,5)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.plot(tExtended,conv2,'b')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Task 3 - Convolution of f2 and f3.')

plt.subplot(1,2,2)
plt.plot(tExtended,verify2,'r')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Verification of f2 and f3.')

#Task 4
conv3 = convolve(a,c)

#verification using scipy.signal.convolve()
verify3 = sig.convolve(a,c)

myFigSize = (20,5)
plt.figure(figsize=myFigSize)
plt.subplot(1,2,1)
plt.plot(tExtended,conv3,'b')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Task 4 - Convolution of f1 and f3.')

plt.subplot(1,2,2)
plt.plot(tExtended,verify3,'r')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('Verification of f1 and f3.')