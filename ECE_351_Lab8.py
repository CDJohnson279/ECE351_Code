# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 8                                                      #
# 2/8/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


#Step Function
steps = 2e-2

t = np.arange(0, 20 + 2e-2, steps)

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
T = 8
w = (2*np.pi)/T
 
 ### k = 1 ###
k=1 
a_k = 0
b_k  = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print("For k = ",k)
print("b",k," =", b_k)
print("a",k," = ",a_k)
print()

### k = 2 ###
k=2
a_k = 0
b_k  = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print("For k = ",k)
print("b",k," =", b_k)
print("a",k," = ",a_k)
print()

###k = 3 ###
k=3
a_k = 0
b_k  = 2*((1-np.cos(np.pi * k))/(np.pi * k))
print("For k = ",k)
print("b",k," =", b_k)
print("a",k," = ",a_k)
print()


### Task 2 ###
n1=0 
n3=0 
n15=0
n50=0
n150=0
n1500 = 0 

for k in range(1,1+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n1 = bk*np.sin(k*w*t)

for k in range(1,3+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n3 += bk*np.sin(k*w*t)
    
for k in range(1,15+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n15 += bk*np.sin(k*w*t)
    
for k in range(1,50+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n50 += bk*np.sin(k*w*t)

for k in range(1,150+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n150 += bk*np.sin(k*w*t)
    
for k in range(1,1500+1):
    bk = 2*((1-np.cos(np.pi * k))/(np.pi * k))
    n1500+= bk*np.sin(k*w*t)


plt.tight_layout()
plt.subplot(3,1,1)
plt.plot(t, n1)
plt.title('N = 1')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,n3)
plt.title('N = 3')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,n15)
plt.title('N = 15')
plt.grid()

plt.subplots(3)
plt.tight_layout()
plt.subplot(3,1,1)
plt.plot(t, n50)
plt.title('N = 50')
plt.grid()

plt.subplot(3,1,2)
plt.plot(t,n150)
plt.title('N = 150')
plt.grid()

plt.subplot(3,1,3)
plt.plot(t,n1500)
plt.title('N = 1500')
plt.grid()