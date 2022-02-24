# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 6                                                        #
# 2/22/2022                                                      #
#                                                               #        
#                                                               #
#################################################################


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


steps = 2e-2
tmin = 0
tmax = 2
t = np.arange(tmin,tmax+steps,steps)
##step function

t = np.arange(0, 2, steps)

def u(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
    return y

### Part 1 ###

def y(t):
    y = (np.exp(-6*t)-.5*np.exp(-4*t)+.5)*u(t)
    return y

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,y(t))
plt.title('Step Response') 
plt.ylabel('Hand Calculated ') 
plt.grid(True)



numerator = [1,6,12]
denomenator = [1,10,24]

t,ys = sig.step((numerator,denomenator),T=t)
plt.subplot(2,1,2) 
plt.plot(t,ys)
plt.ylabel('Response with scipy') 
plt.grid(True)
plt.xlabel('t')
plt.show() 

#partial fraction expansion reuslts
R,P,K = sig.residue(numerator,denomenator)
print("R=", R)
print("P=", P)
print("K=", K)

### Part 2 ###
tmax = 4.5
deny2 = [1,18,218,2036,9085,2550]
numx2 = [25250] 

R2, P2, K2 = sig.residue(numx2,deny2)
print("R=",R2)
print("P=",P2)
print("K=",K2)

def cosmeth(R,P,t):
    y = 0
    for i in range(len(R)):
        y += (abs(R[i])*np.exp(np.real(P[i])*t)*np.cos(np.imag(P[i]*t)+np.angle
        (R[i]))*u(t))
    return y

y1 = cosmeth(R2,P2,t)


numx2 = [25250]
deny2 = [1,18,218,2036,9085,25250]
t,ys = sig.step((numx2,deny2),T=t)

plt.figure(figsize=(12,8))
plt.subplot(2,1,1) 
plt.plot(t,y1)
plt.title('Step Response Part 2') 
plt.ylabel('Cosine Method') 
plt.grid(True)

plt.subplot(2,1,2) 
plt.plot(t,ys)
plt.ylabel('Step Response using Scipy') 
plt.grid(True)
plt.xlabel('t')
plt.show() 
