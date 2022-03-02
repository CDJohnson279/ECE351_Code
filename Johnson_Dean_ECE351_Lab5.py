# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 4                                                        #
# 2/15/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


#transfer function H(s) = 10000s/(s^2+(s/10000)+3.7037*10^8)

### Part 1 ###

steps = 1.2e-5

t = np.arange(0, 1.2e-3, steps)

#impulse response h(t) hand calculated
ht = 10000*np.e**(-5000*t)*np.cos(18584*t) - 2690.5*np.e**(-5000*t)*np.sin(18584*t)

R = 1000 
L = 0.027
C = 100E-9
z = 1/(R*C)
al = -0.5*z
w = 0.5*np.sqrt(z**2 - 4/(L*C)+0*1j)


p = al + w
g = p * z
g = np.sqrt((-0.5*z**2)**2 + (z*w)**2)
g_abs = np.abs(g)
g_ang = np.angle(g)


system = ([10000.0, 0.0], [1.0, 10000.0, 370370370.4])
t, ht_verify = sig.impulse(system) #impulse using scipy


plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.tight_layout()
plt.title('h(t) with hand calculated')
plt.plot(t, ht)
plt.subplot(2,1,2)
plt.tight_layout()
plt.title('h(t) using scipy')
plt.plot(t, ht_verify)
plt.ylabel('h(t)')
plt.xlabel('t')
#### Part 2 ###

t, hs = sig.step(system)

plt.figure(figsize = (10, 7))
plt.tight_layout()
plt.title('Step response of H(s) using scipy.step')
plt.plot(t, hs)
plt.tight_layout()
plt.grid()
plt.ylabel('H(t)')
plt.xlabel('t')
