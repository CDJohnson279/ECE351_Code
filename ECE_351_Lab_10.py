# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 10                                                     #
# 3/29/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack
import control as con

steps = 1

w = np.arange(10e2, 10e5, steps)
Hz = np.arange(10e3, 10e6, steps)

C = 100e-9
R = 1000
L = 27e-3

###  Part 1 ###


mH = 20*np.log10(abs((w/(R*C))/(np.sqrt((((1/(L*C))-w**2)**2)+(w/(R*C))**2))))

mH2 = 20*np.log10(abs((w/(R*C))/(np.sqrt(w**4+((1/(R*C))**2-(2/(L*C)))*w**2+(1/(L*C))**2))))

pH = np.rad2deg((np.pi/2) - np.arctan2((w/(R*C)),((1/(L*C))-w**2)))

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]

sys = sig.TransferFunction(num,den)
sysHz = con.TransferFunction(num,den)

w2, mag2, phase2 = sig.bode(sys, w)
_ = con.bode(sysHz, w, dB = True, Hz = True, deg = True, Plot = True)

plt.subplots(2)
plt.tight_layout()
plt.subplot(2,1,1)
plt.semilogx(w, mH)
plt.title('Magnitude Plot')
plt.grid()

plt.subplot(2,1,2)
plt.semilogx(w,pH)
plt.title('Phase Plot')
plt.grid()



#############################################
##                  Part 2                 ##
#############################################

fs = 1000000
stepP2 = 1/fs

t = np.arange(0, 0.01, stepP2)

xt = np.cos(np.pi*200*t) + np.cos(2*np.pi*3024*t)+np.sin(2*np.pi*50000*t)

plt.figure()
plt.plot(t, xt)
plt.title("x(t) Plot")
plt.show()

z, p = sig.bilinear(num, den,fs)

y = sig.lfilter(z, p, xt)

plt.figure()
plt.plot(t, y)
plt.title("y(t) Filter Plot")
plt.show()