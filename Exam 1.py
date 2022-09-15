# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE450                                                        #
# Exam 1                                                        #
# 9/15/2022                                                     #
#                                                               #        
#                                                               #
#################################################################
import matplotlib.pyplot as plt
import numpy as np
from math import exp, cos, sin, log, pi, sqrt
from cmath import exp



simsec = 200
dt = 0.01
NN = int(simsec/dt)
TT = np.arange(0, dt*NN, dt)
time0 = np.zeros(NN)
time1 = np.zeros(NN)
time2 = np.zeros(NN)
time3 = np.zeros(NN)
time4 = np.zeros(NN)


""" Matrix setup """
A = np.matrix('0 1 0 0 0; -10 -5 1 0 0; 0 0 -10 1 0; 0 0 0 0 1; 0 0 0 -5 -2') 

print('A=\n', A)

B = np.matrix('0; 0; 0; 0; 1')
print('B=\n', B)
C = np.matrix('-3 1 0 -3 1.')
print('C=\n', C)

D = np.matrix('0') 
x = np.matrix('0; 0; 0; 0; 0')
u = np.matrix('0')

    
   

# # f = u(t-2)-u(t-4)
f = np.zeros(NN)
for i in range(int(2/dt)+1, int(4/dt)+1):
    f[i] = 1
    

y = np.zeros(NN)


""" Simulation """
nsteps = NN
for i in range(nsteps):
    y[i] = C*x + D*u
    u[0] = f[i]
   
    x = x + dt*A*x + dt*B*u
    time0[i] = x[0]
    time1[i] = x[1]
    time2[i] = x[2]
    time3[i] = x[3]
    time4[i] = x[4]

plt.figure(figsize=(16, 9))
plt.subplot(411)
plt.plot(TT, f, 'b', label='f(t)')
plt.title("Exam 1")
plt.axis([0, 10, -.5, 1.5])
plt.legend()
plt.grid()


plt.subplot(412)    
plt.plot(TT,time0,'m-.',label='x1')
plt.plot(TT,time2,'c-.',label='x2')
plt.plot(TT,time1,'b-.',label='x3')
plt.axis([0, 10, -.01, .04])
plt.legend(loc='upper right')
plt.grid()

plt.subplot(413) 

plt.plot(TT,time3,'g-.',label='x4')
plt.plot(TT,time4,'r--',label='x5')
plt.axis([0, 10, -.5, .5])
plt.legend(loc='upper right')
plt.grid()


plt.subplot(414)
plt.plot(TT, y, 'r', label='y(t)')
plt.axis([0, 10, -1, 1])
plt.xlabel("t (sec)")

plt.legend()
plt.grid()
plt.show()

