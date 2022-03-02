# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 7                                                       #
# 3/1/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

#Initial code from previous lab with ramp and step functions as well as import scipy and math
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-2 # Define step size
t = np . arange (0 , 20 + steps , steps ) 

### Part 1 ### 
# Gs = (s+9)/((s-8)*(s+2)*(s+4))
# As= (s+4)/((s+3)*(s+1))
# Bs = (s+14)*(s+12)

zg,pg,gg = sig.tf2zpk([1,9], sig.convolve([1,-6,-16],[1,4]))
print("G(s) Poles and Zeroes")
print("Zeroes = ",zg)
print("Poles = ",pg)
print("\n")

za, pa, ga = sig.tf2zpk([1,4], [1,4,3])
print("A(s) Poles and zeroes")
print("Zeroes = ",za)
print("Poles = ",pa)
print(ga)
print("\n")

broots = np.roots([1,26,168])
print("B(s) Roots")
print("Roots = ",broots)
print("\n")
system_open = [1,9],sig.convolve([1,4,3],[1,-6,-16])
print("Denomanator = ",sig.convolve([1,4,3],[1,-6,-16]))
t, step = sig.step(system_open)

plt.figure(figsize = (10, 7))
#plt.subplot(2,1,1)
plt.plot(t, step)
plt.grid()
plt.ylabel('Step')
plt.xlabel('t')
plt.title('Total Transfer Function')

print("\n")
print("\n")
### Part 2 ###

numG = [1,9]
denG = sig.convolve([1,-6,-16],[1,4])
numA = [1,4]
denA = [1,4,3]
numB = [1,26,168]
denB = [1]


numTotal = sig.convolve(numG, numA)
denTotal = sig.convolve(denA, denG) + sig.convolve(sig.convolve(denA, numB), numG)

print("Total Num = ",numTotal)
print("Total Den = ",denTotal)

transfer_closed = numTotal, denTotal
t, step_closed = sig.step(transfer_closed)

plt.figure(figsize = (10, 7))
#plt.subplot(2,1,1)
plt.plot(t, step_closed)
plt.grid()
plt.ylabel('Step')
plt.xlabel('t')
plt.title('Closed Loop H(t)')