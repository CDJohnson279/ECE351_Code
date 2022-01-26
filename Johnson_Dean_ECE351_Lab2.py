# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 1                                                         #
# 2/1/2022                                                      #
#                                                               #        
#                                                               #
#################################################################


#            Part 1
####################################

#TASK 1.1 

import numpy as np
import matplotlib.pyplot as plt


#plt.rcParams.update ({'fontsize': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t = np . arange (0 , 10 + steps , steps ) # Add a step size to make sure the
                                         # plot includes 5.0. Since np. arange () only
                                         # goes up to , but doesn ’t include the
                                         # value of the second argument





print ('Number of elements : len(t) = ', len( t ) , '\ nFirst Element : t[0] = ', t [0] , 
'\ nLast Element : t[len(t) - 1] =' , t [len( t ) - 1])

#TASK 1.2

def func1(t) : # The only variable sent to the function is t
    y = np.zeros(t.shape) #initialze y(t) as an array of zeros

    for i in range (len(t)) : # run the loop once for each index of t
       
            y[i] = np.cos(t[i])
    return y # send back the output stored in an array

y = func1(t)

plt . figure (figsize = (10 , 7) )
plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('Task 2 of Part 1')

t = np.arange (0 , 10 + .5, 0.5) # redefine t with poor resolution
y = func1(t)

plt.subplot(2,1,2)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t) with Poor Resolution ')
plt.xlabel('t')
plt.show()

#           Part 2
################################

#TASK 2.1 
#Equation 
#  y = r(t) - r(t-3) + 5*u(t-3) - 2*u(t-6) - 2*r(t-6) + 2*r(t-10)

#TASK 2.2 The user defined functions
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

#Complete function
t = np.arange(-5, 10 + steps, steps)
def func2(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y = ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6) + 2*ramp(t-10)
    return y

y = func2(t)

plt.figure(figsize = (10, 7))
#plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Plot for Lab 2')



#         Part 3
######################################

#Task 1 Time reversal Plot

y = -func2(t)

plt.figure(figsize = (10, 7))
#plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Reversal Plot for Part 3')

# #Task 2 time shifts f(t-4) and f(-t-4)

y = func2(t-4)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(t-4) Plot')
plt.xlabel('t')
plt.title('Part 3 Task 2')

t = np.arange(-15, 0 + steps, steps)
y = func2(-t-4)
plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(-t-4) Plot')
plt.xlabel('t')
plt.title('')

# #Task 3 time scale operations f(t/2) and f(2t)
t = np.arange(0, 20 + steps, steps)
y = func2(t/2)

plt.figure(figsize = (10, 7))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.grid()
plt.ylabel('f(t-4) Plot')
plt.xlabel('t')
plt.title('f(t/2)')

t = np.arange(0, 5 + steps, steps)
y = func2(2*t)
plt.figure(figsize = (10, 7))
plt.subplot(2,1,2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('f(2t)')

#Task 5 numpy.diff() 
steps = 1e-3
t = np.arange(-5, 10 + steps, steps)
arr = np.array(func2(t))
dt = np.diff(arr)
dy = np.diff(arr)
plt.figure(figsize = (10, 7))
plt.plot(t[:-1], dy/dt)
plt.ylim(-3,10)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Derivative Plot')
