# ###############################################################
#                                                               #
# Dean Johnson                                                  #
# ECE351                                                        #
# Lab 11                                                    #
# 4/5/2022                                                      #
#                                                               #        
#                                                               #
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack

#y[k] = 2x[k] − 40x[k − 1] + 10y[k − 1] − 16y[k − 2]

### Task 1 ###

# H(z) = (2z(z-20)/((z-2)(z-8))) = (2z^2-40z)/((z^2-10z+16))
numx = [2, -40, 0]
deny = [1,-10, 16]

r,p,k = sig.residuez(numx,deny)
 
print("r =", r)
print("p = ", p)

def zplane (b , a , filename = None ) :
 """ Plot the complex z- plane given a transfer function """

 import numpy as np
 import matplotlib . pyplot as plt
 from matplotlib import patches

 # get a figure / plot
 ax = plt . subplot (1 , 1 , 1)

 # create the unit circle
 uc = patches . Circle ((0 ,0) , radius =1 , fill = False , color ="black", ls ="dashed")
 ax . add_patch ( uc )

 # the coefficients are less than 1 , normalize the coefficients
 if np . max( b ) > 1:
     kn = np . max( b )
     b = np . array ( b ) / float ( kn )
 else :

     kn = 1

 if np . max( a ) > 1:
     kd = np . max( a )
     a = np . array ( a ) / float ( kd )
 else :
     kd = 1

 # get the poles and zeros
 p = np . roots ( a )
 z = np . roots ( b )
 k = kn / float ( kd )

 # plot the zeros and set marker properties
 t1 = plt . plot ( z . real , z . imag , "o", ms =10 , label ="Zeros")
 plt . setp ( t1 , markersize =10.0 , markeredgewidth =1.0)

 # plot the poles and set marker properties
 t2 = plt . plot ( p . real , p . imag , "x", ms =10 , label ="Poles")
 plt . setp ( t2 , markersize =12.0 , markeredgewidth =3.0)

 ax . spines ["left"]. set_position ("center")
 ax . spines ["bottom"]. set_position ("center")
 ax . spines ["right"]. set_visible ( False )
 ax . spines ["top"]. set_visible ( False )

 plt . legend ()

 # set the ticks

 # r = 1.5; plt. axis ( ’ scaled ’); plt. axis ([ -r, r, -r, r])
 # ticks = [ -1 , -.5 , .5 , 1]; plt. xticks ( ticks ); plt. yticks ( ticks )

 if filename is None :
     plt . show ()
 else :
     plt . savefig ( filename )

 return z , p , k

z, p, k = zplane(numx, deny)

w, h = sig.freqz(numx, deny, whole=True)

plt.subplots(2)
plt.tight_layout()
plt.subplot(2,1,1)
plt.plot(w/np.pi, abs(h))
plt.title('Magnitude of H(z)')
plt.grid()

plt.subplot(2,1,2)
plt.plot(w/np.pi, np.angle(h))
plt.title('Phase of H(z)')
plt.grid()
