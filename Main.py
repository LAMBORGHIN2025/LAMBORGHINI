# -*- coding: utf-8 -*-
"""
Main: This file provides instructions regarding how to run the experiments described in the main body of the paper.
"""

from LAMBORGHINI import LAM

#++++++++++++++++++++++++++++++++++++++++Initializations++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#\textbf{Note:} If needed, you can change the following settings. However, please ensure that the parameters do not exceed the specified limits:
#d.h.W: Should not exceed 560.


Iterations = 1
d = 64
h = 3
W = 2
 

#.............................E1, E2 and E3 for 1st, 2nd and 3rd claims.......................................................

#-------------------------------------Fig 1,2 and 3-------------------------------------------------------------------------

#Set Input to be 1  to run this experiment 




#.............................E4 for 4th claim.......................................................

#-------------------------------------Tab.1-------------------------------------------------------------------------

#Set Input to be 2  to run this experiment 


#.............................E5 for 5th claim.......................................................

#-------------------------------------Tab.2-------------------------------------------------------------------------
#Set Input to be 3  to run this experiment 




Input = input('Please enter the Input parameter: ')


LAM(Iterations,d,h,W,int(Input))
