# -*- coding: utf-8 -*-
"""
Sim
"""
def To_list(matrix):
    """Convert a numpy matrix to a list. If the result is a single row, return it as a flat list."""
    matrix_list = matrix.tolist()
    return matrix_list[0] if len(matrix_list) == 1 else matrix_list
# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle

from Message_ import message

from Mix_Node_ import Mix

from NYM import MixNet

from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing


def Analytical(nn):
    import math
    a = 1/(nn+1)
    b = nn*a
    s = 0
    for i in range(10000):
        s = s - (a*(b**i))*(math.log((a*(b**i))))/(math.log(math.exp(1)))
        
        
    return s
        
def Ent(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent

def Med(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_

class Simulation(object):
    
    def __init__(self,Dict_List,Targets,run,delay1,delay2,d,h,W,U ):
        self.Dict_List = Dict_List
        self.d1 = delay1
        self.d2 = delay2
        self.N_target = Targets
        self.depth = d
        self.hops = h
        self.Wings = W
        self.U = U
        self.N = self.depth*self.hops*self.Wings
        self.run = run



    def Simulator(self,corrupted_Mix,nn): 
        import simpy
        Mixes = [] #All mix nodes

        env = simpy.Environment()    #simpy environment
        capacity=[]
        Capp = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
        for j in range(self.N):# Generating capacities for mix nodes  
            c = simpy.Resource(env,capacity = Capp)
            capacity.append(c)           
        for i in range(self.N):#Generate enough instantiation of mix nodes  
            ll = i +1
            X = corrupted_Mix['PM%d' %ll]
            x = Mix(env,'M%02d' %i,capacity[i],X,self.N_target,self.d1)
            Mixes.append(x)
        
 

        MNet = MixNet(env,Mixes,self.Dict_List,self.depth,self.hops,self.Wings,self.U)  #Generate an instantiation of the mix net
        random.seed(42)  

        Process = Message_Genartion_and_mix_net_processing(env,Mixes,Capp,MNet,self.N_target,self.d2,self.depth,self.hops,self.Wings,self.U,nn)


        env.process(Process.Prc())  #process the simulation

        env.run(until = self.run)  #Running time

        Latencies0 = MNet.LL0
        Latencies = MNet.LL

        Distributions = np.matrix(MNet.EN)
        DT = np.transpose(Distributions)
        ENT = []

        for i in range(self.N_target):
            llll = To_list(DT[i,:])
            ENT.append(Ent(llll))
            
        Distributions_ = np.matrix(MNet.EN0)
        DT0 = np.transpose(Distributions_)
        ENT0 = []

        for i in range(self.N_target):
            llll_ = To_list(DT0[i,:])
            ENT0.append(Ent(llll_))            
            
        return Med([Latencies0,Latencies]),Med([ENT0,ENT])


'''
#Test the function :)
import numpy as np    
d = 3
h = 3
W = 2
U = 3
N= d*h*W
Targets = 100
run = 2
delay1 = 0.05
delay2 = delay1/(10*d)
Iterations = 200

nn = 1000

corrupted_Mix = {}

for i in range(N):
    corrupted_Mix['PM'+str(i+1)] = False


a = np.matrix([[0.3,0.2,0.5],[0.1,0.1,0.8],[0.6,0.4,0]])

b = np.matrix([[0.7,0.25,0.05],[0.1,0.5,0.4],[1/3,1/3,1/3]])


a_ = np.matrix([[0.3,0.5,0.9],[0.1,0.05,0.03],[0.001,0.04,0.04]])

b_ = np.matrix([[0.1,0.2,0.3],[0.3,0.04,0.03],[0.1,0.7,0.035]])

Dict_List = [[a_,b_],[a,b]]

Sim = Simulation(Dict_List,Targets,run,delay1,delay2,d,h,W,U)

data0 = Sim.Simulator(corrupted_Mix, nn)


print(data0[1])


'''



