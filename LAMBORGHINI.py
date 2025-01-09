# -*- coding: utf-8 -*-
"""
LAMBORGHINI: In this file, we provide the main function used to process the clients in LAMBORGHINI.
"""
from Main_F import Carmix 
from Fancy_Plot import PLOT 
import numpy  as np
import os

def To_list(matrix):
    """Convert a numpy matrix to a list. If the result is a single row, return it as a flat list."""
    matrix_list = matrix.tolist()
    return matrix_list[0] if len(matrix_list) == 1 else matrix_list

def devide_list(L1,L2):
    return [L1[i]/L2[i] for i in range(len(L1))]


#print(devide_list([1,2,3],[4,5,6]))

def Norm(list_,number):
    import numpy as np
    x = np.matrix(list_)+ number
    
    return To_list(x)



def print_cost_table(list1, list2, list3, list4):
    # Table headers
    headers = ["Cost Parameter θ", "0.0", "0.05", "0.1", "0.15", "0.2", "0.3"]
    sub_headers = ["H(r), LAS", "H(r), GPR", "H(r), WPR", "H(r), LAR"]

    # Helper function to print separator lines
    def print_separator():
        print("+" + "-" * 20 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+")

    # Print the table
    print_separator()
    print(f"| {headers[0]:<20} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<8} | {headers[5]:<8} | {headers[6]:<8} |")
    print_separator()

    # Print rows for each sub-header and list
    for sub_header, data in zip(sub_headers, [list1, list2, list3, list4]):
        print(f"| {sub_header:<20} | {data[0]:<8.2f} | {data[1]:<8.2f} | {data[2]:<8.2f} | {data[3]:<8.2f} | {data[4]:<8.2f} | {data[5]:<8.2f} |")
        print_separator()



def print_table(list1,list2,list3):

    # Table headers
    headers = ["Tuning Parameter τ", "0.0", "0.6", "1.0"]
    sub_headers = ["Number of wings", "W=1", "W=2", "W=1", "W=2", "W=1", "W=2"]

    # Helper function to print separator lines
    def print_separator():
        print("+" + "-" * 25 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+" + "-" * 8 + "+")

    # Print the table
    print_separator()
    print(f"| {headers[0]:<25} | {headers[1]:<8} | {headers[1]:<8} | {headers[2]:<8} | {headers[2]:<8} | {headers[3]:<8} | {headers[3]:<8} |")
    print_separator()
    print(f"| {'':<25} | {sub_headers[1]:<8} | {sub_headers[2]:<8} | {sub_headers[3]:<8} | {sub_headers[4]:<8} | {sub_headers[5]:<8} | {sub_headers[6]:<8} |")
    print_separator()

    # Data categories and rows
    categories = [
        ("Advanced Corruption b=10", list1),
        ("Advanced Corruption b=1", list2),
        ("Sloppy Corruption", list3)
    ]

    for category, data in categories:
        print(f"| {category:<25} | {data[0][0]:<8.3f} | {data[1][0]:<8.3f} | {data[0][1]:<8.3f} | {data[1][1]:<8.3f} | {data[0][2]:<8.3f} | {data[1][2]:<8.3f} |")
        print_separator()

class LAM(object):
    #In this class we are gonna make an instantiation of the  mix net
    def __init__(self,Iterations, d,h,W,EXP):
        self.Iterations = Iterations
        self.d = d
        self.h = h
        self.W = W
        self.N = self.d*self.h*self.W
        self.EXP = EXP
        if not os.path.exists('Figures'):
            os.mkdir(os.path.join('', 'Figures'))
        
        self.F = 'Figures'
        if EXP== 1:
            self.Entropy_Latency()
        elif EXP==2:
            self.Adversary()
        elif EXP==3:
            self.Noise()
        
        
        
        
    def Entropy_Latency(self):
        Targets = 200
        run = 0.5
        delay1 = 0.05
        delay2 = delay1/self.d
        Iterations = 200
        
        Mix_Threshold = 20
        nn = 200
        corrupted_Mix = {}
        for i in range(self.N):
            corrupted_Mix['PM'+str(i+1)] = False
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        C = Carmix(self.d,self.h,self.W,Targets,run,delay1,delay2,Mix_Threshold,corrupted_Mix) 
        
        data0 = C.Latency_Entropy(T_List,self.Iterations)


        #################################################################################    
        #############################Latency#############################################
        #################################################################################
            
        #################################Latency LAR######################################
        data0_ = data0['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = self.F+'/Fig_2a.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'Vanilla,$W = 1$',r'LONA,$W = 2$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        
        #############################Latency EXP############################
        data0_ = data0['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = self.F+'/Fig_2b.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'Vanilla,$W = 1$',r'LONA,$W = 2$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        
        #############################Latency GPR############################
        data0_ = data0['GPR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = self.F+'/Fig_2c.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'Vanilla,$W = 1$',r'LONA,$W = 2$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        
        
        #############################Latency LAS############################
        data0_ = data0['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = self.F+'/Fig_2d.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'Vanilla,$W = 1$',r'LONA,$W = 2$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0_['L_WW_Random'],data0_['L_W_Random'],data0_['L_WW_LONA'],data0_['L_W_LONA']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.scatter_line(True,limit)
        
        
        
        #################################################################################    
        #############################Entropy#############################################
        #################################################################################
            
        #################################Entropy LAR######################################
        data0_ = data0['LAR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name =self.F+'/Fig_3a.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'LONA,$W = 2$',r'Vanilla,$W = 1$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 6.2
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        #Entropy_Plot.Place = 'lower left'
        Entropy_Plot.colors[1] = 'blue'
        Entropy_Plot.colors[2] = 'fuchsia'
        Entropy_Plot.Line_style[1] = '--'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.scatter_line(True,limit)
        
        
        #############################Entropy EXP############################
        data0_ = data0['EXP']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = self.F+'/Fig_3b.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'LONA,$W = 2$',r'Vanilla,$W = 1$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy= [data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 6.2
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        
        #Entropy_Plot.Place = 'lower left'
        Entropy_Plot.colors[1] = 'blue'
        Entropy_Plot.colors[2] = 'fuchsia'
        Entropy_Plot.Line_style[1] = '--'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.scatter_line(True,limit)
        
        
        #############################Entropy GPR############################
        data0_ = data0['GPR']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = self.F+'/Fig_3c.png'
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'LONA,$W = 2$',r'Vanilla,$W = 1$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 6.2
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Entropy_Plot.Place = 'lower left'
        Entropy_Plot.colors[1] = 'blue'
        Entropy_Plot.colors[2] = 'fuchsia'
        Entropy_Plot.Line_style[1] = '--'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.scatter_line(True,limit)
        
        
        
        
        #############################Entropy LAS############################
        data0_ = data0['LAS']
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = self.F+'/Fig_3d.png'
        
        
        Descriptions = [r'Vanilla,$W = 2$' ,r'LONA,$W = 2$',r'Vanilla,$W = 1$',r'LONA,$W = 1$']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [data0_['E_WW_Random'],data0_['E_WW_LONA'],data0_['E_W_Random'],data0_['E_W_LONA']]
        
        limit = 6.2
        
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        
        Entropy_Plot.Place = 'lower left'
        Entropy_Plot.colors[1] = 'blue'
        Entropy_Plot.colors[2] = 'fuchsia'
        Entropy_Plot.Line_style[1] = '--'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.scatter_line(True,limit)


        #############################Frac Especial############################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = self.F+'/Fig_4c.png'
        Descriptions = ['GPR','GWR' ,'LAS','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Entropy = [devide_list(data0['GPR']['E_W_LONA'],data0['GPR']['L_W_LONA']),devide_list(data0['EXP']['E_W_LONA'],data0['EXP']['L_W_LONA']),devide_list(data0['LAS']['E_W_LONA'],data0['LAS']['L_W_LONA']),devide_list(data0['LAR']['E_W_LONA'],data0['LAR']['L_W_LONA'])]
        
        limit = 120
        Entropy_Plot = PLOT(T_List,Entropy, Descriptions, X_Label, Y_Label, name, condition=False)
        Entropy_Plot.Place = 'lower right'
        Entropy_Plot.colors[0] = 'blue'
        Entropy_Plot.colors[1] = 'cyan'
        Entropy_Plot.colors[2] = 'red'
        Entropy_Plot.colors[3] = 'fuchsia'
        
        Entropy_Plot.Line_style[0] = '--'
        Entropy_Plot.Line_style[1] = '-.'
        Entropy_Plot.Line_style[2] = '-'
        Entropy_Plot.Line_style[3] = ':'
        Entropy_Plot.scatter_line(True,limit)
        
        
        
        
        #################################################################################    
        #############################Latency Balance#############################################
        #################################################################################
            
        #################################Latency W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Latency ($l_P$)" 
        name = self.F+'/Fig_4a.png'
        
        Descriptions = ['LAS','LAR' ,'GPR','GWR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['L_W_LONA_B'],data0['LAR']['L_W_LONA_B'],data0['GPR']['L_W_LONA_B'],data0['EXP']['L_W_LONA_B']]
        
        limit = 0.55
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[1] = 'fuchsia'
        Latency_Plot.scatter_line(True,limit)
        
        
        
        
            
        #################################Entropy W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy ($\mathsf{H}(r)$)" 
        name = self.F+'/Fig_4b.png'
        
        Descriptions = ['LAS' ,'GPR','GWR','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [data0['LAS']['E_W_LONA_B'],data0['GPR']['E_W_LONA_B'],data0['EXP']['E_W_LONA_B'],data0['LAR']['E_W_LONA_B']]
        
        limit = 6.2
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[1] = 'blue'
        Latency_Plot.colors[2] = 'cyan'
        Latency_Plot.colors[3] = 'fuchsia'
        
        Latency_Plot.Line_style[1] = '--'
        Latency_Plot.Line_style[2] = '-.'
        Latency_Plot.Line_style[3] = ':'
        
        Latency_Plot.Place = 'lower right'
        #Latency_Plot.colors[1] = 'fuchsia'
        Latency_Plot.scatter_line(True,limit)
        
        
        
        
            
        #################################Frac W######################################
        X_Label = r"Tuning parameter ($\tau$)"
        Y_Label = r"Entropy/Latency ($\frac{\mathsf{H}(r)}{l_P}$)" 
        name = self.F+'/Fig_4d.png'
        
        Descriptions = ['GPR','GWR' ,'LAS','LAR']
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        Latency = [devide_list(data0['GPR']['E_W_LONA_B'],data0['GPR']['L_W_LONA_B']),devide_list(data0['EXP']['E_W_LONA_B'],data0['EXP']['L_W_LONA_B']),devide_list(data0['LAS']['E_W_LONA_B'],data0['LAS']['L_W_LONA_B']),devide_list(data0['LAR']['E_W_LONA_B'],data0['LAR']['L_W_LONA_B'])]
        
        limit = 120
        Latency_Plot = PLOT(T_List,Latency, Descriptions, X_Label, Y_Label, name, condition=False)
        Latency_Plot.colors[0] = 'blue'
        Latency_Plot.colors[1] = 'cyan'
        Latency_Plot.colors[2] = 'red'
        Latency_Plot.colors[3] = 'fuchsia'
        
        Latency_Plot.Line_style[0] = '--'
        Latency_Plot.Line_style[1] = '-.'
        Latency_Plot.Line_style[2] = '-'
        Latency_Plot.Line_style[3] = ':'
        
        Latency_Plot.Place = 'lower right'
        #Latency_Plot.colors[1] = 'fuchsia'
        Latency_Plot.scatter_line(True,limit)
        
    def Adversary(self):
        Targets = 200
        run = 0.5
        delay1 = 0.05
        delay2 = delay1/self.d
        Iterations = 200
        
        Mix_Threshold = 20
        nn = 200
        corrupted_Mix = {}
        for i in range(self.N):
            corrupted_Mix['PM'+str(i+1)] = False
        
        T_List = [0,0.2,0.4,0.6,0.8,1]
        C = Carmix(self.d,self.h,self.W,Targets,run,delay1,delay2,Mix_Threshold,corrupted_Mix) 
        
        data0 = C.Adversary_Analysis(T_List,self.Iterations)
        
        l11 = [data0['Advance']['FCP_W_LONA'][0],data0['Advance']['FCP_W_LONA'][3],data0['Advance']['FCP_W_LONA'][5]]
        l12 = [data0['Advance']['FCP_WW_LONA'][0],data0['Advance']['FCP_WW_LONA'][3],data0['Advance']['FCP_WW_LONA'][5]]  
        l1 = [l11,l12]

        l21 = [data0['Greedy']['FCP_W_LONA'][0],data0['Greedy']['FCP_W_LONA'][3],data0['Greedy']['FCP_W_LONA'][5]]
        l22 = [data0['Greedy']['FCP_WW_LONA'][0],data0['Greedy']['FCP_WW_LONA'][3],data0['Greedy']['FCP_WW_LONA'][5]]  
        l2 = [l21,l22]    
        
        l31 = [data0['Random']['FCP_W_LONA'][0],data0['Random']['FCP_W_LONA'][3],data0['Random']['FCP_W_LONA'][5]]
        l32 = [data0['Random']['FCP_WW_LONA'][0],data0['Random']['FCP_WW_LONA'][3],data0['Random']['FCP_WW_LONA'][5]]  
        l3 = [l31,l32]         



        # Call the function
        print_table(l1,l2,l3)

    def Noise(self):
        Targets = 200
        run = 0.5
        delay1 = 0.05
        delay2 = delay1/self.d
        Iterations = 200
        
        Mix_Threshold = 20
        nn = 200
        corrupted_Mix = {}
        for i in range(self.N):
            corrupted_Mix['PM'+str(i+1)] = False
        
        T_List = [0,0.05,0.1,0.15,0.2,0.3]
        C = Carmix(self.d,self.h,self.W,Targets,run,delay1,delay2,Mix_Threshold,corrupted_Mix) 
        
        data0 = C.Noise_Latency_Entropy(T_List,self.Iterations)
        
        
        list1 = data0['LAS']['E_W_LONA_Noise']
        list2 = data0['GPR']['E_W_LONA_Noise']
        list3 = data0['EXP']['E_W_LONA_Noise']
        list4 = data0['LAR']['E_W_LONA_Noise']
        # Call the function
        print_cost_table(list1, list2, list3, list4)

#EXP = LAM(100,64,3,2,3)















