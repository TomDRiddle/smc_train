import numpy as np
from math import sin,cos
from random import random
class Wind:
    def __init__(self):
        super(Wind,self).__init__()


    def wind(self,t):
        if t >30 and t<50:
            W = 0.094*sin(0.02*t)+0.0072*sin(0.015*t)
            #W = 0
        elif t>70 and t<100:
            W = 0.065*sin(0.018*t)+0.0085*sin(0.03*t)
            #W = 0
        else:
            W = 0

        return W 

class General_Resistance:
    def __init__(self,num_carriage):
        super(General_Resistance,self).__init__()
        self.num_carriage=num_carriage
        self.g = 9.8
        self.pi = 3.14
        self.theta = 0
        self.Le = 0
        self.Lr = 200
        self.Ls = 1000
        self.R = 0
        self.happen = [0,0,0,0]
        self.t = 0
        
    def general_resistance(self):
        # N (not kN)
        Wr = 10.5*((2*self.pi)/3)*self.g/(self.Lr)
        Ws = 0.00013*self.Ls*self.g
        Wi = self.g*sin(self.theta)*1000
        We = 0.08*self.g*sin(0.2*self.t)*cos(0.2*self.t)

        return self.happen[0]*Ws+self.happen[1]*Wr+self.happen[2]*We+self.happen[3]*Wi  

    def railway(self):
        
        if self.t>0 and self.t<=100:
            self.happen = [1,1,1,0]

        elif self.t>100 and self.t<=250:
            self.happen = [1,0,1,0]
    
        elif self.t>250 and self.t<=600:
            self.happen = [0,1,1,0]
        elif self.t>600 and self.t<=1000:
            self.happen = [0,0,1,0]
        else:
            self.happen = [1,1,1,1]    
  
        return self.general_resistance()
    
    def update(self,t):
        self.t = t

        Res = np.zeros(self.num_carriage)
        for i in range(self.num_carriage):

            Res[i] = self.railway()

        return Res

 