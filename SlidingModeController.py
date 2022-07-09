# -*- coding: utf-8 -*-
import numpy as np

#趋近律
def defaultRLF(s):
    return s

def RLF1(s):
    return np.tanh(s)



class SMC:
    def __init__(self,k1=2.0,k2=1,GainK=10,rlF=defaultRLF,order=2):
        self.K1=k1
        self.K2=k2
        self.rlF=rlF
        self.order=order
        self.GainK=GainK
    def U(self,x1,x2,yd,dyd,ddyd,f,b0=1):
        
        e=yd-x1
        de=dyd-x2
        if self.order==2:
            s= self.K1*e + self.K2*de
            M = self.K1*de + ddyd
        if self.order==1:
            s = self.K1*e
            M = dyd   
            
        u = ( M - f + self.GainK*self.rlF(s)) / b0

        return u,s
    