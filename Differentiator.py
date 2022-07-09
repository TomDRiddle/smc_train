# -*- coding: utf-8 -*-
"""
微分器
"""
import numpy as np

# ↓《自抗扰控制技术——估计补偿不确定因素的控制技术》(2.4.5)式
def fhan(x1,x2,r,h):
      d=r*h
      d0=h*d
      y=x1+h*x2
      absy=np.abs(y) # |y|
      a0=np.sqrt(d*d+8*r*absy)
      
#      alpha = np.round(absy>d0)
#      beta  = np.round(absy<=d0)
#      a = x2 + alpha* (a0-d)/2*np.sign(y) + beta*y/h
      if absy>d0:
            a = x2 + (a0-d)/2*np.sign(y)
      else:
            a= x2 + y/h
      
      absa=np.abs(a)
#      flag1 = np.round(absa>d)
#      flag2 = np.round(absa<=d)
#      valf= -r*(flag1*np.sign(a) + flag2*a/d)
      
      if absa > d:
            valf = -r*np.sign(a)
      else:
            valf = -r*a/d
      
      return valf            
# end

class LTD():
    def __init__(self,r=20,h=0.01):
        self.v1=0.0
        self.v2=0.0
        self.h=h
        self.r=r
    
    def Update(self,v):
        dv1 = self.v2
        dv2 =  -self.r**2 *(self.v1-v ) - 2*self.r*self.v2 
        
        self.v1+=self.h*dv1
        self.v2+=self.h*dv2
        return dv1,dv2

class NTD():
    def __init__(self,r=20,h=0.003,t_sampling=0.01):
        self.v1=0.0
        self.v2=0.0
        self.t_sampling=t_sampling
        self.h=h
        self.r=r    
    
    def Update(self,v):
        dv1 = self.v2
        dv2 = fhan(self.v1-v,self.v2,self.r,self.h)
        
        self.v1+=self.t_sampling*dv1
        self.v2+=self.t_sampling*dv2
        return dv1,dv2
        
class NTD_m():
    def __init__(self,num_carriage,r=1000,h=0.03,t_sampling=0.01):
        self.v1=np.zeros(num_carriage)
        self.v2=np.zeros(num_carriage)
        self.t_sampling=t_sampling
        self.h=h
        self.r=r    
    
    def Update(self,v):
        dv1 = self.v2
        dv2 = np.array([])
        for i in range(len(v)):
              d = fhan(self.v1[i]-v[i],self.v2[i],self.r,self.h)
              dv2 = np.append(dv2,d)

        self.v1+=self.t_sampling*dv1
        self.v2+=self.t_sampling*dv2
        return dv1,dv2    
if __name__ =='__main__':
    ltd=LTD()
    ntd=NTD()        