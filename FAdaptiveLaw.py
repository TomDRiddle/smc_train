# -*- coding: utf-8 -*-

def LowPass(x,oldy,alpha=0.005):
    return alpha*x +(1.0-alpha)*oldy  


class fAL:
    def __init__(self,dt=0.01):
        self.dt=dt
        self.df=0.0
        self.x_1=0.0
        self.x_2=0.0
        self.oldu=0.0
        self.oldx=0.0
        self.dx=0.0
        self.olddx=0.0
        
    def dF(self,x,u,s,b=1.0):
        
        x_=LowPass(x,self.oldx)
        
        self.dx = (x_-self.oldx)/self.dt
        
        self.dx= LowPass(self.dx,self.olddx)
        
        ddx = (self.dx -self.olddx)/self.dt    
#        ddx =( x_ - 2*self.x_1 + self.x_2)/self.dt/self.dt
        
        u_=LowPass(u,self.oldu)
        du = (u_-self.oldu)/self.dt
        
        self.x_2=self.x_1
        self.x_1=x_
        self.oldu=u_
        self.oldx=x_
        self.olddx=self.dx
        
        
        df = ddx - b*du - s
        return df
    
    def dF2(self,ddx,du,s,b0=1.0):
        df = ddx - b0*du - s
        return df