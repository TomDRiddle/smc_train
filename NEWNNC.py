# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:37:59 2019

@author: Administrator
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SlidingModeController import SMC
from Differentiator import LTD,NTD
from FAdaptiveLaw import fAL

class ControllerNet(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output,activationfunction=F.sigmoid):
        super(ControllerNet,self).__init__()                               
        self.hidden =nn.Linear(n_feature,n_hidden)
        self.hidden.weight.data.normal_(0,0.1)
        self.predict =nn.Linear(n_hidden,n_output)
        self.predict.weight.data.normal_(0,0.1)
        self.activationfunction=activationfunction
    def forward(self,x,p):
        x = self.activationfunction(self.hidden(x))
        x = self.predict(x)
#        x = torch.relu(x)
        x = x*p
        return x

class NNC():
    def __init__(self,alf,net,optimizer,p,v,k1=30,k2=1.0,GainK=10.0,f=0.0,b0=1.0,r=36,dt=0.01,order=2):
        
        
        self.GainK=GainK
        self.net = net
        self.smc=SMC(k1,k2,GainK=GainK,rlF=alf,order=order)
        self.f=f
        self.b0=b0
        self.dt=dt
        self.al=fAL()
        self.td_u=NTD(r)
        self.td_ddx2=NTD(r)
        self.u_sm=0.0
        self.u_nn=0.0
        self.u=0.0 
        self.s=0.0
        self.optimizer=optimizer
        self.NumbericU =0.0
        self.p=p
        self.v=v
    def U(self,x1,x2,Yd,dYd,ddYd):
        self.u_sm,self.s=self.smc.U(x1,x2,Yd,dYd,ddYd,self.f,self.b0)
        s = torch.Tensor([self.s])
        TS_yd = torch.Tensor([Yd])
        TS_yd = torch.unsqueeze(TS_yd,dim=1)
        TS_y = torch.Tensor([x1])
        TS_y = torch.unsqueeze(TS_y,dim=1)
        TS_usm = torch.Tensor([self.u_sm])
        TS_usm = torch.unsqueeze(TS_usm,dim=1)
        nn_x = torch.cat((TS_y,TS_usm,TS_yd),dim=1)

        self.u_nn = self.net(nn_x,self.p)

        self.u = self.u_sm +torch.tanh(s)*self.u_nn
        u_smnn = self.u.data.numpy()
        u_smnn = u_smnn[0][0]
        self.NumbericU= u_smnn
        return u_smnn
    def Update(self,x1,x2,dx1,dx2,Yd,dYd,ddYd):
       
        nn_Yd = torch.Tensor([Yd])
        nn_Yd = torch.unsqueeze(nn_Yd,dim=1)
        Y = torch.Tensor([x1])
        loss = self.v*(Y-nn_Yd)*self.u
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.du,_=self.td_u.Update(self.NumbericU)
        
        if self.smc.order==2: 
            _,self.ddx2=self.td_ddx2.Update(x2)
        if self.smc.order==1:
            _,self.ddx2=self.td_ddx2.Update(x1)
        
        df=self.al.dF2(self.ddx2,self.du,self.s,b0=self.b0)
        self.f+=df*self.dt

        
