import numpy as np
from pyparsing import Opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from MLP import MLP
class ControllerNet(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output,activationfunction=torch.sigmoid):
        super(ControllerNet,self).__init__()                               
        self.hidden =nn.Linear(n_feature,n_hidden)
        self.predict =nn.Linear(n_hidden,n_output)

        self.activationfunction=activationfunction
    def forward(self,x,p):
        x = self.activationfunction(self.hidden(x.to(torch.float32)))
        x = self.predict(x)
        x = x*p
        return x  

class Neraul_Network_Controller:
    def __init__(self,p,v,loss_type):
        super(Neraul_Network_Controller,self).__init__()
        self.net = MLP('controlnet',[28,32,16,4])
        self.p = p
        self.v = v
        self.opt = torch.optim.SGD(self.net.parameters(),lr = 0.001)
        self.loss_type = loss_type
    def np2torch(self,np_array):
        return torch.from_numpy(np_array)

    def torch2np(self,Tensor_array):
        return Tensor_array.detach().numpy()

    def create_input(self,loc,speed,acc,target,base_control):
        net_input=np.concatenate((loc,speed,acc,target,base_control),axis=0)
        return self.np2torch(net_input)

    def control(self,S,net_input,base_control,B):
        
        S = self.np2torch(S)
        base_control = self.np2torch(base_control)
        net_output = self.net(net_input,self.p)
        #print('net:',net_output)
        B = self.np2torch(B)
        self.U = B*(torch.tanh(S)*net_output) + base_control
        return self.torch2np(self.U)

    def loss(self,E,E_dot):
        if self.loss_type == 'L1_e':
            l = self.v*torch.matmul(E,self.U)
        elif self.loss_type == 'L1_e_edot':
            l = self.v*torch.matmul(E,self.U)+self.v*torch.matmul(E_dot,self.U)
        elif self.loss_type == 'L2_e_y':
            l = torch.matmul(E,E)+torch.matmul(self.U,self.U)
        else:
            l = self.v*torch.matmul(E,self.U)
        return l

    def create_loss(self,E,E_dot):
        #loss=np.concatenate((E,E_dot),axis=0)
        #return self.np2torch(loss)
        return self.np2torch(E),self.np2torch(E_dot)

    def Update(self,E_loss,E_dot_loss):
        #
        l = self.loss(E_loss,E_dot_loss)

        l.requires_grad_(True)
        self.opt.zero_grad()
        l.backward()
        self.opt.step()
        return l
     