
import numpy as np
from train import Train_System
from Silding_mode_controller import Silding_Mode_Controller,Adp_silding_mode_control
from Adaptive_law import Adaptive_Law
from network import Neraul_Network_Controller
import matplotlib.pyplot as plt
from railway2 import A,V,X
t_sampling = 0.001
real_train_conf = {'M':np.array([45,55,48,57]),
                   'Davis_coef':np.array([[4.7,150,1],
                                             [4.600,1.43,0.00013],
                                             [4.800,1.44,0.00012],
                                             [4.300,1.52,0.00012],]),
                   'K':np.array([81000,79000,80000,0]),
                   'Le':400}
virtual_train_conf = {'M':np.array([50,50,50,50]),
                   'Davis_coef':np.array([[4.5,150,1],
                                             [4.500,1.50,0.0001],
                                             [4.500,1.50,0.0001],
                                             [4.500,1.50,0.0001]]),
                   'K':np.array([80000,80000,80000,0]),
                   'Le':400}

num_carriage = len(real_train_conf['M'])    
K1 = 5000
K2 = 100000
C = 1000
p =1
v =1

Ad = np.zeros(num_carriage)
Mrange = [79000,81000]
Davisrange = np.array([[4400,4600],[140,160],[0.9,1.1]])
Wrange = [-1000000,1000000]
Zrange = [-1.5,1.5]
speed = [0,80]
A_list = A(num_carriage,t_sampling)
V_list = V(num_carriage,A_list,t_sampling)
X_list = X(num_carriage,V_list,t_sampling)
#Vd = V_list[:,0]
Vd = np.ones(4)*80
time = np.arange(0,10,t_sampling)
re_l = len(time)
#Ad = ((speed[1]-speed[0])/t)*np.ones(num_carriage)
def NNSMC_run():

    real_train = Train_System(real_train_conf,t_sampling)
    virtual_train = Train_System(virtual_train_conf,t_sampling) 


    smc_controller1 = Silding_Mode_Controller(C,K1,K2)

    nn_controller = Neraul_Network_Controller(p,v)
    adp = Adaptive_Law(t_sampling)

    X_start = np.zeros(num_carriage)
    Bf = np.ones(num_carriage)
    re_l = len(time)
    Vd = V_list[:,0]
    real_train.Sync(X_start,Vd)
    virtual_train.Sync(X_start,Vd)
    refer_F = np.zeros(num_carriage)
    F_hat = np.zeros(num_carriage)
    E_list = np.zeros((num_carriage,re_l))
    l_list = np.zeros(re_l)
    B_F=virtual_train.M.copy()
    for i in range(len(list(time))):

        t = i*t_sampling
        Ad = np.zeros(num_carriage)#A_list[:,i]
        Xd = t*Vd#X_list[:,i]
        Vd = np.ones(num_carriage)#V_list[:,i]

        if t>5:
            Bf[2] = 0
        refer_F= virtual_train.refer_run_force()
        E = real_train.train_location - Xd
        E_dot = real_train.train_velocity - Vd

        
        smc_U,S = smc_controller1.control(E,E_dot,refer_F,F_hat,Ad,B_F)
        target = np.concatenate((Xd,Vd,Ad),axis=0)
        net_input = nn_controller.create_input(real_train.train_location,real_train.train_velocity,real_train.train_acceleration,target,smc_U)
        U = nn_controller.control(S,net_input,smc_U,B_F)
    
        [X,V,A] = real_train.Update(U,Bf,t)
        F_hat = adp.Update(refer_F,A,U,S,B_F) 
        l = nn_controller.Update(real_train.train_velocity,Vd)
        E_list[:,i] = E
        l_list[i] = l
        virtual_train.Sync(X,V)
    print(np.sum(np.abs(E_list)))
    return l_list,E_list 
     
def SMC_run():

    real_train = Train_System(real_train_conf,t_sampling)
    virtual_train = Train_System(virtual_train_conf,t_sampling)

    smc_controller = Silding_Mode_Controller(C,K1,K2)

    adp = Adaptive_Law(t_sampling)



    X_start = np.zeros(num_carriage)
    Bf = np.ones(num_carriage)
    Vd = V_list[:,0]
    re_l = len(time)
    real_train.Sync(X_start,np.ones(num_carriage))
    virtual_train.Sync(X_start,Vd)
    refer_F = np.zeros(num_carriage)
    F_hat = np.zeros(num_carriage)
    E_list = np.zeros((num_carriage,re_l))
    F_hat_l = np.zeros((num_carriage,re_l))
    B_F=virtual_train.M.copy()#np.ones(num_carriage)
    for i in range(len(list(time))):

        t = i*t_sampling
        Xd = np.ones(num_carriage)*80*t
        if t>5:
            Bf[2] = 0
        
        refer_F= virtual_train.refer_run_force()
        E = real_train.train_location - Xd
        E_dot = real_train.train_velocity - Vd
        Ad = np.zeros(num_carriage)#A_list[:,i]
        Xd = t*Vd#X_list[:,i]
        Vd = np.ones(num_carriage)#V_list[:,i]
        U,S = smc_controller.control(E,E_dot,refer_F,F_hat,Ad,B_F)   
        [X,V,A] = real_train.Update(U,Bf,t)
        F_hat = adp.Update(refer_F,A,U,S,B_F) 
        #print("rf",refer_F+F_hat)
        #print(real_train.F_real(t))
        E_list[:,i] = E
        F_hat_l[:,i] = F_hat
        virtual_train.Sync(X,V)

    print(np.sum(np.abs(E_list)))
    return E_list,F_hat_l 

def ALSMC_run():

    real_train = Train_System(real_train_conf,t_sampling)

    smc_controller = Adp_silding_mode_control(1.99,2,150000,1,1,1,0.0001,0.0001,0.0001,1,Mrange,Davisrange,Wrange,Zrange,virtual_train_conf,t_sampling)

    X_start = np.zeros(num_carriage)
    Bf = np.ones(num_carriage)
    #Bf[2] = 0.5

    real_train.Sync(X_start,V_list[:,0])

    E_list = np.zeros((num_carriage,re_l))
    smc_controller.net_init(15)
    for i in range(len(list(time))):

        t = i*t_sampling
        Ad = A_list[:,i]
        Xd = X_list[:,i]
        Vd = V_list[:,i]
        if t>5:
            Bf[2] = 1

        E = real_train.train_location - Xd
        
        print(E)
        print(real_train.train_location)

        smc_controller.Sync(real_train.train_location,real_train.train_velocity,Xd,Vd,Ad)
        U = smc_controller.control()   
        [X_t,V_t,A_t] = real_train.Update(U,Bf,t)
        smc_controller.Update()
        #print("rf",refer_F+F_hat)
        #print(real_train.F_real(t))
        E_list[:,i] = X_t



    print(np.sum(np.abs(E_list)))
    return E_list
l_list,E1 = NNSMC_run()
E2,F_hat_l= SMC_run()
plt.subplot(311)
plt.plot(time,l_list) 

plt.subplot(312)
plt.plot(time,E1[0,:]) 
plt.subplot(313)
plt.plot(time,E2[0,:]) 
plt.show()