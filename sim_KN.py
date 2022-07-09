import numpy as np
from train_KN import Train_System
from Silding_mode_controller_KN import Silding_Mode_Controller
from network import Neraul_Network_Controller
import matplotlib.pyplot as plt
from railway import A,V,X

t_sampling = 0.001
real_train_conf = {'M':np.array([45,55,48,57]),
                   'Davis_coef':np.array([[4.7,1.50,0.00013],
                                             [4.6,1.43,0.00013],
                                             [4.8,1.44,0.00012],
                                             [4.3,1.52,0.00012],]),
                   'K':np.array([80000,80000,80000,0]),
                   'Le':400}
virtual_train_conf = {'M':np.array([50,50,50,50]),
                   'Davis_coef':np.array([[4.5,1.50,0.0001],
                                             [4.5,1.50,0.0001],
                                             [4.5,1.50,0.0001],
                                             [4.5,1.50,0.0001]]),
                   'K':np.array([80000,80000,80000,0]),
                   'Le':400}

num_carriage = len(real_train_conf['M'])    
K2 = 100
K1 = 10000
C =50
p =1
v =1
Bf = np.ones(num_carriage)
Bf[2]=0
A_list = A(num_carriage,t_sampling)
V_list = V(num_carriage,A_list,t_sampling)
X_list = X(num_carriage,V_list,t_sampling)
Vd = V_list[:,0]
time = np.arange(0,1300,t_sampling)
re_l = len(time)

def NNSMC_e_run():

    real_train = Train_System(real_train_conf,t_sampling)
    smc_controller1 = Silding_Mode_Controller(C,K1,K2,virtual_train_conf,t_sampling)
    nn_controller = Neraul_Network_Controller(p,v,'L1_e')

    X_start = np.zeros(num_carriage)
    
    re_l = len(time)
    Vd = V_list[:,0]
    real_train.Sync(X_start,Vd)

    E_list = np.zeros((num_carriage,re_l))
    l_list = np.zeros(re_l)
    B_F=virtual_train_conf['M']
    X=np.zeros(num_carriage)
    V=np.zeros(num_carriage)
    A=np.zeros(num_carriage)
    for i in range(len(list(time))):

        t = i*t_sampling
        Ad = A_list[:,i]
        Xd = X_list[:,i]
        Vd = V_list[:,i]

        smc_controller1.Sync(X,V,A,Xd,Vd,Ad)
        
        smc_U,S = smc_controller1.control()
        target = np.concatenate((Xd,Vd,Ad),axis=0)
        net_input = nn_controller.create_input(real_train.train_location,real_train.train_velocity,real_train.train_acceleration,target,smc_U)
        U = nn_controller.control(S,net_input,smc_U,B_F)
        [X,V,A] = real_train.Update(U,Bf,t)        
        smc_controller1.Update()

        E = real_train.train_location - Xd
        E_dot = real_train.train_velocity - Vd        
        E_loss,E_dot_loss = nn_controller.create_loss(E,E_dot)

        l = nn_controller.Update(E_loss,E_dot_loss)

        E_list[:,i] = E
        l_list[i] = l
        
        if t%100==0:
            print(t)
            #print(U)


    print(np.sum(np.abs(E_list)))
    return l_list,E_list 

def NNSMC_eedot_run():

    real_train = Train_System(real_train_conf,t_sampling)
    smc_controller1 = Silding_Mode_Controller(C,K1,K2,virtual_train_conf,t_sampling)
    nn_controller = Neraul_Network_Controller(p,v,'L1_e_edot')


    X_start = np.zeros(num_carriage)

    re_l = len(time)
    Vd = V_list[:,0]
    real_train.Sync(X_start,Vd)
    E_list = np.zeros((num_carriage,re_l))
    l_list = np.zeros(re_l)
    B_F=virtual_train_conf['M']
    X=np.zeros(num_carriage)
    V=np.zeros(num_carriage)
    A=np.zeros(num_carriage)
    for i in range(len(list(time))):

        t = i*t_sampling
        Ad = A_list[:,i]
        Xd = X_list[:,i]
        Vd = V_list[:,i]

        E = real_train.train_location - Xd
        E_dot = real_train.train_velocity - Vd
        smc_controller1.Sync(X,V,A,Xd,Vd,Ad)
        
        smc_U,S = smc_controller1.control()
        target = np.concatenate((Xd,Vd,Ad),axis=0)
        net_input = nn_controller.create_input(real_train.train_location,real_train.train_velocity,real_train.train_acceleration,target,smc_U)
        U = nn_controller.control(S,net_input,smc_U,B_F)

        [X,V,A] = real_train.Update(U,Bf,t)
        
        E_loss,E_dot_loss = nn_controller.create_loss(E,E_dot)
        l = nn_controller.Update(E_loss,E_dot_loss)
        smc_controller1.Update()
        

        E_list[:,i] = E
        l_list[i] = l
        if t%100==0:
            print(t)

    print(np.sum(np.abs(E_list)))
    return l_list,E_list 
     
def SMC_run():

    real_train = Train_System(real_train_conf,t_sampling)
    smc_controller = Silding_Mode_Controller(C,K1,K2,virtual_train_conf,t_sampling)
    X_start = np.zeros(num_carriage)

    Vd = V_list[:,0]
    re_l = len(time)
    real_train.Sync(X_start,np.ones(num_carriage))
    F_hat = np.zeros(num_carriage)
    E_list = np.zeros((num_carriage,re_l))
    F_hat_l = np.zeros((num_carriage,re_l))
    X=np.zeros(num_carriage)
    V=np.zeros(num_carriage)
    A=np.zeros(num_carriage)
    for i in range(len(list(time))):
        Ad = A_list[:,i]
        Xd = X_list[:,i]
        Vd = V_list[:,i]
        t = i*t_sampling

        E = real_train.train_location - Xd
        E_dot = real_train.train_velocity - Vd
        smc_controller.Sync(X,V,A,Xd,Vd,Ad)
        U,S = smc_controller.control()   
        [X,V,A] = real_train.Update(U,Bf,t)
        smc_controller.Update()
        #print("rf",refer_F+F_hat)

        E_list[:,i] = E
        F_hat_l[:,i] = F_hat
        if t%100==0:
            print(t)

    print(np.sum(np.abs(E_list)))
    return E_list,F_hat_l 


_,E1 = NNSMC_e_run()
_,E2 = NNSMC_eedot_run()
E3,F_hat_l= SMC_run()
plt.subplot(311)
plt.plot(time,E1[0,:]) 
plt.subplot(312)
plt.plot(time,E2[0,:]) 
plt.subplot(313)
plt.plot(time,E3[0,:]) 

plt.show()

