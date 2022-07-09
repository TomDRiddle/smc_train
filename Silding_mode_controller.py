import numpy as np
from Differentiator import NTD_m
class Adaptive_Law:
    def __init__(self,t_sampling):
        self.t_sampling = t_sampling
        self.F_hat = 0
        self.F0_TD = NTD_m(t_sampling=self.t_sampling)
        self.U_TD = NTD_m(t_sampling=self.t_sampling)
        self.A_TD = NTD_m(t_sampling=self.t_sampling)

    def dF_hat_law(self,ddx,dF0,S,dU,B0=1):
        return ddx-dF0-S-B0*dU

    def Update(self,F0,A,U,S,B0):
        dF0,_ = self.F0_TD.Update(F0)
        dA,_ = self.A_TD.Update(A)
        dU,_ = self.U_TD.Update(U)
        dF_hat = self.dF_hat_law(dA,dF0,S,dU,B0)
        self.F_hat=self.F_hat+dF_hat*self.t_sampling
        return self.F_hat


class Silding_Mode_Controller:
    def __init__(self,C,K1,K2):
        super(Silding_Mode_Controller,self).__init__()
        self.C = C
        self.K1 = K1
        self.K2 = K2

        self.S = 0
        self.E = 0
        self.E_dot = 0
        self.E_intergrade = 0

    def silding_mode_surface(self,E,E_dot):
        self.E = E
        self.E_dot = E_dot
        self.S = self.C*self.E + self.E_dot
        return self.S

    def control(self,E,E_dot,F0,F_hat,refer_acc,B):
        S = self.silding_mode_surface(E,E_dot)
        #print(S)
        U = - self.K2*np.sign(self.S)-self.K1*self.S - self.C*self.E_dot - (F0 + F_hat - refer_acc)*B
        return U,S

        
class Adp_silding_mode_control:
    def __init__(self,Ks,sigma,alpha,gamma_m,gamma_D,gamma_net,sigma_D,sigma_w,sigma_m,epsilon,Mrange,Davisrange,Wrange,Zrange,train_conf,t_sampling):
        super(Adp_silding_mode_control,self).__init__()
        self.t_sampling = t_sampling
        self.C1 = 200
        self.C2 = 100
        self.Ks = Ks

        self.M = train_conf['M']
        self.Davis_coef = train_conf['Davis_coef']
        self.K = train_conf['K']
        self.num_carriage = len(self.M)

        self.gamma_D = gamma_D
        self.gamma_m = gamma_m
        self.gamma_net = gamma_net
        self.sigma_D = sigma_D
        self.sigma_m = sigma_m
        self.sigma_w = sigma_w

        self.S = np.zeros(self.num_carriage)

        self.E = np.zeros(self.num_carriage)
        self.E_dot = np.zeros(self.num_carriage)
        self.E_intergrade = np.zeros(self.num_carriage)
        self.train_location = np.zeros(self.num_carriage)
        self.train_velocity = np.zeros(self.num_carriage) 
        

        
        self.epsilon = epsilon
        self.net_W = np.zeros(self.num_carriage) 
        self.Z_center = np.zeros(self.num_carriage)
        self.sigma = sigma
        self.alpha = alpha
        
        self.Z = 0
        self.M_range = Mrange
        self.Davis_coef_range = Davisrange
        self.W_range = Wrange
        self.Z_range = Zrange
    
    def sat(self):
        Y= self.S/self.epsilon
        out = np.array([])
        for y in Y:
            if y>1:
                y = 1
            elif y<-1:
                y = -1
            else:
                y = y
            out = np.append(out,y)

        return out

    def act_func(self,z,i):
        return np.exp(-np.dot((z-self.Z_center[i]),(z-self.Z_center[i]))/(2*self.sigma**2))

    def net(self,input):
        self.act = np.zeros(self.num_node)
        for i in range(self.num_node):
            self.act[i] = self.act_func(input,i)
        
        output = np.dot(self.act,self.net_W)
        return output

    def davis_function(self):
        davis_resistance_force = self.Davis_coef[:,0]+self.Davis_coef[:,1]*self.train_velocity+self.Davis_coef[:,2]*self.train_velocity*self.train_velocity
        return davis_resistance_force

    def spring_function(self):
        before_train_location = self.train_location.copy()
        before_train_location[-1] = 0
        after_train_location = self.train_location.copy()
        after_train_location[0] = 0
        spring_force = np.roll(self.K,1)*(np.roll(before_train_location,1)-after_train_location) \
                    - self.K*(before_train_location-np.roll(after_train_location,-1))
        return spring_force

    def Proj(self,x_range,X_dot,X):
        Xd = X_dot.copy()
        for i in range(len(X)):
            if X[i]<=x_range[0] and X_dot[i]<0:
                Xd[i] = 0
            elif X[i]>=x_range[1] and X_dot[i]>0:
                Xd[i] = 0
            else:
                Xd[i] = X_dot[i]
        X_update=X+self.t_sampling*Xd

        return X_update

    def Proj_mat(self,x_range,X_dot,X):

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j]<=x_range[0] and X_dot[i,j]<0:
                    X_dot[i,j] = 0
                elif X[i,j]>=x_range[1] and X_dot[i,j]>0:
                    X_dot[i,j] = 0
                else:
                    X_dot[i,j] = X_dot[i,j]
        X_update=X+self.t_sampling*X_dot

        return X_update        

    def silding_mode_surface(self):
        self.S = self.C1*self.E+self.E_dot+self.C2*self.E_intergrade
        return 0


    def Sync(self,X,V,Xd,Vd,Ad):

        self.train_location = X
        self.train_velocity = V
        self.Ad = Ad
        self.E = X-Xd
        self.E_dot = V-Vd
        self.E_intergrade = self.E_intergrade + self.E*self.t_sampling
        #print(self.E_intergrade)
        self.Z = np.concatenate((self.E,self.E_dot),axis = 0)

    def control(self):
        self.silding_mode_surface()
        U = -self.M*(self.C1*self.E_dot+self.C2*self.E-self.Ad)+self.davis_function()-self.spring_function()- self.Ks*self.S + self.net(self.Z) - self.alpha*self.sat()

        return U

    def Update(self):
        Davis_coef_dot = np.zeros((self.num_carriage,3))
        S_epsilon = self.S - self.epsilon*self.sat()
        M_dot = self.gamma_D*(S_epsilon*(self.C1*self.E_dot-self.Ad+self.C2*self.E)-self.sigma_m*self.M)
        W_dot = -(np.dot(np.linalg.inv(self.GAMMA),np.dot(self.act.reshape((self.num_node,1)),S_epsilon.reshape((1,self.num_carriage))))+self.sigma_w*self.net_W)
        Davis_coef_dot[:,0] = -self.gamma_D*(S_epsilon+self.sigma_D*self.Davis_coef[:,0])
        Davis_coef_dot[:,1] = -self.gamma_D*(S_epsilon*self.train_velocity+self.sigma_D*self.Davis_coef[:,1])
        Davis_coef_dot[:,2] = -self.gamma_D*(S_epsilon*(self.train_velocity**2)+self.sigma_D*self.Davis_coef[:,2])

        self.M = self.Proj(self.M_range,M_dot,self.M)
        self.Davis_coef[:,0] = self.Proj(self.Davis_coef_range[0,:],Davis_coef_dot[:,0],self.Davis_coef[:,0])
        self.Davis_coef[:,1] = self.Proj(self.Davis_coef_range[1,:],Davis_coef_dot[:,1],self.Davis_coef[:,1])
        self.Davis_coef[:,2] = self.Proj(self.Davis_coef_range[2,:],Davis_coef_dot[:,2],self.Davis_coef[:,2])
        self.net_W = self.Proj_mat(self.W_range,W_dot,self.net_W)
        self.M = self.M +self.t_sampling*M_dot
        self.net_W = self.net_W + self.t_sampling*W_dot
        self.Davis_coef = self.Davis_coef+self.t_sampling*Davis_coef_dot
        C1_dot = - self.S*self.E
        C2_dot = - self.S*self.E_intergrade

        self.C1 = self.C1 + self.t_sampling*C1_dot
        self.C2 = self.C2 + self.t_sampling*C2_dot

    def net_init(self,num_node):
        self.num_node = num_node
        Z_inter = (np.abs(self.Z_range[0])+np.abs(self.Z_range[1]))/(self.num_node - 1)
        self.Z_center = np.arange(self.Z_range[0],self.Z_range[1]+Z_inter,Z_inter)

        self.net_W = np.ones((num_node,self.num_carriage))#np.random.normal(0,1,(num_node,self.num_carriage))
        self.GAMMA = np.eye(self.num_node)*self.gamma_net



        
