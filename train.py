import numpy as np
from math import sin,cos
from Resistance import Wind,General_Resistance

class Train_System:
    def __init__(self,train_conf,t_sampling):
        super(Train_System,self).__init__()
        self.t_sampling = t_sampling

        self.Le = train_conf['Le']
        self.M = train_conf['M']
        self.Davis_coef = train_conf['Davis_coef']
        self.K = train_conf['K']
        self.num_carriage = len(self.M)

        self.train_location = np.zeros(self.num_carriage)
        self.train_velocity = np.zeros(self.num_carriage)
        self.train_acceleration = np.zeros(self.num_carriage)

        self.general_resistance = General_Resistance(self.num_carriage)
        
    def davis_function(self):
        davis_resistance_force = self.Davis_coef[:,0]+self.Davis_coef[:,1]*self.train_velocity+self.Davis_coef[:,2]*self.train_velocity*self.train_velocity
        #print(self.train_velocity)
        return davis_resistance_force

    def spring_function(self):
        before_train_location = self.train_location.copy()
        before_train_location[-1] = 0
        after_train_location = self.train_location.copy()
        after_train_location[0] = 0
        spring_force = np.roll(self.K,1)*(np.roll(before_train_location,1)-after_train_location) \
                    - self.K*(before_train_location-np.roll(after_train_location,-1))
        return spring_force

    def refer_run_force(self):
        refer_force = self.spring_function() - self.davis_function()
        return refer_force

    
    def Update(self,U,Bf,t):
        F = self.refer_run_force() + self.general_resistance.update(t)
        self.train_acceleration = Bf*U + F

        self.train_velocity = self.train_velocity + self.train_acceleration*self.t_sampling
        self.train_location = self.train_location + self.train_velocity*self.t_sampling
        
        return self.train_location,self.train_velocity,self.train_acceleration
    
    def Sync(self,train_location,train_velocity):
        self.train_location = train_location
        self.train_velocity = train_velocity

class TV_Train_System(Train_System):
    def __init__(self):
        super(TV_Train_System,self).__init__()

    def TV_Update(self,t):
        self.Davis_coef = 0
        self.K = 0
        self.M = 0
