from matplotlib import pyplot as plt
import numpy as np


def A(num_carriage,t_s):
    length = int(2000/t_s)
    A = np.zeros((num_carriage,length))
    for i in range(length):
        t = i*t_s
        if t >1200:
            a = 0.075 - 0.00009375*(t-1200)

        elif t <800:
            a = 0.00009375*t

        else:
            a = 0.075
        A[:,i] = np.ones(num_carriage)*a
    return A

def V(num_carriage,A_list,t_s):

    v = 0
    length = int(2000/t_s)
    V = np.zeros((num_carriage,length))

    for i in range(length):

        v = v + A_list[:,i]*t_s
        V[:,i] = v
    return V

def X(num_carriage,V_list,t_s):
    x = 0
    length = int(2000/t_s)
    X = np.zeros((num_carriage,length))

    for i in range(length):

        x = x + V_list[:,i]*t_s
        X[:,i] = x
    return X