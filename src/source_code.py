
import pandas as pd
import numpy as np

def F(w):
    """
    Define test function: 
    3 * w[0] ** 2 + 4 * w[1] ** 2 - 5 * w[0] + 7
    """
    return(
        3*w[0] ** 2 + 4*w[1] ** 2 - 5 * w[0] + 7
    )

def grad(w):
    
    g = [0] * 2
    
    g[0] = 6*w[0] - 5
    g[1] = 8 * w[1]
    
    return g

def descent(w_new,w_prev,lr):
    
    print(w_prev)
    print(F(w_prev))
    
    while True:
        w_prev = w_new
        w_0 = w_prev[0] - lr*grad(w_prev)[0]
        w_1 = w_prev[1] - lr*grad(w_prev)[1]
        
        w_new = [w_0, w_1] # simultaneous update
        
        print(w_new)
        print(F(w_new))
        
        if (w_new[0] - w_prev[0]) ** 2 + (w_new[1] - w_prev[1]) ** 2 < pow(10,-6): # power of a number
            
            break