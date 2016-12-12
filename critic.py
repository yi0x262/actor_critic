#!/usr/env/bin python3
import numpy as np

#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)
#!!basic version (no eligibility traces)!!

class critic(object):
    def __init__(self,state_num,alpha,gamma):
        self.alpha = alpha
        self.gamma = gamma

        W1 = np.ones((1,state_num))
        self.W_crt = np.random.normal(0*W1,0.5*W1)*W1
        self.lastState = 0*W1
    def TDerror(self,state,reward,dt):
        """
        TDerr = r_t + g*V(s_t+1) - V(s_t)
        """
        try:
            TDerr = reward + self.gamma*self.Value(state) - self.Value(self.lastState)
        except RuntimeWarning:
            print('critic.TDerror',self,state,self.lastState)
            raise RuntimeWarning
        #print('TDerr',self.W,dt,self.alpha,TDerr,state)
        self.W_crt += dt*self.alpha*TDerr*state
        self.W_crt = np.nan_to_num(self.W_crt)

        self.lastState = state
        return np.nan_to_num(TDerr)
    def Value(self,state):
        """
        V(s) = s.W^T / len(s)

        state -> expected reward
        R^1*s -> R
        """
        #print('c_value\n',state,self.W_crt)
        return np.dot(state,self.W_crt.T)/state.shape[0]
