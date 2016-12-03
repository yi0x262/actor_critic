#!/usr/env/bin python3
import numpy as np

#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)
#!!basic version (no eligibility traces)!!
class critic(object):
    def __init__(self,state_num,alpha=0.05,gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma

        W1 = np.ones((1,state_num))
        self.W          = np.random.normal(0*W1,0.5*W1)
        self.lastState  = 0*W1[0]
        print('c_init',self.W,self.lastState)
    def TDerror(self,state,reward,dt):
        """
        TDerr = r_t + g*V(s_t+1) - V(s_t)
        """
        TDerr = reward + self.gamma*self.Value(state) - self.Value(self.lastState)
        #print('TDerr',self.W,dt,self.alpha,TDerr,state)
        self.W += dt*self.alpha*TDerr*state

        self.lastState = state
        return TDerr
    def Value(self,state):
        """
        V(s) = s.W^T / |\{s\}|

        state -> expected reward
        R^1*s -> R
        """
        #print('c_value\n',state,self.W)
        return np.dot(state,self.W.T)/state.shape[0]
