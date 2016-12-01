#!/usr/env/bin python3

import numpy as np
#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)
#!!basic version (no eligibility traces)!!
class actor(object):
    def __init__(self,inout,alpha=0.05):
        """
        inout = (in,out)
        alpha = learning gain (0 < alpha)
        """
        self.alpha = alpha

        Mi  = np.empty((1,inout[0]))#input
        Mo  = np.empty((1,inout[1]))#output

        self.W_exp      = np.zeros(inout)       #
        self.W_var      = np.ones_like(Mo)      #

        self.lastState  = np.zeros_like(Mi)     #
        self.lastAct  = np.zeros_like(Mo)       #

    def action(self,state):
        """
        state : np.array(ndim=1) or list_like
        """
        s = np.array([state])
        mu    = self.mu(s)
        sigma = self.sigma()
        self.lastAct   = np.random.normal(mu,sigma)
        self.lastState = s
        return self.lastAct
    def update(self,TDerror,dt):
        """
        dW_exp/dt = alpha*TDerr*state.(Act-Mu)
        dW_var/dt = alpha*TDerr*((Act-Mu)^2 - Sigma^2)
        """
        mu = self.mu(self.lastState)
        sigma = self.sigma()

        a = dt*self.alpha*TDerror
        self.W_exp += a*np.dot(self.lastState.T,self.lastAct-mu)
        self.W_var += a*(np.power(self.lastAct-mu,2)-np.power(sigma,2))
    def mu(self,state):
        return np.dot(state,self.W_exp)
    def sigma(self):
        try:
            return np.exp(self.W_var)/(1+np.exp(self.W_var))
        except RuntimeWarning:#avoid nan
            return 1/(1+np.exp(-self.W_var))

if __name__ == '__main__':
    inout = (2,3)
    a = actor(inout)
    print(a.action(np.zeros(inout[0])))
    print(a.update(1,0.1))
    print(a.action(np.zeros(inout[0])))
