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

        Mi  = np.ones((1,inout[0]))
        Mo  = np.ones((1,inout[1]))
        Mio = np.ones(inout)

        self.W_exp      = 0*Mio     #
        self.W_var      = -10*Mo      #

        self.lastState  = 0*Mi      #
        self.lastAct    = 0*Mo      #

    def action(self,state):
        mu    = self.mu(state)
        sigma = self.sigma()
        print(sigma)
        self.lastAct   = np.random.normal(mu,sigma)
        self.lastState = state
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
            return np.exp(-self.W_var)/(1+np.exp(-self.W_var))
        except RuntimeWarning:#avoid nan
            return 1/(1+np.exp(self.W_var))
