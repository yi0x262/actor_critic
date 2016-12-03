#!/usr/env/bin python3
import numpy as np
from npcounter import npcounter
#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)
#!!basic version (no eligibility traces)!!
class actor(object):
    def __init__(self,inout,alpha=0.05,beta=0.9):
        """
        inout = (in,out)
        alpha = learning gain (0 < alpha)
        beta  = discount rate (for eligibility)
        """
        self.alpha = alpha
        self.beta = beta

        i,o = inout

        self.W_exp      = np.zeros(inout)           #expectation
        self.W_var      = 0.5*np.ones((1,o))        #variance

        self.D_exp      = np.zeros_like(self.W_exp) #eligibility tracer
        self.D_var      = np.zeros_like(self.W_var)

        self.lastState  = np.zeros((1,i))           #for update
        self.lastAct    = np.zeros((1,o))

        self.update_counter = npcounter(o)          #

    def action(self,state):
        """
        state : np.array(ndim=1) or list_like
        """
        self.lastState = np.array([state])
        mu    = self.mu(self.lastState)
        sigma = self.sigma()
        print('mu',mu,'sigma',sigma)
        self.lastAct   = np.random.normal(mu,sigma)
        return self.lastAct

    def update(self,TDerror,dt):
        """
        dW_exp/dt = alpha*TDerr*state.(Act-Mu)
        dW_var/dt = alpha*TDerr*((Act-Mu)^2 - Sigma^2)

        at:act,st:state,mu:mu,sg:sigma
        e_exp = (at - mu).st
        e_var = ((at - mu)**2 - sg**2)(1-sg)

        """
        mu = self.mu(self.lastState)
        sigma = self.sigma()
        update = self.update_counter(sigma[0],dt)

        e_exp = self.lastState.T.dot(self.lastAct - mu)
        e_var = ((self.lastAct - mu)**2 - sigma**2)*(1-sigma)

        self.D_exp += update*((1-self.beta)*self.D_exp + e_exp)
        self.D_var += update*((1-self.beta)*self.D_var + e_var)

        self.W_exp += update*self.alpha*TDerror*self.D_exp
        self.W_var += update*self.alpha*TDerror*self.D_var

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
