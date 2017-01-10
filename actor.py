#!/usr/env/bin python2
import numpy as np
#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)

class counter(np.ndarray,object):
    def __new__(self,shape):#,*args,**keys):
        return 0.*super(counter,self).__new__(self,shape)#,*args,**keys)
    def __call__(self,dt,threshold):
        """
        change continuous time into by threshold
        0 :no time passed
        ~0:update
        """
        self += dt
        judge = self > threshold
        self *= ~judge
        return judge*threshold

class optional_matrix(np.ndarray,object):
    def __new__(self,shape,**keys):
        w0 = keys.pop('w0',0.)
        for key in [k for k in keys if not hasattr(np.ndarray,k)]:
            setattr(self,key,keys.pop(key))
        return 0.*super(optional_matrix,self).__new__(self,shape,**keys)+w0

class eligibility_tracer(optional_matrix,object):
    def __call__(self,dt,e):
        self += ((self.beta-1)*self + e)*dt
        self = np.nan_to_num(self)
        return self

import abc
class weight_eligibility(optional_matrix,object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,shape,alpha=0.05,beta=0.90,**keys):
        super(weight_eligibility,self).__init__(self,shape,**keys)
        self.alpha = keys.pop('alpha',alpha)
        self.D = eligibility_tracer(shape,beta=beta)
    def update(self,dt,TDerror,*args):
        #print('weight_eligibility',args)
        e = self.eligibility(*args)
        self += self.alpha*TDerror*self.D(dt,e)*dt
        self = np.nan_to_num(self)
        #print('weight',e,self.D,dt)
    @abc.abstractmethod
    def eligibility(self):
        """calc e in update()"""
        pass

class weight_exp(weight_eligibility,object):
    def __call__(self,x):
        return np.array(x.dot(self))
    def eligibility(self,a,x,mu):
        return x.T.dot(a-mu)
class weight_var(weight_eligibility,object):
    def __call__(self):
        try:
            return np.array(1./(1.+np.exp(-self)))
        except RuntimeWarning,e:
            return np.array(np.exp(self)/(1.+np.exp(self)))
    def eligibility(self,a,mu,sigma):
        return ((a-mu)**2-sigma**2)*(1-sigma)

class actor(object):
    def __init__(self,inout,alpha=0.05,beta=0.90):
        print inout
        self.W_exp = weight_exp(inout,alpha=alpha)
        self.W_var = weight_var((1,inout[1]),w0=0.55,alpha=alpha)
        self.timer = counter((1,inout[1]))
    def __call__(self,state):
        self.lastState = state
        self.mu = self.W_exp(state)
        self.sigma = self.W_var()
        #print('actor.__call__\nmu:\n{}\nsigma:\n{}'.format(self.mu,self.sigma))
        try:
        #print 'mu:\n',self.mu
        #print 'sigma:\n',self.sigma
            self.lastAct = np.random.normal(self.mu,self.sigma)#*np.ones_like(self.sigma)
        #print 'action:\n',self.lastAct
        except ValueError,e:
            print e
            raise RuntimeWarning('actor:lastAct\nmu:\n{}\nsigma:\n{}'.format(self.mu,self.sigma))
        #np.ones_like save answer type ndarray if shape is (1,1)
        #(if not,it goes a float)
        return self.lastAct
    def update(self,t,TDerr):
        try:
            dt = self.timer(t,self.sigma)*self.sigma
        except AttributeError:
            return
        #print('actor.update\na:\n{}\nx:\n{}\nmu\n{}\nsigma\n{}'.format(self.lastAct,self.lastState,self.mu,self.sigma))
        self.W_exp.update(dt,TDerr,self.lastAct,self.lastState,self.mu)
        self.W_var.update(dt,TDerr,self.lastAct,self.mu,self.sigma)

if __name__ == '__main__':
    i,o = 2,3
    s = np.ones((1,i))
    a = actor((i,o))
    print 'main\n',a(s)
    print 'actor.W_exp\n',a.W_exp
    print 'actor.W_var\n',a.W_var
    a.update(1,1)
    print 'actor.W_exp\n',a.W_exp
    print 'actor.W_var\n',a.W_var
