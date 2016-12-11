#!/usr/env/bin python3
import numpy as np
#An Analysis of Actor-Critic Algorithms using Eligibility Traces:Reinforcement Learning with Imperfect Value Functions (kimura,1996?)

class counter(np.ndarray):
    def __new__(self,shape):#,*args,**keys):
        #print('counter',self,shape)#,args,keys)
        return 0.*super().__new__(self,shape)#,*args,**keys)
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

class optional_matrix(np.ndarray):
    def __new__(self,shape,**keys):
        #print(shape,keys)
        w0 = keys.pop('w0',0.)
        for key in [k for k in keys if not hasattr(np.ndarray,k)]:
            setattr(self,key,keys.pop(key))
        return w0*super().__new__(self,shape,**keys)

class eligibility_tracer(optional_matrix):
    def __call__(self,dt,e):
        self += ((self.beta-1)*self + e)*dt
        return self

class weight_eligibility(optional_matrix):
    def __new__(self,shape,alpha=0.05,beta=0.90,**keys):
        #print('w_eligibility',keys)
        self.alpha = keys.pop('alpha',alpha)
        self.D = eligibility_tracer(shape,beta=beta)
        return super().__new__(self,shape,**keys)
    def update(self,dt,TDerror,*args):
        #print('weight_eligibility',args)
        e = self.eligibility(*args)
        self += self.alpha*TDerror*self.D(dt,e)

class actor(object):
    def __init__(self,inout,alpha=0.05,beta=0.90):
        self.alpha = alpha
        self.W_exp = type('weight_exp',(weight_eligibility,),
                    {
                        '__call__'    : lambda self,x:x.dot(self),
                        'eligibility' : lambda self,a,x,mu:x.T.dot(a-mu)
                    })(inout)
        def weight_var_calc(self):
            try:
                return 1./(1.+np.exp(-self))
            except RuntimeWarning:
                return np.exp(self)/(1.+np.exp(self))
        self.W_var = type('weight_var',(weight_eligibility,),
                    {
                        '__call__'    : weight_var_calc,
                        'eligibility' : lambda self,a,mu,sigma:((a-mu)**2-sigma**2)*(1-sigma)
                    })((1,inout[1]))
        self.timer = counter((1,inout[1]))
    def __call__(self,state):
        self.lastState = state
        self.mu = self.W_exp(state)
        self.sigma = self.W_var()
        #print('actor.__call__\nmu:\n{}\nsigma:\n{}'.format(self.mu,self.sigma))
        self.lastAct = np.random.normal(self.mu,self.sigma)
        return self.lastAct
    def update(self,t,TDerr):
        dt = self.timer(t,self.sigma)*self.sigma
        #print('actor.update\na:\n{}\nx:\n{}\nmu\n{}\nsigma\n{}'.format(self.lastAct,self.lastState,self.mu,self.sigma))
        self.W_exp.update(dt,TDerr,self.lastAct,self.lastState,self.mu)
        self.W_var.update(dt,TDerr,self.lastAct,self.mu,self.sigma)

if __name__ == '__main__':
    i,o = 2,3
    s = np.ones((1,i))
    a = actor((i,o))
    print('main\n',a(s))
    print('actor.W_exp\n',a.W_exp)
    print('actor.W_var\n',a.W_var)
    a.update(1,1)
    print('actor.W_exp\n',a.W_exp)
    print('actor.W_var\n',a.W_var)
