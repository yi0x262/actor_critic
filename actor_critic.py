#!/usr/env/bin python2
from actor import actor
from critic import critic

class actor_critic(actor,critic,object):
    def __init__(self,inout,alpha=0.05,beta=0.90,gamma=0.95):
        """
        inout : (in,out)
        alpha : learning gain
        beta  : discount rate (for eligibility)
        gamma : discount rate (for reward)
        """
        super(actor_critic,self).__init__(inout,alpha,beta)
        super(actor,self).__init__(inout[0],alpha,gamma)

    def action(self,state,reward,dt):
        #print(state,reward,dt)
        #print('ac\n',state,reward)
        TDerr = self.TDerror(state,reward,dt)
        self.update(dt,TDerr)
        #print('actor_critic',dt,TDerr)

        return super(actor_critic,self).__call__(state),TDerr

if __name__ == '__main__':
    import numpy as np
    i,o = 3,2
    s = np.ones((1,i))

    ac = actor_critic((i,o))
    print ac.action(s,1,0.1)
