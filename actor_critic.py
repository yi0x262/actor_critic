#!/usr/env/bin python3
from actor import actor
from critic import critic

class actor_critic(actor,critic):
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
        self.update(TDerr,dt)

        return super(actor_critic,self).action(state),TDerr

if __name__ == '__main__':
    inout = (2,2)
    ac = actor_critic(inout)
