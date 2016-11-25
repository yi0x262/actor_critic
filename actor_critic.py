#!/usr/env/bin python3
from actor import actor
from critic import critic

class actor_critic(object):
    def __init__(self,inout,alpha=0.05,gamma=0.95):
        """
        inout : (in,out)
        alpha : learning gain
        gamma : discount rate
        """
        self.a = actor(inout,alpha=alpha)
        self.c = critic(inout[0],alpha=alpha,gamma=gamma)

    def action(self,state,reward,dt):
        #print(state,reward,dt)
        TDerr = self.c.TDerror(state,reward,dt)
        self.a.update(TDerr,dt)

        return self.a.action(state),TDerr

if __name__ == '__main__':
    pass
