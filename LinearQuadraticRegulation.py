#!/usr/env/bin python3
import numpy as np
from actor import optional_matrix

from collections import deque
class LQR(optional_matrix):
    act = deque(maxlen=100)
    def update(self,a,dt):
        self += a*dt
        #return np.clip(self,self.range[0],self.range[1])
        return max(self.range[0],min(self.range[1],self))
    def reward(self):
        return 0.1-self**2# + 0.1*abs(self)<0.2

if __name__ == '__main__':
    shape = (1,1)

    from actor_critic import actor_critic
    ac = actor_critic(shape,alpha=0.2,beta=0.75)
    lqr= LQR(shape,range=(0,4),w0=1.)
    print(lqr,lqr.range)

    dt = 0.1
    t = list(np.arange(0,5000,dt))

    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(__file__))+ '/central_pattern_generator')
    from save_plot import logger
    lgr = logger(['act','state','reward','actor_W_exp','W_exp_D','actor_W_var','W_var_D','critic_W','TDerror'])

    from collections import deque
    action_deq = deque(maxlen=100)

    rec_t = []

    for _ in t:
        #print('LQR_main',lqr,lqr.reward())
        try:
            act,TDerr = ac.action(lqr,lqr.reward(),dt)
        except RuntimeWarning:
            break
        rec_t.append(_)
        #print('LQR_MAIN',act,TDerr)
        lqr.update(act,dt)
        action_deq.append(act)
        lgr.append([sum(action_deq)[0]/len(action_deq),lqr[0],lqr.reward()[0],ac.W_exp[0],ac.W_exp.D[0],ac.W_var[0],ac.W_var.D[0],ac.W_crt[0],TDerr[0]])

    lgr.output('/home/yihome/Pictures/log/LQR_actor_critic',rec_t)
