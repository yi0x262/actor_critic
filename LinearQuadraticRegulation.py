#!/usr/env/bin python3
import numpy as np
class LQR:
    def __init__(self,x0,x_range=(-4,4)):
        self.x = np.ones((1,1))*x0
        self.x_range = x_range
    def update(self,a,dt):
        self.x += np.ones((1,1))*a*dt
        #limit range
        self.x = np.clip(self.x,-4,4)#self.x_range[0],self.x_range[1])
        return self.x
    def state(self):
        return self.x
    def reward(self):
        return 1-np.power(self.x,2)

if __name__ == '__main__':
    from actor_critic import actor_critic
    ac = actor_critic((1,1))
    lqr= LQR(1)

    dt = 0.01
    t = list(np.arange(0,10000,dt))

    from collections import deque
    dq = deque(maxlen=100)

    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(__file__))+ '/central_pattern_generator')
    from save_plot import logger
    lgr = logger(['act','state','reward','avr_deq'+str(dq.maxlen),'actor_W_ver','TDerror'])

    for _ in t:
        act,TDerr = ac.action(lqr.state(),lqr.reward(),dt)
        lqr.update(act,dt)
        dq.append(lqr.reward())
        #if sum(dq)/len(dq) > 1-1e-4:
        #    lqr = LQR(1)
        #print(lqr.state(),lqr.reward(),sum(dq),_)
        lgr.append([act,lqr.state()[0],lqr.reward()[0],sum(dq)[0]/len(dq),ac.a.W_var[0],TDerr[0]])

    lgr.output('/home/yihome/Pictures/log/LQR_actor_critic',t)
