import numpy as np
class npcounter(object):
    def __init__(self,num):
        """
        num : num of counter
        """
        self.counter = np.zeros(num)
    def __call__(self,limits,dt):
        """
        limits  : time limits (list or numpy ndim=1)
        dt      : add time (float scalar or numpy ndim=1)
        """
        print('counter',self.counter)
        self.counter += dt               #countup
        ret = limits < self.counter      #cmp
        self.counter -= ret*self.counter #count reset
        return ret

if __name__ == '__main__':
    c = npcounter(3)
    d = [0,0,0]
    timelist = [1,2,3]
    dt = 0.1
    for t in range(100):
        d = [a+b for a,b in zip(c(timelist,dt),d)]
        print('{0:3.2f}'.format(dt*t),c.counter,d)
