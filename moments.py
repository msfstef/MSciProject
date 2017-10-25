import qutip as q
import numpy as np
from scipy import integrate as ig
import matplotlib.pyplot as plt
N=20

m_state = q.rand_ket(N)
system = q.rand_dm(N, 0.75)
H = q.charge(N,1)


a=np.diag(np.sqrt(system.diag())*m_state)
print(np.abs(np.vdot(a,a)))

def prob_dist(t):
    m_state_t = q.sesolve(H, m_state,[0.,t]).states[1]
    #print(t/(np.pi))
    #print(m_state)
    #print(m_state_t)
    #print("~~~~")
    #print(system)
    #print(q.expect(system,m_state_t))
    return q.expect(system,m_state_t)

def integrand(t,n):
    return np.power(prob_dist(t),n)


def moment(n):
    M = ig.quad(integrand,0.,2*np.pi,(n))
    return M



b = []
t_list =np.linspace(0.,2*np.pi,100)
for t in t_list:
    b.append(prob_dist(t))

plt.plot(t_list,b)
    