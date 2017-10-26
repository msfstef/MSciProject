import qutip as q
import numpy as np
from scipy import integrate as ig
import matplotlib.pyplot as plt
N=2

# Generate coherent measurement state.
m_state = q.basis(N,0) * 1./(np.sqrt(N))
for i in range(1,N):
    m_state += q.basis(N,i) * 1./(np.sqrt(N))

# Generate random measurement state.
#m_state = q.rand_ket(N)

# Generate pure state density matrix.
system = q.rand_dm_ginibre(N, rank=1)

# Generate random density matrix.
#system = q.rand_dm(N, 0.75)

H = q.charge(N,1)
print(system)

a=np.diag(np.sqrt(system.diag())*m_state)
#print(np.abs(np.vdot(a,a)))

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
    return M[0]


def mom_func3(c=1):
    m1 = moment(1)
    m2 = moment(2)
    m3 = moment(3)
    print(m1,m2,m3)
    return 2*c*m1*m1*m1 -3*c*m2*m1 + c*m3
    


#print(moment(2))
print(mom_func3())    


b = []
t_list =np.linspace(0.,2*np.pi,200)
for t in t_list:
    b.append(prob_dist(t))

plt.plot(t_list,b)
plt.xlim(0,2*np.pi)
plt.ylim(0,1)



    