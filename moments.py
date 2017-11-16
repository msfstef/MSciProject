import qutip as q
import numpy as np
from scipy import integrate as ig
import matplotlib.pyplot as plt
import decompose as dcp

N=3


def max_coherent(k):
    state = q.basis(N,0) * 1./(np.sqrt(N))
    for i in range(1,k):
        state += q.basis(N,i) * 1./(np.sqrt(N))
    return state



# Generate coherent measurement state.
m_state = max_coherent(N)

# Generate random measurement state.
#m_state = q.rand_ket(N)

# Generate pure state density matrix.
system = q.rand_dm_ginibre(N, rank=1)

# Generate random density matrix.
#system = q.rand_dm(N, 1)

# Generate maximally coherent density matrix.
#system = q.ket2dm(max_coherent(N))


#Testing for 2-coherent mixed states.
#state1 = (1/np.sqrt(2))*(q.basis(3,1) + q.basis(3,0))
#state2 = (1/np.sqrt(2))*(q.basis(3,2) + q.basis(3,0))
#system1 = q.ket2dm(state1)
#system2 = q.ket2dm(state2)
#system = 0.5*system1 + 0.5*system2


H = q.charge(N,1)
print(system)


def prob_dist(t):
    m_state_t = q.sesolve(H, m_state,[0.,t]).states[1] * np.sqrt(N)
    #print(m_state_t)
    return q.expect(system,m_state_t)

def integrand(t,n):
    return np.power(prob_dist(t),n)


def moment(n):
    M = ig.quad(integrand,0.,2*np.pi,(n))
    return (1/(2*np.pi))*M[0]

def mom_product(*args):
    """
    *args is the moment numbers for
    the desired product.
    
    mom_product(2,3,1) =
    moment(2)*moment(3)*moment(1)
    """
    result = 1.
    for i in range(len(args)):
        result *= moment(args[i])
    return result


def mom_func3_OLD(c=100):
    m1 = moment(1)
    m2 = moment(2)
    m3 = moment(3)
    #print(8*c/243.)
    return 2*c*m1*m1*m1 -3*c*m2*m1 + c*m3

def mom_func3(a,b,c):
    m1 = moment(1)
    m2 = moment(2)
    m3 = moment(3)
    print(m1,m2,m3)
    #print(8*c/243.)
    return a*m1*m1*m1 +b*m2*m1 + c*m3

def mom_func(n, *args):
    """
    General F_n function.
    Note that moments are recalculated for each term,
    can make more efficient.
    """
    combinations = dcp.decomposeInt(n)
    assert (len(args) == len(combinations)), (
    "%r coefficients must be specified." % len(combinations))
    terms = []
    for comb in combinations:
        terms.append(mom_product(*comb))
    terms = np.array(terms)
    coeff = np.array(args)
    result = np.sum(coeff*terms)
    return result



def plot_pattern():
    b = []
    t_list =np.linspace(0.,2*np.pi,200)
    for t in t_list:
        b.append(prob_dist(t))        
    plt.plot(t_list,b)
    plt.xlim(0,2*np.pi)
    plt.ylim(0,N)


def convex_test(func, *args):
    p = np.random.rand()
    rho1 = q.rand_dm(N, 1)
    rho2 = q.rand_dm(N, 1)
    combined  = p*rho1 + (1.-p)*rho2
    
    global system
    system = rho1
    a = p*func(*args)
    system = rho2
    b = (1-p)*func(*args)
    system = combined
    c = func(*args)
    
    if (a+b < c):
        print("FAILED")
        print(a+b)
        print(a,b)
        print(c)


#plot_pattern()


#for i in range(100):
#    print(i)
#    convex_test(mom_func3, -1,-1,2)