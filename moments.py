import qutip as q
import numpy as np
from scipy import integrate as ig
import matplotlib.pyplot as plt
import decompose as dcp
import scipy.optimize as opt

warnings=True
N=10
H = q.num(N,1)


def max_coherent(k):
    if (k==0 or k>N):
        raise ValueError('Parameter k must satisfy 1 <= k <= N')
    state = q.basis(N,0)
    for i in range(1,k):
        state += q.basis(N,i)
    return state * 1./np.sqrt(k)


def state(*coeff):
    if (np.sum(coeff) != 1 or len(coeff)>N) and warnings:
        raise ValueError('There must be at most N probabilities that sum to 1')
    state = np.sqrt(coeff[0]) * q.basis(N,0)
    for i in range(len(coeff)-1):
        state += np.sqrt(coeff[i+1])*q.basis(N,i+1)
    return state
        
def two_coherent(c1):
    if c1 > 1. or c1 < 0.:
        raise ValueError('c1 must be 0<= c1 <=1')
    state = np.sqrt(c1) * q.basis(N,0)
    state += np.sqrt(1-c1) * q.basis(N,1)
    return state

def three_coherent(c1,c2):
    if c1 > 1. or c1 < 0. or c2 > 1-c1 or c2 < 0.:
        raise ValueError('c1 and c2 must be 0<= c1, c2 <=1')
    state = np.sqrt(c1) * q.basis(N,0)
    state += np.sqrt(c2) * q.basis(N,1)
    state += np.sqrt(1-c1-c2) * q.basis(N,2)
    return state

# Generate coherent measurement state.
m_state = max_coherent(N)

# Generate ALMOST coherent measurement state.

#eps = 1
#
#m_state = max_coherent(N)
#m_state -= q.basis(N,0)*eps
#m_state += q.basis(N,1)*(np.sqrt(1 - m_state[0]**2 - (N-2)/N) - 1/np.sqrt(N))

# Generate random measurement state.
#m_state = q.rand_ket(N)

# Generate pure state density matrix.
#system = q.rand_dm_ginibre(N, rank=1)

# Generate random density matrix.
#system = q.rand_dm(N, 1)

# Generate maximally coherent density matrix.
system = q.ket2dm(max_coherent(N))

#Testing for 2-coherent mixed states.
#state1 = (1/np.sqrt(2))*(q.basis(3,1) + q.basis(3,0))
#state2 = (1/np.sqrt(2))*(q.basis(3,2) + q.basis(3,0))
#system1 = q.ket2dm(state1)
#system2 = q.ket2dm(state2)
#system = 0.5*system1 + 0.5*system24





# Basic functions
def prob_dist(t):
    m_state_t = q.sesolve(-H, m_state, [0.,t]).states[1]
    expec = q.expect(system, m_state_t)
    #print(m_state_t, '\n\n', expec)  #Normalisation with np.sqrt(N)?
    return expec

def integrand(t,n):
    return np.power(prob_dist(t), n)

def moment(n):
    M = ig.quad(integrand, 0., 2*np.pi, (n), full_output=0)
    #print(M)
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


def plot_pattern():
    b = []
    t_list =np.linspace(0., 2*np.pi, 200)
    for t in t_list:
        b.append(prob_dist(t))        
    plt.plot(t_list, b)
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 1)
    #return np.array(b)

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


def mom_func3(a,b,c):
    m1 = moment(1)
    m2 = moment(2)
    m3 = moment(3)
    #print(m1,m2,m3)
    #print(8*c/243.)
    return a*m1*m1*m1 +b*m2*m1 + c*m3

def convex_test(func, *args):
    p = np.random.rand()
    
    rho1 = q.rand_dm(N, 1)
    rho2 = q.rand_dm(N, 1)
   
    combined  = p*rho1 + (1.-p)*rho2
    
    global m_state
    m_state = q.rand_ket(N)
    global system
    system = rho1
    a = p*func(*args)
    system = rho2
    b = (1-p)*func(*args)
    system = combined
    c = func(*args)
    
    if (a+b < c):
        print("FAILED")
        print(p)
        print(a,b)
        print(a+b)
        print(c)


def plot_max_func(steps=10,moment_no=3):
    global system
    global m_state
    s_coeff = np.linspace(1e-9,1,steps)
    m_coeff = np.linspace(1e-9,1,steps)
    func_vals = np.empty((steps,steps))
    for i in range(len(s_coeff)):
        for j in range(len(m_coeff)):
            system = q.ket2dm(two_coherent(s_coeff[i]))
            m_state = two_coherent(m_coeff[j])
            #func_vals[i,j] = moment(moment_no)
            #func_vals[i,j] = mom_func3(2,3,1)
            func_vals[i,j]= moment(moment_no)/moment(1)**(moment_no-1)
    s_c, m_c = np.meshgrid(s_coeff, m_coeff)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(s_c,m_c,func_vals)
    plt.show()


def func(args):
    global warnings
    warnings=False
    global system
    global m_state
    system = q.ket2dm(state(*args[:int(len(args)/2)]))
    m_state = state(*args[int(len(args)/2):])
    #print(m_state)
    return -(moment(3)/(moment(1)**2))

def norm_con1(args):
    return np.sum(args[:int(len(args)/2)]) - 1.

def norm_con2(args):
    return np.sum(args[int(len(args)/2):]) - 1.

def find_max_func(N_, guess=[], e=0):
    if guess == []:
        guess = (1./N_)*np.ones(2*N_)
    #print(guess)
    result = opt.minimize(func, guess, 
                        bounds=[(e,1.-e) for i in range(2*N_)],
                        constraints = [{'type':'eq', 'fun': norm_con1},
                                       {'type':'eq', 'fun': norm_con2}])
    return result #(result.x[:int(len(guess)/2)], result.x[int(len(guess)/2):])

def find_prob_pattern(n, plot_probs=False): #legends don't work
    res = find_max_func(n)
    syst = np.array(res.x[:n])
    meas = np.array(res.x[n:])
    val = -func(res.x)
    
    if plot_probs:
        x = np.arange(1,n+1)
        xx = np.linspace(1,n,200)
        f = lambda arg, aa, bb, cc: aa*arg**2 + bb*arg + cc
        
        sopt, scov = opt.curve_fit(f, x, syst)
        mopt, mcov = opt.curve_fit(f, x, meas)
        
        plt.plot(x, syst, label='Probabilities')
        plt.plot(xx, f(xx, *sopt), label='Curve fit')
        plt.title('System')
        plt.legend()
        plt.figure()
        plt.plot(x, meas, label='Probabilities')
        plt.plot(xx, f(xx, *mopt), label='Curve fit')
        plt.title('Measurement')
        plt.legend()
        
        return val, syst, meas, (sopt, scov, mopt, mcov)
    else:
        return val, syst, meas

def plot_max_pattern(n):
    vals, syst, meas = [], [], []
    for i in range(1,n+1):
        print(i)
        v,s,m = find_prob_pattern(i)
        vals.append(v); syst.append(s); meas.append(m)
    x = np.arange(1,n+1)
    plt.plot(x, vals, label='Maxima')
    x = np.linspace(1,n,200)
    plt.plot(x, 11./20 * x + 3/20, label='Curve fit')
    plt.legend()
    
    return vals, syst, meas

#a = np.array([moment(1), moment(2), moment(3),moment(4)])
#system = q.ket2dm(max_coherent(N-1))
#b = np.array([moment(1), moment(2), moment(3),moment(4)])
#system = q.ket2dm(max_coherent(N-2))
#c = np.array([moment(1), moment(2), moment(3),moment(4)])

    
    
#nt = 1
#tmp = moment(nt)
#plot_pattern()
#system = q.ket2dm(max_coherent(N-1))
#curr = moment(nt)
#plot_pattern()
#print(m_state)
#print(system)
#print()
#print()
#print("N:", tmp)
#print("N-:", curr)
#print("diff:", tmp - curr)

#plot_pattern()
# Testing hierarchical method        
#a,b,c = 1,10,10
#system = q.ket2dm(max_coherent(3))
#k3=mom_func(3,a,b,c)
#system = q.ket2dm(max_coherent(2))
#k2=mom_func(3,a,b,c)
#print((k3-k2)/k2)

#system = q.rand_dm(N, 1)
#b1 = plot_pattern()
#print(system)
#m1 = []
#for i in range(1,10):
#    m1.append(moment(i))
#system = q.rand_dm(N, 1)
#b2 = plot_pattern()
#print(system)
#m2 = []
#for i in range(1,10):
#    m2.append(moment(i))
#(m, n) = (m1, m2) if m1[-1] > m2[-1] else (m2, m1)
#for i in range(len(m)):
#    if m[i]<n[i]: print('FAIL')
#    print('{0:.4f}, {1:.4f}'.format(m[i], n[i]))

#print(system)
#print()
#print(system.eigenenergies())
#print()
#print(system.eigenstates())
#
#

#for i in range(100):
#    print(i)
#    convex_test(mom_func3, -1,-1,2)