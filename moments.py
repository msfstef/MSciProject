import qutip as q
import numpy as np
from scipy import integrate as ig
import matplotlib.pyplot as plt
import decompose as dcp
import scipy.optimize as opt
import scipy.linalg as lin
import scipy.misc as misc

plt.rc('axes', labelsize=14)

warnings=False
N=2
H = q.num(N,1)
#nss=0.5
#H = q.Qobj(np.diag([0,2,4]))


def max_coherent(k):
    if (k==0 or k>N):
        raise ValueError('Parameter k must satisfy 1 <= k <= N')
    state = q.basis(N,0)
    for i in range(1,k):
        state += q.basis(N,i)
    return state * 1./np.sqrt(k)

def mixed_state(coeff, s):
    if (s<0 or s>1):
       raise ValueError('Parameter s must satisfy 0 <= k <= 1')
    identity = q.Qobj(np.diag(np.concatenate(
            [np.ones(len(coeff)),np.zeros(N-len(coeff))])))
    sys = (1-s)*q.ket2dm(state(*coeff)) + s/len(coeff) * identity
    return sys
     
def state(*coeff):
    if (np.sum(coeff) != 1 or len(coeff)>N) and warnings:
        print(np.sum(coeff), len(coeff))
        print(np.sum(coeff) != 1, len(coeff) > 0)
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

def EpsState(coeff, eps):
     c = coeff.copy()
     c.append(eps*eps)
     for i in range(len(c)):
          c[i] /= (1+eps*eps)
     return c
     

def ratio_anal(k):
    return ((1/k)+(6/k**3)*((1/6.)*k*(k-1)*(2*k-1)) + 
            (2/k**4)*((1/40)*k*(k-2)*(k-1)*(2-7*k+11*k**2)))

# Generate coherent measurement state.
m_state = max_coherent(N)

# Generate ALMOST coherent measurement state.

#eps = 1
#
#m_state = max_coherent(N)
#m_state -= q.basis(N,0)*eps
#m_state += q.basis(N,1)*(np.sqrt(1 - m_state[0]**2 - (N-2)/N) - 1/np.sqrt(N))

# Generate random measurement state.
m_state = q.rand_ket(N)

# Generate pure state density matrix.
#system = q.rand_dm_ginibre(N, rank=1)

# Generate random density matrix.
# system = q.rand_dm(N, 1)

# Generate maximally coherent density matrix.
# system = q.ket2dm(max_coherent(N))
system = q.ket2dm(q.rand_ket(N))
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


def plot_pattern(label='system', ax = 'None'):
    b = []
    t_list =np.linspace(0., 2*np.pi, 200)
    for t in t_list:
        b.append(prob_dist(t))        
    if ax == 'None':
        plt.plot(t_list, b, label=label)
        plt.xlim(0, 2*np.pi)
        plt.ylim(0, 1)
    else:
        ax.plot(t_list, b, label=label)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 1)

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
            print(i,j)
            system = q.ket2dm(two_coherent(s_coeff[i]))
            m_state = two_coherent(m_coeff[j])
            #func_vals[i,j] = moment(moment_no)
            func_vals[i,j] = mom_func(3,2,3,1)
            #func_vals[i,j]= moment(moment_no)/moment(1)**(moment_no-1)
    s_c, m_c = np.meshgrid(s_coeff, m_coeff)
    
    fig=plt.figure()
    
    manyax=False
    if manyax:
        ax = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=2,projection='3d')
        ax2 = plt.subplot2grid((2,3),(1,2),rowspan=1,colspan=1)
        ax3 = plt.subplot2grid((2,3),(0,2),rowspan=1,colspan=1)
    else:
        ax = fig.gca(projection='3d')

    ax.plot_surface(s_c,m_c,func_vals, cmap='Greens_r')
    
    max_coh_vals = np.empty(len(s_coeff))
    not_rob_vals = np.empty(len(s_coeff))
    not_rob_vals2 = np.empty(len(s_coeff))
    for i in range(len(s_coeff)):
        system = q.ket2dm(two_coherent(s_coeff[i]))
        m_state = two_coherent(0.5)
        if manyax:
            max_coh_vals[i] = moment(moment_no)
            m_state = two_coherent(moment_no/float(moment_no+1))
            not_rob_vals[i] = moment(moment_no)
            m_state = two_coherent(1/float(moment_no+1))
            not_rob_vals2[i] = moment(moment_no)
    
    if manyax:
        ax.plot(s_coeff,np.ones(len(s_coeff))*0.5, max_coh_vals, 'b--')
        ax.plot(s_coeff,np.ones(len(s_coeff))*moment_no/float(moment_no+1),
            not_rob_vals, 'r-.')
        ax.plot(s_coeff,np.ones(len(s_coeff))*1/float(moment_no+1),
            not_rob_vals2, 'r-.')
    
    ax.set_ylabel(r'$|\chi_1|^2$')
    ax.set_xlabel(r'$|\psi_1|^2$')
    ax.set_zlabel(r'$M_3$')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_ylim(1,0)
    ax.set_xlim(0,1)
    
    if manyax:
        ax2.plot(s_coeff, max_coh_vals, 'b--')
        ax2.set_xlim(0,1)
        ax2.set_ylim(0,1)
        ax2.set_xlabel(r'$|\psi_1|^2$')
        ax2.set_ylabel(r'$M_3$', rotation='horizontal')


        ax3.plot(s_coeff,not_rob_vals,'r-.')
        ax3.set_xlim(0,1)
        ax3.set_ylim(0,1)
        ax3.set_xlabel(r'$|\psi_1|^2$')
        ax3.set_ylabel(r'$M_3$',rotation='horizontal')
    
        plt.tight_layout()



def error_prob(t, chi, psi, eps):
    k = chi.size
    coses = np.cos((k+1-np.arange(1,k+1))*t)

    error = eps**4 + 2*eps*eps*np.sum(np.sqrt(psi)*np.sqrt(chi)*coses)
    return error

def error_integrand(t, n, chi, psi, eps, order=0):
    if order==0:
        prob = prob_dist(t) + error_prob(t, chi, psi, eps)
        integrand = np.power(prob, n)
    elif order==1:
        integrand = np.power(prob_dist(t), n) + n*error_prob(
            t, chi, psi, eps)*np.power(prob_dist(t), n-1)
    elif order==2:
        integrand = np.power(prob_dist(t), n) + n*error_prob(
            t, chi, psi, eps)*np.power(prob_dist(t), n-1) + n*(n-1)/2*np.power(
            error_prob(t, chi, psi, eps), 2) * np.power(prob_dist(t), n-2)
    
    return integrand

def calc_hierarchy_approx(coeff, eps, n=3, order=0):
    k = int(len(coeff)/2)
    psi = np.array(coeff[:k])
    chi = np.array(coeff[k:])
    global system
    global m_state
    system = q.ket2dm(state(*coeff[:k]))
    m_state = state(*coeff[k:])
    
    og_res = -func(coeff, n=n)
    eps_psi = EpsState(list(psi), eps)
    eps_chi = EpsState(list(chi), eps)
    expec_res = -func(eps_psi+eps_chi, n=n)
    
    system = q.ket2dm(state(*coeff[:k]))
    m_state = state(*coeff[k:])
    
    omega = np.sum(psi*chi)
    e1 = (1+eps*eps)**2
    e2 = (omega + eps**4)**(n-1)
    denom = e1 * e2
    
    
    numer = 1/(2*np.pi) * ig.quad(
            error_integrand, 0., 2*np.pi, (n, chi, psi, eps, order), 
            full_output=0)[0]
    
    return og_res, numer/denom, expec_res, (e1, omega, e2)



def check_hierarchy(k, epsila, iters, n=3, order=0, coeffs=None):
    results = np.zeros((3, len(epsila), iters))
    errors = np.zeros((3, len(epsila), iters))
    bad_coeffs = []
    
    for i in range(iters):
        if coeffs is None:
            chi = list(*np.random.dirichlet(np.ones(k),size=1))
            psi = chi #list(*np.random.dirichlet(np.ones(k),size=1))
            coeff = chi+psi
        else:
            coeff = coeffs[i]
        for j in range(len(epsila)):
            if j%10==0: print(i,j)
            og, res, exp, errs = calc_hierarchy_approx(coeff, epsila[j], n=n, 
                                                       order=order)
            results[0,j,i] = og
            results[1,j,i] = res
            results[2,j,i] = exp
            errors[0,j,i] = errs[0]
            errors[1,j,i] = errs[1]
            errors[2,j,i] = errs[2]
        
        if np.where(results[0,1:5,i] > results[1,1:5,i])[0].size>0:
            bad_coeffs.append(coeff)
    return results, errors, bad_coeffs
            
        
         

def func(args, s=0, n=3):
    global warnings
    warnings=False
    global system
    global m_state
    system = mixed_state(args[:int(len(args)/2)], s)
    m_state = state(*args[int(len(args)/2):])
    #print(m_state)
    return -(moment(n)/(moment(1)**(n-1)))

def func_half(var, param=-1, meas=True, s=0, n=3):
    global warnings
    warnings=False
    global system
    global m_state
    
    if meas:
         system = mixed_state([*var],s)
         m_state = state(*param)
    else:
         system = mixed_state([*param],s)
         m_state = state(*var)
    return -(moment(n)/(moment(1)**(n-1)))

def norm_con1_half(args):
    return np.sum(args) - 1.

def norm_con1(args):
    return np.sum(args[:int(len(args)/2)]) - 1.

def norm_con2(args):
    return np.sum(args[int(len(args)/2):]) - 1.

def guess_func(N_):
    vals = np.linspace(-1,1,N_)
    c = 1.15/N_ # numerical
    guess = 3*(N_-1)/(N_+1.)*(1./N_ -c)*vals**2 + c
    return list(guess)

def rand_ginibre(N_):
    return q.Qobj(np.sum(np.random.randn(*((N_,N_) + (2,))) * 
                         np.array([1,1j]), axis=-1))

def imperfect_state(coeff,error=0.2, H_rand=False, with_phase=False):
    s = state(*coeff)
    if H_rand == False:
        H_rand = q.rand_dm_ginibre(len(coeff))
    H_rand = H_rand/H_rand.norm()
    H_rand = q.Qobj(lin.block_diag(H_rand[:],
                    np.zeros((N-len(coeff),N-len(coeff)))))
    s_e = q.sesolve(H_rand, s, [0.,error]).states[1]
    if with_phase:
        return s_e
    else:
        return list(np.ndarray.flatten(np.abs(s_e[:len(coeff)])**2))
    


def find_max_func(N_, guess=[], s=0, n=3):
    if guess == []:
        #guess = (1./N_)*np.ones(2*N_)
        guess = np.tile(guess_func(N_),2)
    result = opt.minimize(func, guess, args=(s, n),
                        bounds=[(0,1.) for i in range(2*N_)],
                        constraints = [{'type':'eq', 'fun': norm_con1},
                                       {'type':'eq', 'fun': norm_con2}])
    return result #(result.x[:int(len(guess)/2)], result.x[int(len(guess)/2):])

def find_max_func_half(N_, param, guess=[], meas=True, s=0, n=3):
    if guess == []:
         #guess = np.array(np.load('meas_prob.npy')[N_-1])
         guess = guess_func(N_)
    result = opt.minimize(func_half, guess, args=(param, meas, s, n),
                        bounds=[(0.,1.) for i in range(N_)],
                        constraints = [{'type':'eq', 'fun': norm_con1_half}])
    return result #(result.x[:int(len(guess)/2)], result.x[int(len(guess)/2):])



def find_mixed_thresh(N_, n, error=0):
    max_coh = list((1./N_)*np.ones(N_)) 
    opt_res = find_max_func_half(N_,max_coh,[],True,0,n)
    result = -opt_res['fun']
    meas_coeff = opt_res['x']
    max_coh_small = list((1./(N_-1))*np.ones(N_-1))
    bound = -find_max_func_half(N_-1,max_coh_small,[],True,0,n)['fun']
    if error:
        meas_coeff = imperfect_state(meas_coeff,error)
    s = 0
    while result > bound:
        s+=0.01
        result = -func_half(max_coh,meas_coeff,True,s,n)
    return (s,result)
    
    
def plot_mixed_thresh(n_max, N_max, error=0):
    global system
    global m_state
    nvals = np.arange(3,n_max+1)
    Nvals = np.arange(3,N_max+1)
    vals = np.empty((len(nvals),len(Nvals)))
    for i in range(len(nvals)):
#        vals = np.array([[ 0.19, 0.13, 0.1,  0.08, 0.06, 0.06, 0.05, 0.04],
#[ 0.28, 0.19, 0.14, 0.11, 0.09, 0.08, 0.07, 0.06],
# [ 0.33, 0.22, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07],
# [ 0.36, 0.24, 0.18, 0.14, 0.12, 0.1,  0.09, 0.08],
# [ 0.39, 0.26, 0.19, 0.15, 0.13, 0.11, 0.1,  0.09],
# [ 0.4,  0.27, 0.2,  0.16, 0.13, 0.11, 0.1,  0.09],
# [ 0.41, 0.28, 0.21, 0.16, 0.14, 0.12, 0.11, 0.09],
# [ 0.42, 0.28, 0.21, 0.17, 0.14, 0.12, 0.1,  0.1 ],
# [ 0.43, 0.29, 0.21, 0.17, 0.15, 0.12, 0.11, 0.1 ],
# [ 0.44, 0.29, 0.22, 0.18, 0.14, 0.13, 0.11, 0.1 ],
# [ 0.44, 0.29, 0.22, 0.18, 0.15, 0.13, 0.11, 0.1 ],
# [ 0.45, 0.3,  0.22, 0.18, 0.15, 0.13, 0.11, 0.1 ],
# [ 0.45, 0.3,  0.23, 0.18, 0.15, 0.13, 0.11, 0.1 ]])
#        break
        for j in range(len(Nvals)):
            print(i+3,j+3)
            vals[i,j]= find_mixed_thresh(Nvals[j],nvals[i], error)[0]
            
    
    n_vals, N_vals = np.meshgrid(Nvals,nvals)
    print(nvals,Nvals)
    print(vals)
    
    
    plt.figure(1)
    for i in range(len(Nvals)):
        y_vals = vals[:,i]
        x_vals = nvals
        plt.plot(x_vals,y_vals, 'b-')
        plt.plot(nvals, (1/(Nvals[i]-1))*np.ones(len(nvals)), 'r--', alpha=0.5)
    plt.ylabel('$\lambda_{th}$')
    plt.xlabel('Moment Index n')
    
    plt.figure(2)
    for i in range(len(Nvals)):
        theory = (1/(Nvals[i]-1))*np.ones(len(nvals))
        y_vals = (theory-vals[:,i])/theory
        x_vals = nvals
        plt.plot(x_vals,y_vals)
        plt.ylabel(r'Fractional Difference', fontsize=16)
        plt.xlabel(r'Moment Index $n$', fontsize=16)
        plt.xlim(3,n_max)
        
    
    
    
    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    ax.plot_surface(n_vals,N_vals,vals,cmap='winter',antialiased=False,linewidth=0)
    
    # Theoretical k-1 threshold
    for i in range(len(Nvals)):
        theor_thresh = (1./(Nvals[i]-1))*np.ones(len(nvals))
        dummy_n = Nvals[i]*np.ones(len(nvals))
        ax.plot(dummy_n, nvals, theor_thresh, 'r--', alpha=0.5)
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)
    ax.set_xlabel(r'$k$',fontsize=16)
    ax.set_ylabel(r'$n$',fontsize=16)
    ax.set_zlabel(r'$\lambda_{thresh}$',fontsize=16)
    plt.show()
    
    fig = plt.figure(4)
    ax = fig.gca(projection='3d')
    # Theoretical k-1 threshold
    for i in range(len(Nvals)):
        theor_thresh = (1./(Nvals[i]-1))*np.ones(len(nvals))
        dummy_n = Nvals[i]*np.ones(len(nvals))
        ax.plot(dummy_n, nvals, theor_thresh, 'r--', alpha=0.5)
        ax.plot(dummy_n, nvals, vals[:,i], 'b-')
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)
    ax.set_xlabel(r'$k$',fontsize=16)
    ax.set_ylabel(r'$n$',fontsize=16)
    ax.set_zlabel(r'$\lambda_{thresh}$',fontsize=16)
    ax.set_xlim(3,N_max)
    ax.set_ylim(3,n_max)
    plt.show()


def plot_func_sensitivity(epsilon_max, N_err=5, step_size=0.01):
    global system
    global m_state
    N_=3
    n=3
    s=0
    max_coh = list((1./N_)*np.ones(N_)) 
    opt_res = find_max_func_half(N_,max_coh,[],False,s,n)
    low_bound = -find_max_func_half(N_-1,list((N_/(N_-1))*
                        np.array(max_coh)[:-1]),[],False,s,n)['fun']
    meas_coeff = opt_res['x']
    epsilon_list = np.linspace(0,epsilon_max,int(epsilon_max/step_size))
    alpha_list = np.empty((N_err,int(epsilon_max/step_size)))
    val_list = np.empty((N_err,int(epsilon_max/step_size)))
    for i in range(N_err):
        H_rand = q.rand_dm_ginibre(N_,1)
        for j in range(len(epsilon_list)):
            meas_coeff_err = imperfect_state(meas_coeff,epsilon_list[j],
                                             H_rand,with_phase=True)
            alpha_list[i][j] = np.sqrt(np.sum(np.abs(np.ndarray.flatten(
                            meas_coeff_err[:N_])-np.sqrt(meas_coeff))**2))
            m_state = meas_coeff_err
            system = mixed_state(max_coh,s)
            val_list[i][j] = moment(n)/moment(1)**(n-1)#-func(max_coh+meas_coeff_err,s,n)

    for i in range(N_err):
        #plt.plot(epsilon_list,val_list[i])
        plt.plot(alpha_list[i],val_list[i])
    plt.plot([0.,np.max(alpha_list)],[low_bound,low_bound],'k--')
    plt.xlim(0,np.max(alpha_list))
    plt.xlabel('||U|x>-|x>||', fontsize=16)
    plt.ylabel(r'$F_3$',fontsize=16, rotate=90)



def pattern_diff(t, args, N_,prob_no):
    m_state_t = q.sesolve(-H, m_state, [0.,t]).states[1]
    P_mixed = q.expect(system, m_state_t)
    probs = np.array(args[-prob_no:])
    P_list = np.empty(int(len(args[:-prob_no])/(N_-1)))
    
    for i in range(int(len(args[:-prob_no])/(N_-1))):
        state_coeffs = [0,*args[i*(N_-1):(i+1)*(N_-1)]]
        state_coeffs = np.roll(state_coeffs,i)
        pattern_state = q.ket2dm(state(*state_coeffs))
        P_list[i]= q.expect(pattern_state, m_state_t)

    return np.abs(P_mixed -  np.sum(probs*P_list))
    
def pattern_diff_func(args, N_, prob_no):
#    result = ig.quad(pattern_diff, 0., 2*np.pi, 
#                     (args,N_, prob_no), full_output=0)[0]
#    result = result/(2*np.pi)
    
    # Discretised integral for efficiency
    result=0
    steps = 100
    for i in np.linspace(0,2*np.pi,steps):
        result += pattern_diff(i,args,N_, prob_no)
    result = result/steps
    
    print(result)
    return result

def bindConstFunction(name,N_,i):
    def func(args):
        return np.sum(args[i*(N_-1):(i+1)*(N_-1)]) - 1.
    func.__name__ = name
    return func

def minimise_pattern_diff(N_, thresh = False, redund=1):
    global warnings
    warnings=False
    global m_state
    global system
    n = 3
    max_coh = list((1./N_)*np.ones(N_))
    opt_res = find_max_func_half(N_,max_coh,[],True,0,n)
    meas_coeff = opt_res['x']
    m_state = state(*meas_coeff)
    #print(m_state)
    
    #m_state = q.Qobj(np.concatenate([q.rand_ket(N_)[:].T[0],np.zeros(N-N_)]).T)
    #print(m_state)
    
    if thresh == False:
        threshold = find_mixed_thresh(N_,n)[0]
        print(threshold)
    else:
        threshold=thresh
    system = mixed_state(max_coh, threshold)
    #print(system)
#    system = q.ket2dm(np.concatenate([q.rand_ket(N_)[:].T[0],np.zeros(N-N_)]).T, threshold)
#    print(system)

    state_no = int(misc.comb(N_,N_-1))
    
    constraints = [{'type':'eq', 'fun': bindConstFunction('func'+str(i),N_,i)} 
                for i in range(redund*state_no)]
    constraints.append({'type':'eq', 'fun': (lambda args: 
                    np.sum(args[-redund*state_no:])-1.)})

    arg_no = redund*(N_-1)*state_no
    arg_guess= []
    for i in range(redund*state_no):
        arg_guess += list(*np.random.dirichlet(np.ones(N_-1),size=1))
    prob_guess = list(*np.random.dirichlet(np.ones(redund*state_no),size=1))
    guess = arg_guess+prob_guess
    print(guess)

    result = opt.minimize(pattern_diff_func, guess, args=(N_, redund*state_no),
                        bounds=[(0,1.) for i in range(arg_no+redund*state_no)],
                        constraints = constraints, method='SLSQP')
    
    
    total_state_no = state_no*redund
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(2, total_state_no)
    total_ax = fig.add_subplot(grid[0, :])
    axarr = [fig.add_subplot(grid[1,0])]
    axarr += [fig.add_subplot(grid[1,i],yticklabels=[]) 
            for i in range(1,total_state_no)]
    
    
    coeffs = result['x']
    probs = np.array(coeffs[-state_no*redund:])
    pattern_state = []
    for i in range(int(len(coeffs[:-state_no*redund])/(N_-1))):
        state_coeffs = [0,*coeffs[i*(N_-1):(i+1)*(N_-1)]]
        state_coeffs = np.roll(state_coeffs,i)
        pattern_state.append(q.ket2dm(state(*state_coeffs)))
    system = np.sum(probs*pattern_state)
    plot_pattern('Mixed '+str(N_-1)+'-coherent State', total_ax)
    
    system = mixed_state(max_coh, threshold)
    plot_pattern('Mixed '+str(N_) +r'-coherent State ($\lambda$='+
                 str(round(threshold,3))+')', total_ax)
    total_ax.legend()

    for i in range(len(axarr)):
        system = pattern_state[i]
        plot_pattern('',axarr[i])
    
    return result
    
    
    



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