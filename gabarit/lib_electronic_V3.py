#
#  lib_electronic LTI system library
#
#  (C) Céline ANSQUER, Eric BOUCHARE, Vincent CHOQUEUSE
#
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy import linalg

from six import *


j=1j
########################################################
# basic signals generation
##
## t     : time
## f     : frequency
## alpha : period ratio
##
########################################################
def ustep(t):
## returns a unit step signal
	u=np.ones(t.shape)
	u[np.nonzero(t<0.0)] = 0.0
	return u


def ramp(t):
## returns a unit slope
	return t*ustep(t)


def pulse(t,theta=1.0):
## returns a pulse of width theta centered on 0
	return ustep(t+theta/2.0)-ustep(t-t>theta/2.0)


def triangle(t,theta=1.0):
# returns a triangle defined by 1 - |t|/theta for |t|<theta, else 0
	return (1.0-np.abs(t)/theta)*pulse(t,2*theta)


def sawwave(t,f=1.0):
## returns a vector containing a unit saw wave
	return t*f-np.floor(t*f)


def sqrwave(t,f=1.0,alpha=0.5):
## returns a vector containing the square signal
	return 2*(sawwave(t,f)<alpha)-1


def triwave(t,f=1.0,alpha=0.5):
## returns a vector containing the triangle signal
	s=(t*f-np.floor(t*f));
	return (2*(s/alpha*(s<=alpha)+(1-s)/(1-alpha)*(s>alpha))-1)

###################################
# Bode asymptotical  traces
# f  : frequency vector
# f0 : cut-off frequency
#
###################################

#1st order lowpass
def alp1(f,f0=1):
	
	H=np.ones(f.shape,dtype=complex)
	# after the cut-off frequency
	i=np.nonzero(f>=f0)
	H[i]=-1j*f0/f[i]
	return H

# 1st order high-pass
def ahp1(f,f0=1):
	H=np.ones(f.shape,dtype=complex)
	# before the cut-off frequency
	i=np.nonzero(f<=f0)
	H[i]=j*f[i]/f0
	return H

# 2nd order low-pass
def alp2(f,f0):
	H=np.ones(f.shape,dtype=complex)
	# after the cut-off frequency
	i=np.nonzero(f>=f0)
	H[i]=-(f0/f[i])**2
	return H

# 2nd order high-pass
def ahp2(f,f0=1):
	H=np.ones(f.shape,dtype=complex)
	# before the cut-off frequency
	i=np.nonzero(f<=f0)
	H[i]=-(f[i]/f0)**2
	return H

#######################################
#Simulate output of a continuous-time linear system.
########################################
def lsim(system, U, T, X0=None, interp=True):
    """
    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 1: (instance of `lti`)
        * 2: (num, den)
        * 3: (zeros, poles, gain)
        * 4: (A, B, C, D)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.  If U = 0 or None, a zero input is used.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.  Must be nonnegative, increasing, and equally spaced.
    X0 : array_like, optional
        The initial conditions on the state vector (zero by default).
    interp : bool, optional
        Whether to use linear (True, the default) or zero-order-hold (False)
        interpolation for the input array.

    Returns
    -------
    T : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time evolution of the state vector.

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    Examples
    --------
    Simulate a double integrator y'' = u, with a constant input u = 1

    >>> from scipy import signal
    >>> system = signal.lti([[0., 1.], [0., 0.]], [[0.], [1.]], [[1., 0.]], 0.)
    >>> t = np.linspace(0, 5)
    >>> u = np.ones_like(t)
    >>> tout, y, x = signal.lsim(system, u, t)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(t, y)
    """
    if isinstance(system, sig.lti):
        sys = system._as_ss()
    elif isinstance(system, sig.dlti):
        raise AttributeError('lsim can only be used with continuous-time '
                             'systems.')
    else:
        sys = sig.lti(*system)._as_ss()
    T = np.atleast_1d(T)
    if len(T.shape) != 1:
        raise ValueError("T must be a rank-1 array.")

    A, B, C, D = map(np.asarray, (sys.A, sys.B, sys.C, sys.D))
    n_states = A.shape[0]
    n_inputs = B.shape[1]

    n_steps = T.size
    if X0 is None:
        X0 = np.zeros(n_states, sys.A.dtype)
    xout = np.zeros((n_steps, n_states), sys.A.dtype)

    if T[0] == 0:
        xout[0] = X0
    else:
        xout[0] = np.dot(X0, linalg.expm(np.transpose(A) * T[0]))
#    elif T[0] > 0:
#        # step forward to initial time, with zero input
#        xout[0] = dot(X0, linalg.expm(transpose(A) * T[0]))
#    else:
#        raise ValueError("Initial time must be nonnegative")

    no_input = (U is None or
                (isinstance(U, (int, float)) and U == 0.) or
                not np.any(U))

    if n_steps == 1:
        yout = np.squeeze(np.dot(xout, np.transpose(C)))
        if not no_input:
            yout += np.squeeze(np.dot(U, np.transpose(D)))
        return T, np.squeeze(yout), np.squeeze(xout)

    dt = T[1] - T[0]
    if not np.allclose((T[1:] - T[:-1]) / dt, 1.0):
        warnings.warn("Non-uniform timesteps are deprecated. Results may be "
                      "slow and/or inaccurate.", DeprecationWarning)
        return sig.lsim2(system, U, T, X0)

    if no_input:
        # Zero input: just use matrix exponential
        # take transpose because state is a row vector
        expAT_dt = linalg.expm(np.transpose(A) * dt)
        for i in xrange(1, n_steps):
            xout[i] = np.dot(xout[i-1], expAT_dt)
        yout = np.squeeze(np.dot(xout, np.transpose(C)))
        return T, np.squeeze(yout), np.squeeze(xout)

    # Nonzero input
    U = np.atleast_1d(U)
    if U.ndim == 1:
        U = U[:, np.newaxis]

    if U.shape[0] != n_steps:
        raise ValueError("U must have the same number of rows "
                         "as elements in T.")

    if U.shape[1] != n_inputs:
        raise ValueError("System does not define that many inputs.")

    if not interp:
        # Zero-order hold
        # Algorithm: to integrate from time 0 to time dt, we solve
        #   xdot = A x + B u,  x(0) = x0
        #   udot = 0,          u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
        #   [ u(dt) ] = exp [  0     0    ] [ u0 ]
        M = np.vstack([np.hstack([A * dt, B * dt]),
                       np.zeros((n_inputs, n_states + n_inputs))])
        # transpose everything because the state and input are row vectors
        expMT = linalg.expm(np.transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd = expMT[n_states:, :n_states]
        for i in xrange(1, n_steps):
            xout[i] = np.dot(xout[i-1], Ad) + np.dot(U[i-1], Bd)
    else:
        # Linear interpolation between steps
        # Algorithm: to integrate from time 0 to time dt, with linear
        # interpolation between inputs u(0) = u0 and u(dt) = u1, we solve
        #   xdot = A x + B u,        x(0) = x0
        #   udot = (u1 - u0) / dt,   u(0) = u0.
        #
        # Solution is
        #   [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
        #   [ u(dt) ] = exp [  0     0    I ] [  u0   ]
        #   [u1 - u0]       [  0     0    0 ] [u1 - u0]
        M = np.vstack([np.hstack([A * dt, B * dt,
                                  np.zeros((n_states, n_inputs))]),
                       np.hstack([np.zeros((n_inputs, n_states + n_inputs)),
                                  np.identity(n_inputs)]),
                       np.zeros((n_inputs, n_states + 2 * n_inputs))])
        expMT = linalg.expm(np.transpose(M))
        Ad = expMT[:n_states, :n_states]
        Bd1 = expMT[n_states+n_inputs:, :n_states]
        Bd0 = expMT[n_states:n_states + n_inputs, :n_states] - Bd1
        for i in xrange(1, n_steps):
            xout[i] = (np.dot(xout[i-1], Ad) + np.dot(U[i-1], Bd0) + np.dot(U[i], Bd1))

    yout = (np.squeeze(np.dot(xout, np.transpose(C))) + np.squeeze(np.dot(U, np.transpose(D))))
    return T, np.squeeze(yout), np.squeeze(xout)

#####################################
#time samples 
###################################
def _default_response_times(A, n):
    """Compute a reasonable set of time samples for the response time.

    This function is used by `impulse`, `impulse2`, `step` and `step2`
    to compute the response time when the `T` argument to the function
    is None.

    Parameters
    ----------
    A : array_like
        The system matrix, which is square.
    n : int
        The number of time samples to generate.

    Returns
    -------
    t : ndarray
        The 1-D array of length `n` of time samples at which the response
        is to be computed.
    """
    # Create a reasonable time interval.
    # TODO: This could use some more work.
    # For example, what is expected when the system is unstable?
    vals = linalg.eigvals(A)
    r = np.min(np.abs(np.real(vals)))
    if r == 0.0:
        r = 1.0
    tc = 1.0 / r
    t = np.linspace(0.0, 4 * tc, n)
    return t

############################################
#Impulse response of continuous-time system.
############################################
def impulse(system, X0=None, T=None, N=None):
    """
    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (np.zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector.  Defaults to zero.
    T : array_like, optional
        Time points.  Computed if not given.
    N : int, optional
        The number of time points to compute (if `T` is not given).

    Returns
    -------
    T : ndarray
        A 1-D array of time points.
    yout : ndarray
        A 1-D array containing the impulse response of the system (except for
        singularities at zero).

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    """
    if isinstance(system, sig.lti):
        sys = system._as_ss()
    elif isinstance(system, sig.dlti):
        raise AttributeError('impulse can only be used with continuous-time '
                             'systems.')
    else:
        sys = sig.lti(*system)._as_ss()
    if X0 is None:
        X = np.squeeze(sys.B)
    else:
        X = np.squeeze(sys.B + X0)
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = np.asarray(T)

    _, h, _ = lsim(sys, 0., T, X, interp=False)
    return T, h*ustep(T)

#########################################
#Step response of continuous-time system.
#########################################
def step(system, X0=None, T=None, N=None):
    """
    Parameters
    ----------
    system : an instance of the LTI class or a tuple of array_like
        describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1 (instance of `lti`)
            * 2 (num, den)
            * 3 (np.zeros, poles, gain)
            * 4 (A, B, C, D)

    X0 : array_like, optional
        Initial state-vector (default is zero).
    T : array_like, optional
        Time points (computed if not given).
    N : int, optional
        Number of time points to compute if `T` is not given.

    Returns
    -------
    T : 1D ndarray
        Output time points.
    yout : 1D ndarray
        Step response of system.

    See also
    --------
    scipy.signal.step2

    Notes
    -----
    If (num, den) is passed in for ``system``, coefficients for both the
    numerator and denominator should be specified in descending exponent
    order (e.g. ``s^2 + 3s + 5`` would be represented as ``[1, 3, 5]``).

    """
    if isinstance(system, sig.lti):
        sys = system._as_ss()
    elif isinstance(system, sig.dlti):
        raise AttributeError('step can only be used with continuous-time '
                             'systems.')
    else:
        sys = sig.lti(*system)._as_ss()
    if N is None:
        N = 100
    if T is None:
        T = _default_response_times(sys.A, N)
    else:
        T = np.asarray(T)
    U = ustep(T)
    vals = lsim(sys, U, T, X0=X0, interp=False)
    return vals[0], vals[1]

#########################################################
#classe TF
#define Transfer Function object (n order) and many usual behaviors 
#########################################################
class TF():
    # classe TF
    nb_zeros_origin = 0
    
    def __init__(self,*system):
        self.system = system
    
    def lti(self):
        return sig.lti(*self.system)

    def save_csv(self,filename,data,type=None):
        M = np.matrix(data)
        if type is not None:
            filename="{}_{}.csv".format(filename,type)
        else:
            filename="{}.csv".format(filename)
        
        np.savetxt(filename,M.T,delimiter=",",comments="")

    def zpk(self,show=True,normalized_factor=1,**options):

        Tsys = self.lti()
        poles=Tsys.poles/normalized_factor
        zeros=Tsys.zeros/normalized_factor
        
        if show:
            plt.plot(zeros.real,zeros.imag, 'bo', markersize=5, label="np.zeros")
            plt.plot(poles.real,poles.imag, 'rx', markersize=5, label="poles")
            
            maximum = 0
            if len(zeros) > 0:
                maximum = max(maximum,np.max(np.abs(zeros)))
            
            maximum = max(maximum,np.max(np.abs(poles)))
            plt.axis("scaled")
            plt.xlabel("Partie Réelle")
            plt.ylabel("Partie Imaginaire")
            plt.axis(1.5*maximum*np.array([-1, 1, -1, 1]))

            if self.nb_zeros_origin > 1:
                plt.text(0.05*maximum,0.05*maximum, "$\\times {}$".format(self.nb_zeros_origin), fontsize=10,color="b")
            
            plt.grid()
            plt.legend()

        return poles, zeros
    
    def step(self,T=None,show=True,filename=None,**options):
        Tsys = self.lti()
        t,s = step(Tsys,T=T)
        if show:
            plt.plot(t,s,**options)
            plt.xlabel("temps [s]")
            plt.ylabel("Réponse Indicielle")
            plt.grid()
        if filename:
            self.save_csv(filename,[t,s],type="RI")
        return t,s

    def impulse(self,T=None,show=True,filename=None,**options):
        Tsys = self.lti()
        t,s = impulse(Tsys,T=T)
        if show:
            plt.plot(t,s,**options)
            plt.xlabel("temps [s]")
            plt.ylabel("Réponse Impulsionnelle")
            plt.grid()
        if filename:
            self.save_csv(filename,[t,s],type="IM")
        return t,s
    
    def output(self,input,T,show=True,filename=None,**options):
        Tsys = self.lti()
        t, s, x  = lsim(Tsys,input,T)
        if show:
            plt.plot(t,s,**options)
            plt.xlabel("temps [s]")
            plt.ylabel("Réponse Temporelle")
            plt.grid()
        if filename:
            self.save_csv(filename,[t,s],type="IM")
        return t,s
    
    def modulus(self,w=None,show=True,filename=None,**options):
        Tsys = self.lti()
        w, Tjw = Tsys.freqresp(w)
        modulus = np.abs(Tjw)
        if show:
            plt.loglog(w/(2*np.pi),modulus,**options)
            plt.xlabel("f [Hz]")
            plt.ylabel("module = amplification Vs/ve")
            plt.grid()
        if filename:
            self.save_csv(filename,[w,modulus],type="B_M")
        return w,modulus
    
    def phase(self,w=None,show=True,filename=None,**options):
        Tsys = self.lti()
        w, Tjw = Tsys.freqresp(w)
        phase = 180*np.angle(Tjw)/np.pi
        if show:
            plt.semilogx(w/(2*np.pi),phase,**options)
            plt.xlabel("f [Hz]")
            plt.ylabel("argument = déphasage ")
            plt.grid()
        if filename:
            self.save_csv(filename,[w,phase],type="B_P")
        return w,phase

##########################
#several 2d ordrer filters
##########################
class TF_2nd_LP(TF):
    # Herite de la classe TF
    nb_zeros_origin = 0
    
    def __init__(self,T0=1,m=0.5,w0=10):
        self.T0 = T0
        self.m = m
        self.w0 = w0
    
    def lti(self):
        den = np.array([(1/(self.w0**2)),2*self.m/self.w0,1])
        num = np.array([self.T0])
        return sig.lti(num,den)

class TF_2nd_HP(TF):
    # Herite de la classe TF
    nb_zeros_origin = 2
    
    def __init__(self,Too=1,m=0.5,w0=10):
        self.Too = Too
        self.m = m
        self.w0 = w0
    
    def lti(self):
        den = np.array([(1/(self.w0**2)),2*self.m/self.w0,1])
        num = np.array([self.Too/(self.w0**2),0,0])
        return sig.lti(num,den)

class TF_2nd_BP(TF):
    # Herite de la classe TF
    nb_zeros_origin = 1
    
    def __init__(self,Tmax=1,m=0.5,w0=10):
        self.Tmax = Tmax
        self.m = m
        self.w0 = w0
    
    def lti(self):
        den = np.array([(1/(self.w0**2)),2*self.m/self.w0,1])
        num = np.array([2*self.m*self.Tmax/self.w0,0])
        return sig.lti(num,den)

class TF_2nd_Notch(TF):
    # Herite de la classe TF
    nb_zeros_origin = 0
    
    def __init__(self,T0=1,m=0.5,w0=10):
        self.T0 = T0
        self.m = m
        self.w0 = w0
    
    def lti(self):
        den = np.array([(1/(self.w0**2)),2*self.m/self.w0,1])
        num = np.array([self.T0/(self.w0**2),0,self.T0])
        return sig.lti(num,den)


if __name__== "__main__":
    system = TF_2nd_LP(2,0.2,10)
    system.zpk()
    plt.figure()
    system.step()
    plt.figure()
    system.modulus()
    plt.show()

################################
#protype trace
################################
def draw_prototype(xc,yc,xs,ys,xaxes_type,yaxes_type,filter_type="lowpass"):
	
	if xaxes_type == "w":
		plt.gca().set_xlabel('pulsation en rad/s')
	else:
		plt.gca().set_xlabel('fréquence en Hz')
		
	if yaxes_type == "T":
		y0=1
		ymin= ys/100
		ymax=y0*10
		plt.yscale('log')
		plt.gca().set_ylabel('amplification sans unité')
		
	if yaxes_type == "G":
		y0=0
		ymin=ys-40
		ymax=y0+20
		plt.gca().set_ylabel('gain en dB')
		
	if filter_type == "lowpass":
		xmin=xc/100
		xmax=xs*100
		# draw polygons
		patch1 = plt.Polygon([[xmin,yc],[xc,yc],[xc,ymin],[xmin,ymin]],fill=False,color="b",hatch="/")
		patch2 = plt.Polygon([[xmin,y0],[xs,y0],[xs,ys],[xmax,ys],[xmax,ymax],[xmin,ymax]],fill=False,color="b",hatch="/")
		
	if filter_type == "highpass":
		xmin=xs/100
		xmax=xc*100
		#draw polygons
		patch1 = plt.Polygon([[xc,yc],[xmax,yc],[xmax,ymin],[xc,ymin]],fill=False,color="b",hatch="/")
		patch2 = plt.Polygon([[xmax,y0],[xs,y0],[xs,ys],[xmin,ys],[xmin,ymax],[xmax,ymax]],fill=False,color="b",hatch="/")
	#trace
	
	plt.gca().set_xlim(xmin, xmax)
	plt.xscale('log')
	plt.gca().set_ylim(ymin, ymax)
	# add patch to axes
	plt.gca().add_patch(patch1)
	plt.gca().add_patch(patch2)
	plt.show()		


#################
#grid definition
#################
def graph_poles(wc):
	plt.gca()
	plt.axhline(0,color="k")
	plt.axvline(0,color="k")
	plt.axis('scaled')
	plt.title("diagramme des pôles normalisés pour les 4 aproximations")
	plt.xlabel("Partie Réelle /wc");
	plt.ylabel("Partie Imaginaire /wc")
	plt.axis([-2*wc, 1*wc,-2*wc,2*wc])
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)