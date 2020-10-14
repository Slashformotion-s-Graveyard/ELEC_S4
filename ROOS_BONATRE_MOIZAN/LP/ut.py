from SecondOrderElec import LP
from SecondOrderElec.plot import plot_time
import numpy as np

def show_influence_w0_fresqresp(w0, T0=1.5, m=0.9):
    T = LP(w0=w0, T0=T0, m=m)
    
    T.freqresp(
        w=np.logspace(0,5,10000)
    )

def show_influence_pz(w0, T0, m):
    T = LP(w0=w0, T0=T0, m=m)
    
    p,z = T.pzmap(plot=False)
    print(f"Zéros: {z}")
    print(f"Poles: {p}")
    print(f"|Poles|: {[np.abs(p_) for p_ in p]}")
    if -1<m<1:
        print(f"Sl=A exp({np.real(p[1])}t) cos({np.imag(p[0])}t + theta)")
    else:
        print(f"Sl=A exp({p[0]}t) + B exp({p[1]}t)")

def show_influence_sl(w0, T0, m):
    T = LP(w0=w0, T0=T0, m=m)
    # T.impulse()
    T.step()



