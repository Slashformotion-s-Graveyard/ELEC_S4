from SecondOrderElec import LP, plot
import numpy as np
T = LP(w0=100, T0=2, m=0.5)
t, s = T.step(plot=False)
hs = np.hstack((np.zeros((int(len(t))),), t)) >0
print(hs)
print(t)

s = np.hstack((np.zeros((int(len(t))),), s))
t = np.hstack((np.zeros((int(len(t))),), t))
print(t)

plot.plot_time(t,s)
plot.plot_time(t,hs)