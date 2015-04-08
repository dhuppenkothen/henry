# Exploratory script to look at a prior, to ballpark sensible hyper-parameters,
# and to look at what inter-arrival distributions might look like

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#ell = 1.5 # lengthscale
ell = 1.0 *90000/10 # lengthscale
# Time range:
T0 = 0
Tend = 90000
T = Tend - T0
Nbins = 300
tbins = np.linspace(T0, Tend, Nbins)
# Basis function centres:
c0 = T0 - ell*5
cc = np.arange(T0 - ell*5, Tend + ell*5, ell)
K = len(cc) # Number of basis functions

# Prior properties
pi_mu = -5.0
pi_std = 1.0


Phi = np.zeros((K, Nbins))
for kk in range(K):
    Phi[kk] = np.exp(-0.5*(tbins - cc[kk])**2/ell**2)

## Draws from prior:
# plt.figure(1)
# plt.clf()
# for kk in range(K):
#     plt.plot(tbins, Phi[kk])
# plt.show()
# 
# plt.figure(2)
# plt.clf()
# for ss in range(12):
#     ww = sp.randn(K)*pi_std
#     plt.plot(tbins, np.exp(pi_mu + np.dot(ww, Phi)))
# plt.show()

# Draw a random intensity
ww = sp.randn(K)*pi_std
intensity = np.exp(pi_mu + np.dot(ww, Phi))

# Draw simulated events by thinning:
max_intensity = intensity.max()
dom_rate = max_intensity * T
num_points = np.random.poisson(dom_rate)
point_ts = np.random.rand(num_points)*T + T0
point_hs = np.random.rand(num_points)*max_intensity
point_Phi = np.zeros((K, num_points))
for kk in range(K):
    point_Phi[kk] = np.exp(-0.5*(point_ts - cc[kk])**2/ell**2)
point_lam = np.exp(pi_mu + np.dot(ww, point_Phi))
tt = point_ts[point_hs < point_lam]
tt.sort()

plt.figure(3)
plt.clf()
#plt.plot(tt, np.zeros_like(tt), 'x')
h, bins, patches = plt.hist(tt, bins=500)
bin_size = bins[1]-bins[0]
plt.plot(tbins, intensity*bin_size, 'r', linewidth=3)
plt.show()

plt.figure(4)
plt.hist(np.log(np.diff(tt)), 100);
plt.show()

