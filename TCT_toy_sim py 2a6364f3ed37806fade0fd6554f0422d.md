# TCT_toy_sim.py

# 

# --------------------------------------------------------------

# Toy simulation of the Thermodynamic Circuit Framework (TCT)

# Implements simplified chi-continuum PDEs (Eqs. 6-8) from the paper

# Author: Logan Ohm | November 2025

# --------------------------------------------------------------

# Description:

# This script integrates fast (V), slow (M), and global (A) fields

# on a 2D grid under Neumann boundary conditions. It computes a

# practical proxy of the consciousness metric ℷ (Ohm metric) and

# shows a bifurcation as the predictive coupling parameter alpha increases.

# --------------------------------------------------------------

# Requirements: numpy, matplotlib, scipy

# Usage: python TCT_toy_sim.py

# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import laplace
from scipy.stats import entropy

# -- Helper functions ---

def neumann_laplacian(u, dx=1.0):
return laplace(u) / (dx**2)

def spatial_entropy_of_gradient(V, bins=64):
gx, gy = np.gradient(V)
gmag = np.sqrt(gx**2 + gy**2).ravel()
h, _ = np.histogram(gmag, bins=bins, density=True)
h = h[h > 0]
return entropy(h, base=2)

def spatial_autocorrelation(V):
Vf = V.ravel()
shifted = np.roll(V, 1, axis=0).ravel()
if np.std(Vf) == 0 or np.std(shifted) == 0:
return 0.0
return np.corrcoef(Vf, shifted)[0, 1]

def compute_tau_from_time_series(V_ts, dt):
sig = V_ts - np.mean(V_ts)
n = len(sig)
if np.all(sig == 0):
return 1.0
yf = np.abs(rfft(sig))
xf = rfftfreq(n, dt)
yf[0] = 0.0
fpeak = xf[np.argmax(yf)]
return 1.0 / fpeak if fpeak > 0 else 1.0

# -- Simulation parameters ---

grid_size = 48
dx = 1.0
dt = 0.05
nsteps = 800
record_every = 4
np.random.seed(1)

# Base constants (dimensionless toy values)

DV, DM = 0.6, 0.02
gamma, kappa = 0.2, 0.05
lambdaV, lambdaM = 0.08, 0.02
beta, delta = 1.0, 0.05
noiseV, noiseA = 0.02, 0.01

# Sweep range for predictive coupling alpha

alpha_vals = np.linspace(0.0, 1.6, 14)

def run_sim(alpha):
V = 0.01 * np.random.randn(grid_size, grid_size)
M = 0.01 * np.random.randn(grid_size, grid_size)
A = 0.0
V[grid_size//2-2:grid_size//2+3, grid_size//2-2:grid_size//2+3] += 0.2

```
nrec = nsteps // record_every
ups_hist = np.zeros(nrec)
V_mean_ts = []

for t in range(nsteps):
    lapV, lapM = neumann_laplacian(V, dx), neumann_laplacian(M, dx)
    etaV = noiseV * np.random.randn(grid_size, grid_size)
    zeta = noiseA * np.random.randn()

    # Field updates
    dVdt = DV * lapV + alpha * (M - V) - lambdaV * V + etaV
    V += dt * dVdt

    gx, gy = np.gradient(V)
    MgradV = np.gradient(M)[0] * gx + np.gradient(M)[1] * gy
    dMdt = DM * lapM + gamma * A * (V - M) - lambdaM * M + kappa * MgradV
    M += dt * dMdt

    dAdt = -beta * A + delta * (1.0 - np.mean(np.abs(M - V))) + zeta
    A += dt * dAdt

    V = np.clip(V, -5, 5)
    M = np.clip(M, -5, 5)

    if t % record_every == 0:
        V_mean_ts.append(np.mean(V))
        eps = spatial_entropy_of_gradient(V)
        tau = compute_tau_from_time_series(np.array(V_mean_ts[-128:]), dt * record_every)
        rho = np.mean(gx * V + gy * V)
        iota = spatial_autocorrelation(V)
        ups = (eps / max(tau, 1e-6)) * rho * (iota if not np.isnan(iota) else 0)
        ups_hist[t // record_every] = ups

return np.mean(ups_hist)

```

# -- Run sweep ---

mean_ups = np.array([run_sim(a) for a in alpha_vals])
base = np.median(mean_ups[:4])
scale = np.max(np.abs(mean_ups - base)) + 1e-9
ups_norm = (mean_ups - base) / scale

# -- Plot results ---

plt.figure(figsize=(7,4))
plt.plot(alpha_vals, ups_norm, marker='o')
plt.title('Toy bifurcation: normalized ℧ (mean) vs α')
plt.xlabel('Predictive coupling α')
plt.ylabel('Normalized ℧ (toy proxy)')
plt.grid(True)
plt.tight_layout()
plt.show()

print('Alpha values:', np.round(alpha_vals,3))
print('Normalized mean ℧:', np.round(ups_norm,4))
print('Baseline:', base, ' Scale:', scale)

# End of script
