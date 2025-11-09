README: Thermodynamic Circuit Framework – Toy Simulation

Author: Logan Ohm
Date: November 2025
File: TCT_toy_sim.py


---

Overview

This repository contains a proof‑of‑concept simulation for the Thermodynamic Circuit Framework (TCT), implementing a minimal numerical model of the χ‑continuum system described in the paper “The Thermodynamic Circuit Framework: Mathematical Foundation.”

The simulation demonstrates how self‑sustaining dynamics and a measurable Ohm metric (℧) can emerge as the predictive coupling parameter (α) increases, showing a phase transition consistent with the theoretical critical threshold ℧₍critical₎.


---

What It Does

Integrates simplified versions of the TCT field equations for:

V(x,t) – Fast voltage field (ms‑scale dynamics)

M(x,t) – Slow memory field (s–years scale)

A(t) – Global affect modulation field (10 ms–1 s scale)


Computes a practical ℧‑proxy from spatial and temporal statistics of the simulated fields.

Sweeps the coupling parameter α to reveal a bifurcation: a clear transition between damped and self‑sustaining behavior.



---

Dependencies

Python ≥ 3.9 and the following libraries:

pip install numpy matplotlib scipy


---

Running the Simulation

Execute directly in a terminal or IDE:

python TCT_toy_sim.py

This will:

1. Sweep α from 0 → 1.6


2. Compute the normalized mean ℧‑metric for each α


3. Display a bifurcation plot showing normalized ℧ vs α



The console output lists normalized ℧ values and scaling factors used for normalization.


---

Output

Figure 1: Normalized ℧ (mean) vs α — bifurcation diagram

Console: Printed α values and corresponding normalized ℧


Expected result: A visible jump or nonlinear rise in normalized ℧ as α crosses a critical value, indicating emergence of self‑sustaining entropy flow.


---

Interpretation

The simulation illustrates how the Ohm metric (℧) behaves as an order parameter for conscious or self‑organized dynamics within the TCT model.
Crossing the critical threshold corresponds to the formation of a self‑sustaining dissipative circuit — a minimal numerical analog of conscious integration in the framework.


---

Collaboration Notes

This script can serve as a reproducible baseline for collaborators in computational neuroscience, physics, or systems modeling.

Suggested next steps:

1. Extend grid size and runtime for richer spatial dynamics.


2. Map ℧‑proxy components to empirical EEG/PCI features.


3. Add parameter sweeps over (α, γ, noise) to build phase diagrams.


4. Publish results alongside the theoretical paper in Entropy or Physica D.





---

Citation

If you use or modify this code, please cite:

> Ohm, L. (2025). The Thermodynamic Circuit Framework: Mathematical Foundation.
Independent research manuscript, November 2025.




---

© 2025 Logan Ohm · All rights reserved.
