---
title: Hypersonic Sphere-Cone Surrogate
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.40.0
app_file: app/app.py
pinned: false
license: mit
---

# Hypersonic sphere-cone flow surrogate

Interactive demo of a Transolver deep ensemble that predicts the steady laminar
flow field around an axisymmetric sphere-cone re-entry capsule in hypersonic
flight. Ground truth is SU2 axisymmetric Navier-Stokes (ideal gas, calorically
perfect air, Sutherland viscosity, isothermal cold wall).

Four controls (Mach, altitude, nose radius, cone half-angle) drive a live
prediction: the geometry is meshed with gmsh at training resolution and five
ensemble members run on CPU. The app reports the flow field, an
uncertainty-aware trust/warn/refuse verdict from the ensemble spread and a
training-envelope guard, and two analytical cross-checks (Billig shock standoff,
Fay-Riddell stagnation heat flux).

One prediction takes tens of seconds on the free CPU tier. This is a research
demo, not a certified engineering tool.

See the main project repository for the model, dataset, and evaluation study.
