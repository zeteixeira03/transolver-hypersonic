---
title: Hypersonic Sphere-Cone Surrogate
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
prediction. The geometry is meshed with gmsh by the recipe of the training
data, once per cone half-angle and scaled to the nose radius, and five ensemble
members run on the processor. The app reports the flow field, a verdict of
trust, warn or refuse from the ensemble spread and the distance to the training
box, and two analytical cross-checks (Billig shock standoff, Fay-Riddell
stagnation heat flux).

This is a research demonstration, not a certified engineering tool.

See the main project repository for the model, dataset, and evaluation study.
