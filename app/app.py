"""Streamlit dashboard for the hypersonic sphere-cone flow surrogate.

Four sliders drive a live deep-ensemble prediction of the steady flow field
around an axisymmetric re-entry capsule, with an uncertainty-aware
trust/warn/refuse verdict and two analytical cross-checks. One prediction
meshes the geometry and runs five Transolver members on CPU, so it takes
tens of seconds; the run button gates it behind a spinner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LogNorm, Normalize
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import (
    GUARD_DIST, KN_MAX, R_B_RATIO, R_S_RATIO, T_WALL, WARN_SPREAD,
    load_ensemble, predict,
)

# ============================================================================================
#                                       page config
# ============================================================================================

st.set_page_config(page_title="Hypersonic sphere-cone surrogate", page_icon="🛰️", layout="wide")

FIELD_META = {
    "T": ("Temperature", "K", False),
    "rho": ("Density", "kg/m³", True),
    "p": ("Pressure", "Pa", True),
    "u": ("Axial velocity", "m/s", False),
    "v": ("Radial velocity", "m/s", False),
}
DECISION_STYLE = {
    "trust": ("✅ Trust", "#1a7f37", "Inputs sit inside the validated envelope and the "
              "ensemble agrees. Prediction is usable."),
    "warn": ("⚠️ Warn", "#9a6700", "Ensemble spread is elevated. Treat the field as "
             "indicative, not quantitative."),
    "refuse": ("⛔ Refuse", "#cf222e", "Inputs fall outside the validated envelope or the "
               "ensemble disagrees strongly. Run a full simulation instead."),
}

# nominal training box per exposed control, for the live per-slider OOD indicator
ENVELOPE_BOX = {
    "mach": (8.0, 25.0),
    "altitude": (45.0, 60.0),
    "R_n_mm": (10.0, 50.0),
    "theta_c": (35.0, 70.0),
}

# injected once; drives the flashing OOD chip
OOD_CSS = """
<style>
@keyframes oodflash {
  0%   { background:#cf222e; box-shadow:0 0 0 0 #cf222e88; }
  50%  { background:#ff5a5a; box-shadow:0 0 0 4px #cf222e00; }
  100% { background:#cf222e; box-shadow:0 0 0 0 #cf222e88; }
}
.ood-chip { display:inline-block; padding:2px 10px; border-radius:11px; color:#fff;
  font-size:0.76rem; font-weight:700; margin:-6px 0 6px 0;
  animation: oodflash 0.7s ease-in-out infinite; }
.ok-chip { display:inline-block; padding:2px 10px; border-radius:11px;
  font-size:0.76rem; font-weight:600; margin:-6px 0 6px 0;
  background:#1a7f3722; color:#1a7f37; }
</style>
"""


def envelope_chip(value: float, lo: float, hi: float, unit: str = "") -> str:
    """A flashing 'out of envelope' chip when the value leaves the training box."""
    if value < lo:
        return f'<div class="ood-chip">⚠ OUT OF ENVELOPE — below {lo:g}{unit}</div>'
    if value > hi:
        return f'<div class="ood-chip">⚠ OUT OF ENVELOPE — above {hi:g}{unit}</div>'
    return f'<div class="ok-chip">in envelope ({lo:g}–{hi:g}{unit})</div>'


@st.cache_resource
def get_ensemble():
    return load_ensemble()


@st.cache_data(show_spinner=False)
def cached_predict(R_n_mm: float, theta_c: float, mach: float, altitude: float):
    """Cache by rounded slider values so re-selecting a field is instant."""
    pred = predict(get_ensemble(), R_n=R_n_mm / 1000.0, theta_c_deg=theta_c,
                   mach=mach, altitude_km=altitude)
    # return a plain dict so Streamlit can hash/cache it
    return {
        "coords": pred.coords, "fields": pred.fields, "n_nodes": pred.n_nodes,
        "spread": pred.spread, "distance": pred.distance, "kn": pred.kn,
        "decision": pred.decision, "qoi": pred.qoi, "params": pred.params,
    }


# ============================================================================================
#                                       plotting
# ============================================================================================

def field_figure(coords: np.ndarray, values: np.ndarray, name: str, R_b: float) -> plt.Figure:
    label, unit, log = FIELD_META[name]
    x, r = coords[:, 0], coords[:, 1]
    # mirror the upper-half solution across the axis for a full-body view
    xx = np.concatenate([x, x])
    rr = np.concatenate([r, -r])
    vv = np.concatenate([values, values])
    tri = Triangulation(xx, rr)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if log:
        vmin = max(vmin, vmax * 1e-4)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        levels = np.geomspace(vmin, vmax, 60)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, 60)
    cf = ax.tricontourf(tri, np.clip(vv, vmin, vmax), levels=levels, norm=norm, cmap="inferno")
    cb = fig.colorbar(cf, ax=ax, shrink=0.9)
    cb.set_label(f"{label} [{unit}]")
    ax.set_aspect("equal")
    # crop to the near-body region: the domain extends ~8 R_b into the far-field,
    # so the full frame is mostly undisturbed freestream. x_base is the foremost
    # extent of the mesh downstream; frame a few base-radii around the forebody.
    x_base = float(x.max())
    ax.set_xlim(x_base - 1.15 * R_b, x_base + 0.05 * R_b)
    ax.set_ylim(-1.2 * R_b, 1.2 * R_b)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.set_title(f"{label} (ensemble mean)")
    fig.tight_layout()
    return fig


# ============================================================================================
#                                       sidebar controls
# ============================================================================================

st.markdown(OOD_CSS, unsafe_allow_html=True)
st.sidebar.title("Flight condition")
st.sidebar.caption("Each slider's green chip turns to a flashing red warning the moment "
                   "the value leaves the trained envelope. Ranges extend past the box so "
                   "you can probe extrapolation.")

mach = st.sidebar.slider("Mach number", 8.0, 30.0, 15.0, 0.5)
st.sidebar.markdown(envelope_chip(mach, *ENVELOPE_BOX["mach"]), unsafe_allow_html=True)
altitude = st.sidebar.slider("Altitude [km]", 40.0, 70.0, 52.0, 1.0)
st.sidebar.markdown(envelope_chip(altitude, *ENVELOPE_BOX["altitude"], unit=" km"), unsafe_allow_html=True)
R_n_mm = st.sidebar.slider("Nose radius R_n [mm]", 5.0, 80.0, 25.0, 1.0)
st.sidebar.markdown(envelope_chip(R_n_mm, *ENVELOPE_BOX["R_n_mm"], unit=" mm"), unsafe_allow_html=True)
theta_c = st.sidebar.slider("Cone half-angle θ_c [deg]", 35.0, 80.0, 50.0, 1.0)
st.sidebar.markdown(envelope_chip(theta_c, *ENVELOPE_BOX["theta_c"], unit="°"), unsafe_allow_html=True)

_ood = [n for n, v, k in (("Mach", mach, "mach"), ("altitude", altitude, "altitude"),
                          ("R_n", R_n_mm, "R_n_mm"), ("θ_c", theta_c, "theta_c"))
        if v < ENVELOPE_BOX[k][0] or v > ENVELOPE_BOX[k][1]]
if _ood:
    st.sidebar.warning(f"Extrapolating on: {', '.join(_ood)}. The prediction may be "
                       f"warn- or refuse-flagged.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fixed: base radius R_b = {R_B_RATIO:g}·R_n, shoulder R_s = {R_S_RATIO:g}·R_b, "
    f"cold wall T_w = {T_WALL:g} K."
)
run = st.sidebar.button("Run prediction", type="primary", use_container_width=True)


# ============================================================================================
#                                       main panel
# ============================================================================================

st.title("Hypersonic sphere-cone flow surrogate")
st.markdown(
    "A Transolver deep ensemble predicts the steady laminar flow field around an "
    "axisymmetric re-entry capsule, trained on SU2 axisymmetric Navier-Stokes ground "
    "truth. Move the sliders, run a prediction, and read the uncertainty verdict. "
    "One prediction meshes the body and runs five members on CPU: expect under a minute."
)

if run:
    with st.spinner("Meshing geometry and running the five-member ensemble on CPU..."):
        st.session_state["pred"] = cached_predict(round(R_n_mm, 1), round(theta_c, 1),
                                                   round(mach, 1), round(altitude, 1))

pred = st.session_state.get("pred")
if pred is None:
    st.info("Set a flight condition in the sidebar and press **Run prediction**.")
    st.stop()

# decision banner
label, color, blurb = DECISION_STYLE[pred["decision"]]
st.markdown(
    f"<div style='padding:0.8rem 1rem;border-radius:0.5rem;background:{color}22;"
    f"border-left:6px solid {color};'>"
    f"<span style='font-size:1.3rem;font-weight:700;color:{color}'>{label}</span>"
    f"<br><span style='color:inherit'>{blurb}</span></div>",
    unsafe_allow_html=True,
)

g1, g2, g3, g4 = st.columns(4)
g1.metric("Envelope distance", f"{pred['distance']:.3f}", help=f"L-inf box exceedance; guard at {GUARD_DIST}")
g2.metric("Knudsen number", f"{pred['kn']:.4f}", help=f"continuum floor {KN_MAX}")
g3.metric("Ensemble spread", f"{pred['spread']:.3f}", help=f"warn at {WARN_SPREAD}")
g4.metric("Mesh nodes", f"{pred['n_nodes']:,}")

st.markdown("---")
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Flow field")
    field = st.selectbox("Field", list(FIELD_META.keys()),
                         format_func=lambda k: f"{FIELD_META[k][0]} ({k})")
    st.pyplot(field_figure(pred["coords"], pred["fields"][field], field, pred["params"]["R_b"]))

with right:
    st.subheader("Quantities of interest")
    so = pred["qoi"]["standoff"]
    st.metric("Shock standoff (predicted)", f"{so['mean']*1e3:.2f} mm",
              delta=f"± {so['std']*1e3:.2f} mm ensemble band", delta_color="off")
    st.caption(f"Billig correlation reference: {so['reference']*1e3:.2f} mm "
               f"(expected agreement ~20%).")

    qw = pred["qoi"]["q_w"]
    st.metric("Stagnation heat flux (Fay-Riddell)", f"{qw['reference']/1e6:.2f} MW/m²")
    st.caption("Analytic cold-wall value. The surrogate does not resolve the thin "
               "near-wall thermal gradient, so its direct heat-flux estimate is "
               "unreliable and is not shown.")

    st.markdown("---")
    st.subheader("Need a trusted answer?")
    if st.button("Request full SU2 simulation", use_container_width=True):
        p = pred["params"]
        st.info(
            "Placeholder only: this demo does not run SU2 and nothing is submitted "
            "anywhere. In a deployed system this button would hand off the case below "
            "to the offline CFD pipeline (an SU2 axisymmetric Navier-Stokes run), which "
            "is the trusted fallback when a case is refuse-flagged or high-stakes. "
            "The surrogate above is the instant estimate.\n\n"
            f"Case: R_n={p['R_n']*1e3:.1f} mm, θ_c={p['theta_c_deg']:.1f}°, M={p['mach']:.1f}, "
            f"T∞={p['T_inf']:.0f} K, p∞={p['p_inf']:.1f} Pa, T_w={p['T_w']:.0f} K."
        )
