"""Streamlit dashboard for the hypersonic sphere-cone flow surrogate.

Four sliders drive a deep-ensemble prediction of the steady flow field around
an axisymmetric re-entry capsule. The page reports the field, a verdict of
trust, warn or refuse that reads the ensemble uncertainty, and two analytical
cross-checks. One prediction meshes the geometry and runs five Transolver
members on the processor, which takes tens of seconds, so the run button holds
it behind a spinner.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from inference import R_B_RATIO, R_S_RATIO, load_default_ensemble, predict
from matplotlib.colors import LogNorm, Normalize
from matplotlib.tri import Triangulation

from src.data.sampler import FS_BOX, GEOM_BOX, KN_MAX, T_WALL
from src.eval.uncertainty import GUARD_DIST

# ============================================================================================
#                                       page config
# ============================================================================================

st.set_page_config(page_title="Hypersonic sphere-cone surrogate", layout="wide")

FIELD_META = {
    "T": ("Temperature", "K", False),
    "rho": ("Density", "kg/m³", True),
    "p": ("Pressure", "Pa", True),
    "u": ("Axial velocity", "m/s", False),
    "v": ("Radial velocity", "m/s", False),
}
DECISION_STYLE = {
    "trust": ("Trust", "#1a7f37", "Inputs sit inside the validated envelope and the "
              "ensemble agrees. Prediction is usable."),
    "warn": ("Warn", "#9a6700", "Ensemble spread is elevated. Treat the field as "
             "indicative, not quantitative."),
    "refuse": ("Refuse", "#cf222e", "Inputs fall outside the validated envelope or the "
               "ensemble disagrees strongly. Run a full simulation instead."),
}

# the training box per exposed control, shown under each slider
ENVELOPE_BOX = {
    "mach": FS_BOX["mach"],
    "altitude": FS_BOX["altitude"],
    "R_n_mm": tuple(1000.0 * v for v in GEOM_BOX["R_n"]),
    "theta_c": GEOM_BOX["theta_c"],
}


def envelope_note(value: float, lo: float, hi: float, unit: str = "") -> str:
    """Say whether the value sits inside the training box."""
    if value < lo:
        return f":red[outside the training box, below {lo:g}{unit}]"
    if value > hi:
        return f":red[outside the training box, above {hi:g}{unit}]"
    return f"inside the training box ({lo:g} to {hi:g}{unit})"


@st.cache_resource
def get_ensemble():
    return load_default_ensemble()


@st.cache_data(show_spinner=False)
def cached_predict(R_n_mm: float, theta_c: float, mach: float, altitude: float):
    """Cache on the rounded slider values, so a change of field costs nothing."""
    ens, thresholds = get_ensemble()
    pred = predict(ens, thresholds, R_n=R_n_mm / 1000.0, theta_c_deg=theta_c,
                   mach=mach, altitude_km=altitude)
    # a plain dict, because Streamlit hashes the return value to cache it
    return {
        "coords": pred.coords, "fields": pred.fields, "n_nodes": len(pred.coords),
        "spread": pred.spread, "distance": pred.distance, "kn": pred.kn,
        "decision": pred.decision, "qoi": pred.qoi, "params": pred.params,
    }


# ============================================================================================
#                                       plotting
# ============================================================================================

def field_figure(coords: np.ndarray, values: np.ndarray, name: str, R_b: float) -> plt.Figure:
    label, unit, log = FIELD_META[name]
    x, r = coords[:, 0], coords[:, 1]
    # mirror the upper half across the axis, so the figure shows the whole body
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
    # crop to the region near the body. the domain reaches about 8 R_b into the
    # far field, so most of a full frame would be undisturbed freestream. x_base
    # is the downstream end of the mesh, and the window holds a few base radii
    # around the forebody.
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

st.sidebar.title("Flight condition")
st.sidebar.caption("The sliders reach past the training box, so you can see how "
                   "the surrogate behaves outside it.")

mach = st.sidebar.slider("Mach number", 8.0, 30.0, 15.0, 0.5)
st.sidebar.caption(envelope_note(mach, *ENVELOPE_BOX["mach"]))
altitude = st.sidebar.slider("Altitude [km]", 40.0, 70.0, 52.0, 1.0)
st.sidebar.caption(envelope_note(altitude, *ENVELOPE_BOX["altitude"], unit=" km"))
R_n_mm = st.sidebar.slider("Nose radius R_n [mm]", 5.0, 80.0, 25.0, 1.0)
st.sidebar.caption(envelope_note(R_n_mm, *ENVELOPE_BOX["R_n_mm"], unit=" mm"))
theta_c = st.sidebar.slider("Cone half-angle θ_c [deg]", 35.0, 80.0, 50.0, 1.0)
st.sidebar.caption(envelope_note(theta_c, *ENVELOPE_BOX["theta_c"], unit="°"))

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
    "One prediction meshes the body and runs five members on the processor, "
    "which takes under a minute."
)

if run:
    with st.spinner("Meshing the geometry and running the five-member ensemble..."):
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
g3.metric("Ensemble spread", f"{pred['spread']:.3f}", help=f"warn at {get_ensemble()[1]['warn']:.3f}")
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

