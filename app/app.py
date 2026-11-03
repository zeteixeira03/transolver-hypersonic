"""Streamlit dashboard for the hypersonic sphere-cone flow surrogate.

One prediction meshes the geometry and runs five Transolver members on the
processor, so it waits for the run button. The first visit runs the default
condition, so the page opens with a result.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from inference import (
    R_B_RATIO,
    R_S_RATIO,
    REGIONS,
    SLICE_HEAD,
    SLICE_LAYER,
    load_default_ensemble,
    normalized_mutual_information,
    predict,
    slice_table,
)
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.data.sampler import FS_BOX, GEOM_BOX, KN_MAX, T_WALL
from src.eval.plots import decade_ticks, masked_triangulation
from src.eval.uncertainty import GUARD_DIST
from src.geometry.sphere_cone import body_outline

# ============================================================================================
#                                       page config
# ============================================================================================

st.set_page_config(page_title="Hypersonic re-entry surrogate", layout="wide")

# label, unit, log scale, colour map, symbol on the field selector
FIELD_META = {
    "T": ("temperature", "K", True, "inferno", "T"),
    "rho": ("density", "kg/m³", True, "viridis", "ρ"),
    "p": ("pressure", "Pa", True, "plasma", "p"),
    "u": ("axial velocity", "m/s", False, "cividis", "u"),
    "v": ("radial velocity", "m/s", False, "RdBu_r", "v"),
}
# the colours read on both the light and the dark theme
VERDICT = {"trust": ("Trust", "usable", "#3d9a63"),
           "warn": ("Warn", "indicative only", "#c98a1b"),
           "refuse": ("Refuse", "run a full simulation", "#e5484d")}
OUT_OF_BOX = "#e5484d"
TRACK = "rgba(151, 166, 195, 0.25)"
LINE = "rgba(128, 128, 128, 0.25)"
INK, MUTED = "#1f2430", "#6b7280"
REGION_GREYS = ["#d7dce4", "#98a3b6", "#4a5568"]
SLICE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
                "#b07aa1", "#ff9da7", "#9c755f", "#86bcb6"]

ENVELOPE_BOX = {
    "mach": FS_BOX["mach"],
    "altitude": FS_BOX["altitude"],
    "R_n_mm": tuple(1000.0 * v for v in GEOM_BOX["R_n"]),
    "theta_c": GEOM_BOX["theta_c"],
}
# key, label, slider range, default, step, value format
SLIDERS = [
    ("mach", "Mach", 8.0, 30.0, 15.0, 0.5, "%.1f"),
    ("altitude", "Altitude", 40.0, 70.0, 52.0, 1.0, "%.0f km"),
    ("R_n_mm", "Nose radius $R_n$", 5.0, 80.0, 25.0, 1.0, "%.0f mm"),
    ("theta_c", r"Cone half-angle $\theta_c$", 35.0, 80.0, 50.0, 1.0, "%.0f°"),
]

CSS = f"""<style>
h1 {{ padding-bottom: 0.25rem; }}
.eyebrow {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.65; }}
.muted {{ opacity: 0.65; }}
[data-testid="stSliderTickBar"] {{ display: none; }}
/* the dark theme's accent is amber, too light for white text */
@media (prefers-color-scheme: dark) {{ [data-testid="stBaseButton-primary"] p {{ color: #0a1020; }} }}
.box {{ position: relative; height: 16px; margin-top: -2.3rem; }}
.box div {{ position: absolute; top: 0; height: 5px; border: 1px solid currentColor; border-top: 0; opacity: 0.5;
  left: calc(6px + var(--lo) * (100% - 12px)); width: calc((var(--hi) - var(--lo)) * (100% - 12px)); }}
.box span {{ position: absolute; top: 4px; font-size: 11px; line-height: 1; opacity: 0.6; transform: translateX(-50%);
  font-variant-numeric: tabular-nums; }}
.st-key-strip {{ border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 0.5rem 0; }}
.st-key-strip [data-testid="stColumn"] + [data-testid="stColumn"] {{ border-left: 1px solid {LINE}; padding-left: 1rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
.verdict-k {{ font-size: 0.875rem; }}
.chip {{ display: inline-flex; align-items: center; gap: 0.5rem; font-size: 1.25rem; font-weight: 700; margin-top: 0.3rem; }}
.chip i {{ width: 10px; height: 10px; border-radius: 50%; background: currentColor; }}
.race {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.6rem 2rem; align-items: center; }}
.race-rows {{ display: grid; grid-template-columns: 6rem minmax(0, 1fr) 6rem; gap: 0.6rem 0.75rem; align-items: center; }}
.race-rows .bar {{ height: 12px; border-radius: 3px; background: currentColor; opacity: 0.35; }}
.race-rows .bar.fast {{ opacity: 1; min-width: 3px; }}
.race-rows .t {{ font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
.factor {{ text-align: right; }}
.factor b {{ display: block; font-size: 2.1rem; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.st-key-field_plate, .st-key-slice_plate {{ background: #ffffff; border: 1px solid {LINE}; border-radius: 8px; padding: 12px; }}
table.slices {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
.slices th {{ text-align: left; font-weight: 400; opacity: 0.65; padding: 0 0.5rem 0.4rem 0; border-bottom: 1px solid {LINE}; }}
.slices td {{ padding: 0.35rem 0.5rem 0.35rem 0; border-bottom: 1px solid {LINE}; font-variant-numeric: tabular-nums; }}
.slices .r {{ text-align: right; padding-right: 0; }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; margin-right: 0.5rem; }}
</style>"""


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
        "mesh_s": pred.mesh_s, "infer_s": pred.infer_s, "slices": pred.slices,
        "regions": pred.regions, "su2": pred.su2,
    }


# ============================================================================================
#                                       plotting
# ============================================================================================

def bare_axes(ax: plt.Axes) -> None:
    ax.axhline(0, color="white", lw=1.0, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def field_figure(coords: np.ndarray, values: np.ndarray, name: str,
                 params: dict[str, float]) -> plt.Figure:
    """The field above the axis and its mirror image below, in millimetres."""
    label, unit, log, cmap, _ = FIELD_META[name]
    x, r = coords[:, 0] * 1e3, coords[:, 1] * 1e3
    outline = body_outline(params["R_n"], params["theta_c_deg"], params["R_b"], params["R_s"]) * 1e3
    # in the mirror image the radial component points the other way
    mirrored = -values if name == "v" else values
    if log:
        hi = float(values.max())
        norm = LogNorm(max(float(values.min()), hi * 1e-4), hi)
        levels = np.geomspace(norm.vmin, norm.vmax, 64)
    elif name == "v":
        # radial velocity changes sign across the axis, so zero sits in the middle of the map
        m = float(np.abs(values).max())
        norm = TwoSlopeNorm(0.0, -m, m)
        levels = np.linspace(-m, m, 64)
    else:
        norm = Normalize(float(values.min()), float(values.max()))
        levels = np.linspace(norm.vmin, norm.vmax, 64)
    kw = dict(levels=levels, norm=norm, cmap=cmap, extend="both")

    fig, ax = plt.subplots(figsize=(4.4, 5.6))
    cf = ax.tricontourf(masked_triangulation(x, r, outline),
                        np.clip(values, levels[0], levels[-1]), **kw)
    ax.tricontourf(masked_triangulation(x, -r, outline),
                   np.clip(mirrored, levels[0], levels[-1]), **kw)
    ax.fill(outline[:, 0], outline[:, 1], color="#d1d5db", zorder=3)
    bare_axes(ax)
    R_b, x_base = params["R_b"] * 1e3, float(x.max())
    ax.set_xlim(x_base - 1.25 * R_b, x_base + 0.02 * R_b)
    ax.set_ylim(-1.25 * R_b, 1.25 * R_b)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, fraction=0.05)
    if log:
        decade_ticks(cb)
    else:
        ticks = MaxNLocator(5).tick_values(levels[0], levels[-1])
        cb.set_ticks([t for t in ticks if levels[0] <= t <= levels[-1]])
        cb.formatter = FuncFormatter(lambda v, _: f"{v:g}")
    cb.set_label(f"{label} [{unit}]", color=INK)
    cb.ax.tick_params(colors=MUTED)
    cb.outline.set_edgecolor(MUTED)
    return fig


def slice_figure(coords: np.ndarray, slices: np.ndarray, regions: np.ndarray,
                 table: list[dict], params: dict[str, float]) -> plt.Figure:
    """Strongest slice per node above the axis, flow regions below, zoomed on the nose."""
    colour_of = {row["slice"]: i for i, row in enumerate(table)}
    idx = np.array([colour_of.get(m, -1) for m in slices], dtype=float)
    x, r = coords[:, 0] * 1e3, coords[:, 1] * 1e3
    outline = body_outline(params["R_n"], params["theta_c_deg"], params["R_b"], params["R_s"]) * 1e3
    cmap = ListedColormap(SLICE_COLORS[:len(table)])
    cmap.set_under("#c7cbd1")

    fig, ax = plt.subplots(figsize=(4.4, 5.4))
    ax.tripcolor(masked_triangulation(x, r, outline), idx, cmap=cmap, vmin=0, vmax=len(table) - 1e-9)
    ax.tripcolor(masked_triangulation(x, -r, outline), regions.astype(float),
                 cmap=ListedColormap(REGION_GREYS), vmin=0, vmax=len(REGIONS) - 1e-9)
    ax.fill(outline[:, 0], outline[:, 1], facecolor="#eef0f3", edgecolor=MUTED, lw=0.8, zorder=3)
    bare_axes(ax)
    # the shock layer and the wall layer are thin, so the view stays on the nose
    nose, rn = outline[:, 0].min(), params["R_n"] * 1e3
    ax.set_xlim(nose - 0.4 * rn, nose + 2.2 * rn)
    ax.set_ylim(-1.6 * rn, 1.6 * rn)
    tag = dict(transform=ax.transAxes, fontsize=9, color=INK,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    ax.text(0.03, 0.97, "slices", va="top", **tag)
    ax.text(0.03, 0.03, "flow regions", va="bottom", **tag)
    return fig


def png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# a figure takes seconds to triangulate and draw, so each one is drawn once per query
@st.cache_data(show_spinner=False)
def field_png(query: tuple[float, float, float, float], name: str) -> bytes:
    pred = cached_predict(*query)
    return png(field_figure(pred["coords"], pred["fields"][name], name, pred["params"]))


@st.cache_data(show_spinner=False)
def slice_png(query: tuple[float, float, float, float]) -> bytes:
    pred = cached_predict(*query)
    table = slice_table(pred["slices"], pred["regions"])
    return png(slice_figure(pred["coords"], pred["slices"], pred["regions"], table, pred["params"]))


# ============================================================================================
#                                       sidebar controls
# ============================================================================================

st.markdown(CSS, unsafe_allow_html=True)
st.sidebar.markdown('<div class="eyebrow">Flight condition</div>', unsafe_allow_html=True)

values = {}
for key, label, lo, hi, default, step, fmt in SLIDERS:
    v = st.sidebar.slider(label, lo, hi, default, step, format=fmt, key=f"sl_{key}")
    values[key] = v
    box_lo, box_hi = ENVELOPE_BOX[key]
    a, b = (box_lo - lo) / (hi - lo), (box_hi - lo) / (hi - lo)
    st.sidebar.markdown(
        f'<div class="box" style="--lo:{a};--hi:{b}" title="training box"><div></div>'
        f'<span style="left:calc(6px + {a} * (100% - 12px))">{box_lo:g}</span>'
        f'<span style="left:calc(6px + {b} * (100% - 12px))">{box_hi:g}</span></div>',
        unsafe_allow_html=True)
    if v < box_lo or v > box_hi:
        p = 100 * (v - lo) / (hi - lo)
        # the thumb is the track's only child positioned by a translate
        track = f".st-key-sl_{key} [data-testid='stSlider'] > div > div > div"
        st.markdown(
            f"<style>{track}[style*='translate'] {{ background-color: {OUT_OF_BOX} !important; }}"
            f".st-key-sl_{key} [data-testid='stSliderThumbValue'] {{ color: {OUT_OF_BOX} !important; }}"
            f"{track}:not([style*='translate']):not([data-testid]) {{ "
            f"background-image: linear-gradient(to right, {OUT_OF_BOX} 0%, {OUT_OF_BOX} {p}%, "
            f"{TRACK} {p}%, {TRACK} 100%) !important; }}</style>",
            unsafe_allow_html=True)

st.sidebar.caption(rf"$R_b = {R_B_RATIO:g}\,R_n$, $R_s = {R_S_RATIO:g}\,R_b$, $T_w = {T_WALL:g}$ K")
run = st.sidebar.button("Run prediction", type="primary", width="stretch")


# ============================================================================================
#                                       main panel
# ============================================================================================

st.title("Neural surrogate for hypersonic atmospheric re-entry")
st.markdown(
    "An attention-based neural surrogate (Transolver) that predicts the steady laminar air flow "
    "around an axisymmetric re-entry capsule. It was trained on SU2 Navier-Stokes simulation data.\n\n"
    "Move the sliders on the left of your screen and watch how long it takes to predict a simulation result."
)

if run or "query" not in st.session_state:
    with st.spinner("Meshing the geometry and running the five-member ensemble..."):
        query = (round(values["R_n_mm"], 1), round(values["theta_c"], 1),
                 round(values["mach"], 1), round(values["altitude"], 1))
        cached_predict(*query)
        st.session_state["query"] = query
query = st.session_state["query"]
pred = cached_predict(*query)

label, note, color = VERDICT[pred["decision"]]
with st.container(key="strip"):
    c0, c1, c2, c3 = st.columns(4)
    c0.markdown(f'<div class="verdict-k">Verdict</div><div class="chip" style="color:{color}"><i></i>{label}</div>'
                f'<div class="muted" style="font-size:0.875rem">{note}</div>', unsafe_allow_html=True)
    c1.metric("Envelope distance", f"{pred['distance']:.3f}", help=(
        "largest excess of any input over the range the training cases span, as a fraction of "
        f"that range; refuse above {GUARD_DIST}"))
    c2.metric("Knudsen number", f"{pred['kn']:.4f}", help=f"the continuum equations stop applying above {KN_MAX}")
    c3.metric("Ensemble spread", f"{pred['spread']:.3f}", help=f"warn at {get_ensemble()[1]['warn']:.3f}")

st.subheader("Solve time")
su2 = pred["su2"]
fast, slow = pred["mesh_s"] + pred["infer_s"], su2["hours"] * 3600.0
hours, minutes = divmod(round(su2["hours"] * 60), 60)
st.markdown(
    f'<div class="race"><div class="race-rows">'
    f'<span>Surrogate</span><div><div class="bar fast" style="width:{100 * fast / slow:.3f}%"></div></div>'
    f'<span class="t">{fast:.1f} s</span>'
    f'<span>SU2</span><div><div class="bar"></div></div><span class="t">{hours} h {minutes:02d} min</span>'
    f'</div><div class="factor"><b>{round(slow / fast, -1):,.0f}×</b><span class="muted">faster</span></div></div>',
    unsafe_allow_html=True)
meshing = f", including {pred['mesh_s']:.0f} s of meshing" if pred["mesh_s"] >= 1 else ""
st.caption(
    f"Surrogate: five members on {pred['n_nodes']:,} nodes, on the processor{meshing}. "
    f"SU2: nearest solved case (Mach {su2['mach']:.1f}, $R_n$ {1e3 * su2['R_n']:.0f} mm, "
    rf"$\theta_c$ {su2['theta_c']:.0f}°), one solver process; median {su2['median_hours']:.1f} h "
    f"over all {su2['n_cases']} sweep cases.")

left, right = st.columns([3, 2], gap="large")
with left:
    head, pick = st.columns([1, 1], vertical_alignment="center")
    head.subheader("Field predictions")
    field = pick.segmented_control("Field", list(FIELD_META), default="T", label_visibility="collapsed",
                                   format_func=lambda k: FIELD_META[k][4]) or "T"
    with st.container(key="field_plate"):
        st.image(field_png(query, field), width="stretch")

with right:
    st.subheader("Quantities of interest")
    so = pred["qoi"]["standoff"]
    st.metric("Shock standoff", f"{so['mean'] * 1e3:.2f} ± {so['std'] * 1e3:.2f} mm")
    st.caption(f"Billig {so['reference'] * 1e3:.2f} mm")
    qw = pred["qoi"]["q_w"]
    st.metric("Stagnation heat flux", f"{qw['reference'] / 1e6:.2f} MW/m²",
              help="analytical value; the surrogate does not resolve the near-wall thermal gradient")
    st.caption("Fay-Riddell, cold wall")

table = slice_table(pred["slices"], pred["regions"])
ens = get_ensemble()[0]
model = ens.models[0]
n_slices = model.blocks[SLICE_LAYER].attn.in_project_slice.out_features
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Learned slices")
    agreement = normalized_mutual_information(pred["regions"], pred["slices"])
    st.caption(f"layer {SLICE_LAYER + 1}/{len(model.blocks)} · head {SLICE_HEAD + 1}/"
               f"{model.blocks[SLICE_LAYER].attn.heads} · member 1/{len(ens.models)} · "
               f"agreement with the flow regions {agreement:.2f}")
    with st.container(key="slice_plate"):
        st.image(slice_png(query), width="stretch")

with right:
    st.caption(
        f"Transolver assigns every mesh node softly to {n_slices} slices and lets the slices, not the "
        f"{pred['n_nodes']:,} nodes, attend to each other. Above the axis each node takes the colour of "
        "its strongest slice; below it, the flow region it falls in. Agreement is the normalized mutual "
        "information of the two maps: 1 is a perfect match.")
    rows = "".join(
        f'<tr><td><span class="sw" style="background:{SLICE_COLORS[i]}"></span>{row["slice"] + 1}</td>'
        f'<td>{row["region"]}</td><td class="r">{row["share"]:.0%}</td><td class="r">{row["match"]:.0%}</td></tr>'
        for i, row in enumerate(table))
    regions = " ".join(f'<span class="sw" style="background:{c}"></span>{n}&emsp;'
                       for n, c in zip(REGIONS, REGION_GREYS))
    st.markdown(
        '<table class="slices"><thead><tr><th>Slice</th><th>Region</th><th class="r">Nodes</th>'
        f'<th class="r">Match</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    st.caption(
        f"Other slices: {1 - sum(row['share'] for row in table):.0%} of nodes, drawn grey. Match is the share "
        "of a slice's nodes inside the region named. Freestream is below 1.5 times the freestream "
        "temperature, near wall within 0.1 $R_n$ of the body, shock layer the rest.")
    st.markdown(f'<div style="font-size:0.875rem">{regions}</div>', unsafe_allow_html=True)
