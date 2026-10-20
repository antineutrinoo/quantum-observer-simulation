"""Quantum Observer Simulator - Streamlit front end.

Run with:  streamlit run app.py
This file only handles widgets and layout. All physics lives in src/ so it can be tested without a browser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (
    REGIME_ANTI,
    antizeno_window,
    classify_change,
    dataframe_to_csv_bytes,
    decay_rate_table,
    interpret_decay_table,
    interpret_zeno_comparison,
    zeno_comparison_table,
    zeno_sweep_table,
)
from src.models import GaussianBand
from src.plots import (
    fig_decay_rate_sweep,
    fig_decay_survival,
    fig_final_survival_sweep,
    fig_measured_minus_unmeasured,
    fig_rate_bars,
    fig_spectral_overlap,
    fig_survival_vs_time,
    fig_transition_vs_time,
    fig_zeno_comparison,
)
from src.simulation import (
    ZenoParameters,
    baseline_final_survival,
    build_exact_bath,
    final_survival,
    log_spaced_counts,
    n_from_interval,
    simulate_decay_curves,
    simulate_zeno,
)

DOCS = Path(__file__).parent / "docs"

st.set_page_config(page_title="Quantum Observer Simulator", page_icon="⚛️", layout="wide")

PAGES = ["Overview", "Quantum Zeno effect", "Quantum anti-Zeno effect", "Mathematical model", "Interpretation & limitations"]


# --------------------------------------------------------------------------- cached heavy calls
@st.cache_resource(show_spinner="Diagonalising the discretised reservoir...")
def cached_bath(detuning: float, width: float, coupling: float, n_modes: int):
    return build_exact_bath(GaussianBand(detuning, width, coupling), n_modes=n_modes)


@st.cache_data(show_spinner=False)
def cached_decay_table(detuning: float, width: float, coupling: float, tau_min: float, tau_max: float, n_tau: int, with_exact: bool, n_modes: int):
    band = GaussianBand(detuning, width, coupling)
    taus = np.geomspace(tau_min, tau_max, n_tau)
    bath = cached_bath(detuning, width, coupling, n_modes) if with_exact else None
    return decay_rate_table(band, taus, bath)


def show_doc(name: str) -> None:
    path = DOCS / name
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning(f"{name} was not found in the docs/ folder.")


# =========================================================================== pages
def page_overview() -> None:
    st.title("⚛️ Quantum Observer Simulator")
    st.caption("An interactive investigation of the Quantum Zeno and Quantum Anti-Zeno effects. A numerical model, not a laboratory experiment.")
    st.subheader("Research question")
    st.markdown(
        "**How does the interval between measurements change the probability that a quantum system stays in (or leaves) its initial state?**"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**What 'measurement' means here.** A measurement is an interaction with an apparatus that forces the system into one of the
states being tested (here |0⟩ or |1⟩). It has nothing to do with a person looking. Each ideal measurement resets the
coherent evolution, so the *spacing* of measurements controls how much transition probability can build up.

**Quantum Zeno effect (QZE).** A driven two-level system leaves |0⟩ slowly at first (quadratically in time). Measure often enough
and it is repeatedly reset before it can leave: survival rises towards 1.
"""
        )
    with c2:
        st.markdown(
            """
**Quantum anti-Zeno effect (QAZE).** For an unstable level decaying into a reservoir, measurements broaden the energy window that
the decay can use. If more strongly coupled states lie inside the broadened window, the decay rate *increases*.
It needs a reservoir model, so it has its own page.

**Not universal.** Frequent measurement does not suppress every transition. Both pages are built to report what the numbers show.
"""
        )
    st.subheader("How to use it")
    st.markdown(
        "1. Choose a section in the sidebar. The sidebar holds the simulation controls for that section.\n"
        "2. Start with **Quantum Zeno effect**: compare N = 0 (no measurement) with larger N.\n"
        "3. Then open **Quantum anti-Zeno effect** and watch the effective decay rate versus the measurement interval.\n"
        "4. **Mathematical model** and **Interpretation & limitations** explain the equations and what the results do and do not prove."
    )
    st.info("Units are arbitrary: time in a.u., frequencies in rad per a.u. Only ratios (such as Ω·τ) have physical meaning.")


def page_zeno() -> None:
    st.title("Quantum Zeno effect: driven two-level system")
    st.caption("H = (Ω/2) σₓ, start in |0⟩, ideal projective measurements at equal intervals τ = T/N. Survival P₀ = probability of |0⟩; transition P₁ = 1 − P₀.")

    # ---- controls (sidebar)
    sb = st.sidebar
    sb.header("Simulation controls")
    omega = sb.slider("Rabi frequency Ω (rad / a.u.)", 0.1, 20.0, float(np.pi), 0.01, key="z_omega", help="Rate of the coherent |0⟩ ↔ |1⟩ oscillation. A full transfer takes T_π = π/Ω.")
    sb.caption(f"π-pulse time T_π = π/Ω = {np.pi / omega:.4g} a.u.")
    if "z_T" not in st.session_state:
        st.session_state["z_T"] = 1.0
    total_time = sb.slider("Total evolution time T (a.u.)", 0.05, 20.0, key="z_T", step=0.01, help="Measurements are placed at τ, 2τ, …, Nτ = T.")
    sb.button("Set T to one π-pulse", key="z_setpi", on_click=lambda: st.session_state.__setitem__("z_T", float(np.clip(round(np.pi / st.session_state["z_omega"], 2), 0.05, 20.0))), help="T = π/Ω, where the unmeasured system is fully transferred to |1⟩ (rounded to 0.01).")
    mode_label = sb.radio("Measurement type", ["Survival (selective)", "Ensemble (non-selective)"], key="z_mode", help="Selective: every measurement must find |0⟩, P = cos^(2N)(Ωτ/2). Ensemble: outcomes are not post-selected, p₀ = (1 + cos^N(Ωτ))/2.")
    mode = "selective" if mode_label.startswith("Survival") else "ensemble"
    by = sb.radio("Specify measurements by", ["Number N", "Interval τ"], key="z_by", horizontal=True)
    if by == "Number N":
        n_meas = sb.slider("Number of measurements N (0 = none)", 0, 256, 8, key="z_n")
    else:
        tau_req = sb.number_input("Requested interval τ (a.u.)", min_value=1e-4, max_value=100.0, value=round(total_time / 8, 4), step=0.01, format="%.4f", key="z_tau", help="N is rounded to a whole number, N = round(T/τ), so the effective interval is T/N.")
        n_meas = n_from_interval(total_time, tau_req, 100_000)
        sb.caption(f"→ N = {n_meas}, effective τ = {total_time / n_meas:.5g}")
    n_points = sb.slider("Time-grid points", 200, 5000, 1500, 100, key="z_points", help="Resolution of the curves (raised automatically for large N).")
    n_max = sb.slider("Largest N in the sweep", 4, 2048, 128, key="z_nmax")
    compare_n = sb.multiselect("Comparison-mode N values (0 = no measurement)", [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], default=[0, 2, 8, 32], key="z_compare")

    params = ZenoParameters(omega, total_time, n_meas, mode)
    ts = simulate_zeno(params, n_points)
    base = baseline_final_survival(omega, total_time)
    surv = final_survival(params)

    # ---- summary
    st.subheader("Results summary")
    c = st.columns(6)
    c[0].metric("Final survival P₀(T)", f"{surv:.5f}")
    c[1].metric("Final transition P₁(T)", f"{1 - surv:.5f}")
    c[2].metric("Measurements N", f"{n_meas}")
    c[3].metric("Interval τ", f"{params.tau:.5g}" if params.tau else "none")
    c[4].metric("Unmeasured baseline", f"{base:.5f}")
    c[5].metric("Difference vs baseline", f"{surv - base:+.5f}")
    if params.tau:
        st.caption(f"Ω·τ = {omega * params.tau:.3g}.  Zeno suppression needs Ω·τ well below 1.  Outcome for this setting: **{classify_change(surv, base)}**.")
    if total_time / (np.pi / omega) > 1.0001:
        st.caption(f"T is {total_time / (np.pi / omega):.3g} × T_π, so the unmeasured system has already passed |1⟩ and may be returning to |0⟩. Compare carefully.")

    # ---- time evolution graphs
    st.subheader("Evolution in time")
    a, b = st.columns(2)
    a.plotly_chart(fig_survival_vs_time(ts), width="stretch")
    b.plotly_chart(fig_transition_vs_time(ts), width="stretch")
    st.plotly_chart(fig_measured_minus_unmeasured(ts), width="stretch")
    st.caption("Dashed grey: unmeasured Rabi oscillation. Blue ticks: measurement times. In the selective model the transition curve is 1 − P₀, the probability that the 'always found in |0⟩' record has been broken.")

    # ---- sweep
    st.subheader("Effect of measurement frequency")
    n_values = log_spaced_counts(n_max, 40)
    sweep = zeno_sweep_table(omega, total_time, n_values, mode)
    x_axis = st.radio("x-axis", ["N", "tau"], horizontal=True, format_func=lambda v: "number of measurements N" if v == "N" else "measurement interval τ", key="z_xaxis")
    st.plotly_chart(fig_final_survival_sweep(sweep, base, x_axis, n_meas if n_meas > 0 else None), width="stretch")

    # ---- comparison mode
    st.subheader("Comparison mode")
    if compare_n:
        st.plotly_chart(fig_zeno_comparison(omega, total_time, compare_n, mode, n_points), width="stretch")
        table = zeno_comparison_table(omega, total_time, compare_n, mode)
        st.dataframe(table.style.format({"interval tau": "{:.5g}", "omega*tau": "{:.4g}", "final survival": "{:.5f}", "final transition": "{:.5f}", "baseline survival": "{:.5f}", "difference vs baseline": "{:+.5f}"}), width="stretch", hide_index=True)
        st.markdown("**Automatic interpretation (computed from the table above):**")
        for line in interpret_zeno_comparison(table, omega, total_time):
            st.markdown(f"- {line}")
        st.download_button("Download comparison table (CSV)", dataframe_to_csv_bytes(table), "zeno_comparison.csv", "text/csv")
    else:
        st.info("Pick at least one N value in the sidebar to use comparison mode.")

    d1, d2 = st.columns(2)
    d1.download_button("Download time series (CSV)", dataframe_to_csv_bytes(pd.DataFrame({"t": ts.t, "survival": ts.survival, "transition": ts.transition, "baseline_survival": ts.baseline_survival, "baseline_transition": ts.baseline_transition})), "zeno_time_series.csv", "text/csv")
    d2.download_button("Download sweep over N (CSV)", dataframe_to_csv_bytes(sweep), "zeno_sweep.csv", "text/csv")

    with st.expander("Equations and assumptions for this page"):
        st.latex(r"|\psi(t)\rangle=\cos\tfrac{\Omega t}{2}\,|0\rangle-i\sin\tfrac{\Omega t}{2}\,|1\rangle,\qquad P_0=\cos^2\tfrac{\Omega t}{2},\; P_1=\sin^2\tfrac{\Omega t}{2}")
        st.latex(r"\text{selective: } P_0(N)=\left[\cos^2\tfrac{\Omega\tau}{2}\right]^{N},\qquad \text{ensemble: } p_0(N)=\tfrac12\left[1+\cos^{N}(\Omega\tau)\right],\qquad \tau=T/N")
        st.markdown("Assumptions: closed two-level system, resonant drive, instantaneous ideal projective measurements in the {|0⟩, |1⟩} basis, no decoherence between measurements.")


def page_antizeno() -> None:
    st.title("Quantum anti-Zeno effect: decay into a reservoir")
    st.caption("Kofman-Kurizki approximation for an unstable level coupled weakly to a Gaussian reservoir band. Approximate model, not an exact prediction for any specific atom.")

    sb = st.sidebar
    sb.header("Simulation controls")
    delta = sb.slider("Detuning Δ = ω_a − ω_c (rad / a.u.)", -6.0, 6.0, -2.0, 0.1, key="a_delta", help="Level frequency minus band-centre frequency. Δ = 0 puts the level at the peak of the band.")
    sigma = sb.slider("Band width σ (rad / a.u.)", 0.2, 5.0, 1.0, 0.1, key="a_sigma", help="Standard deviation of the reservoir band. 1/σ is the reservoir memory time.")
    g = sb.slider("Coupling g (rad / a.u.)", 0.05, 0.6, 0.2, 0.01, key="a_g", help="Total coupling strength: ∫G(ω)dω = g². Keep the golden-rule rate Γ much smaller than σ.")
    band = GaussianBand(delta, sigma, g)
    gamma = band.golden_rule_rate
    sb.caption(f"Golden-rule rate Γ = {gamma:.4g}; lifetime 1/Γ = {1 / gamma:.4g}")
    if gamma > 0.2 * sigma:
        sb.warning("Γ is not ≪ σ: the weak-coupling approximation is unreliable here.")
    total_time = sb.slider("Total time for survival curves (a.u.)", 1.0, 1000.0, 150.0, 1.0, key="a_T")
    sb.markdown("**Measurement intervals to compare**")
    t1 = sb.number_input("τ₁ (a.u.)", 0.01, 500.0, 0.2, 0.05, key="a_t1", format="%.3f")
    t2 = sb.number_input("τ₂ (a.u.)", 0.01, 500.0, 1.0, 0.05, key="a_t2", format="%.3f")
    t3 = sb.number_input("τ₃ (a.u.)", 0.01, 500.0, 5.0, 0.05, key="a_t3", format="%.3f")
    tau_lo, tau_hi = sb.select_slider("Sweep range of τ (a.u.)", options=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200], value=(0.02, 100), key="a_range")
    n_tau = sb.slider("Sweep points", 20, 200, 80, 10, key="a_ntau")
    with_exact = sb.checkbox("Overlay exact finite-bath check", value=True, key="a_exact", help="Diagonalises a discretised reservoir exactly and compares the result with the approximation.")
    n_modes = sb.selectbox("Reservoir modes for the check", [401, 801, 1601], index=1, key="a_modes", help="More modes = longer trusted time range, slower startup.")

    taus = [t1, t2, t3]
    table_sweep = cached_decay_table(delta, sigma, g, float(tau_lo), float(tau_hi), int(n_tau), bool(with_exact), int(n_modes))
    table_choice = cached_decay_table_choice(delta, sigma, g, tuple(taus))
    bath = cached_bath(delta, sigma, g, int(n_modes)) if with_exact else None
    if bath is not None and tau_hi > bath.max_valid_time:
        st.warning(f"The exact check is only trusted for τ ≤ {bath.max_valid_time:.4g} with {n_modes} modes; larger τ are left out of the check.")

    st.subheader("Results summary")
    cols = st.columns(4)
    cols[0].metric("Golden-rule rate Γ", f"{gamma:.4g}")
    for col, (i, row) in zip(cols[1:], table_choice.iterrows()):
        col.metric(f"R(τ = {row['interval tau']:.3g})", f"{row['rate R(tau)']:.4g}", f"{row['R / Gamma']:.3f} × Γ  ·  {row['regime'].split(' (')[0]}", delta_color="off")
    win = antizeno_window(table_sweep)
    if win:
        st.caption(f"Anti-Zeno window in the sweep: τ from {win[0]:.3g} to {win[1]:.3g}, largest enhancement near τ = {win[2]:.3g}.")
    else:
        st.caption("No anti-Zeno interval in the swept range for these parameters.")

    st.subheader("Why it happens: spectral overlap")
    st.plotly_chart(fig_spectral_overlap(band, taus), width="stretch")
    st.caption("A measurement every τ acts like a filter of width about 2π/τ around the level. Short τ: very broad filter, but little total weight (Zeno). Medium τ: the filter reaches the band, which couples more strongly than the level itself (anti-Zeno). Long τ: the filter is narrow and gives back the golden-rule rate.")

    st.subheader("Effective decay rate versus measurement interval")
    st.plotly_chart(fig_decay_rate_sweep(table_sweep), width="stretch")

    st.subheader("Survival and rate comparison")
    curves = simulate_decay_curves(band, taus, total_time, 400)
    a, b = st.columns([3, 2])
    a.plotly_chart(fig_decay_survival(curves, bath), width="stretch")
    b.plotly_chart(fig_rate_bars(table_choice), width="stretch")

    st.markdown("**Automatic interpretation (computed from the numbers):**")
    for line in interpret_decay_table(table_choice, band):
        st.markdown(f"- {line}")

    st.dataframe(table_sweep.style.format("{:.5g}", subset=[c for c in table_sweep.columns if c != "regime"]), width="stretch", hide_index=True)
    st.download_button("Download decay-rate sweep (CSV)", dataframe_to_csv_bytes(table_sweep), "antizeno_rate_sweep.csv", "text/csv")

    with st.expander("Equations and assumptions for this page"):
        st.latex(r"G(\omega)=\frac{g^2}{\sqrt{2\pi}\,\sigma}\,e^{-(\omega-\omega_c)^2/2\sigma^2},\qquad \Gamma=2\pi\,G(\omega_a)")
        st.latex(r"R(\tau)=2\pi\int G(\omega)\,F_\tau(\omega-\omega_a)\,d\omega,\qquad F_\tau(x)=\frac{\tau}{2\pi}\,\mathrm{sinc}^2\!\left(\frac{x\tau}{2}\right),\qquad S(t)=e^{-R(\tau)\,t}")
        st.markdown("Assumptions: single excitation, weak coupling (Γ ≪ σ), rotating-wave approximation, band entirely at positive frequency, ideal projective measurement of the level, short-time (second-order) decay per interval then exponentiated.")


@st.cache_data(show_spinner=False)
def cached_decay_table_choice(detuning: float, width: float, coupling: float, taus: tuple[float, ...]):
    return decay_rate_table(GaussianBand(detuning, width, coupling), list(taus))


def page_math() -> None:
    st.title("Mathematical model")
    show_doc("mathematical_model.md")


def page_limits() -> None:
    st.title("Interpretation & limitations")
    show_doc("validation.md")
    st.divider()
    st.markdown(
        """
### How to read the results
* **Zeno page:** compare the measured curve with the dashed unmeasured one. The comparison text is generated from the table and will say so when measurements *lower* survival.
* **Anti-Zeno page:** only the ratio R(τ)/Γ matters. Below 1 is Zeno, above 1 is anti-Zeno, near 1 means the measurement changes little.

### Limits you should state when presenting
* This is a numerical model. It does not perform a quantum measurement or reproduce a laboratory experiment.
* The Zeno page uses ideal, instantaneous projective measurements on a closed two-level system.
* The anti-Zeno page is an approximation (weak coupling, Gaussian band, short-time rate). The exact finite-bath overlay checks it, but both share the same Gaussian-band idealisation.
* The two pages are two different physical models. Neither one is a fit to measured data.
"""
    )


# =========================================================================== navigation
st.sidebar.title("Quantum Observer")
page = st.sidebar.radio("Section", PAGES, key="page")
st.sidebar.divider()
{
    PAGES[0]: page_overview,
    PAGES[1]: page_zeno,
    PAGES[2]: page_antizeno,
    PAGES[3]: page_math,
    PAGES[4]: page_limits,
}[page]()
