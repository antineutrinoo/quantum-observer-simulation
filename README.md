# Quantum Observer Simulator

An interactive Python simulator that investigates the **Quantum Zeno effect** and the **Quantum Anti-Zeno effect**: how the interval between measurements changes survival and transition probabilities. Built with NumPy, SciPy, Plotly and Streamlit, with an automated pytest suite.

> This is a numerical model. It does not perform quantum measurements and does not reproduce a laboratory experiment. It is not deployed online; you run it locally.

## 1. Project overview

The simulator contains two clearly separated models:

* **Zeno page:** a resonantly driven two-level system with ideal repeated projective measurements.
* **Anti-Zeno page:** an unstable level decaying into a Gaussian reservoir, using the Kofman-Kurizki short-time approximation, cross-checked against an exact finite-reservoir simulation.

## 2. Problem statement

Does measuring a quantum system more often keep it in its initial state, and can measurement ever speed up a transition? Popular accounts say "watching prevents change". The truth depends on the system, the measurement interval and the assumptions.

## 3. Objectives

1. Implement an unmeasured reference (Rabi oscillation) and a repeated-measurement model.
2. Show Zeno suppression, and show where it does *not* occur.
3. Implement a scientifically defensible anti-Zeno model, not a reversed Zeno graph.
4. Let the user change parameters, compare conditions and export results.
5. Validate every calculation with independent checks.

## 4. Physics background

A two-level system $H=\tfrac{\Omega}{2}\sigma_x$ starting in $|0\rangle$ has survival $P_0=\cos^2(\Omega t/2)$ and transition $P_1=\sin^2(\Omega t/2)$. Early on, $P_1\propto t^2$. An ideal projective measurement resets the state, so measuring every $\tau$ keeps $P_1$ near zero if $\Omega\tau\ll1$: the Zeno effect (Misra and Sudarshan, 1977). In a decay into a continuum, measurements broaden the energy window used by the decay; if more strongly coupled states lie in that window the decay rate rises: the anti-Zeno effect (Kofman and Kurizki, 2000). The supplied presentation also discusses experiments with beryllium ions and repeated measurement pulses (Itano et al., 1990).

## 5. Features

* Five pages: Overview, Quantum Zeno effect, Quantum anti-Zeno effect, Mathematical model, Interpretation & limitations.
* Controls: Rabi frequency, total time, number of measurements or interval, selective/ensemble measurement type, time resolution, band detuning, width, coupling, comparison intervals, sweep range.
* Graphs: survival vs time, transition vs time, measured minus unmeasured, final survival vs N or τ, multi-condition comparison, spectral-overlap picture, decay rate vs τ (with exact check), survival with measurements, rate comparison bars.
* Results summary with final survival, final transition, N, τ, difference from baseline, R/Γ.
* Automatic interpretation computed from the numbers (it reports "lower than baseline" when that is what happens).
* CSV export of time series, sweeps and comparison tables.
* 166 automated tests.

## 6. Mathematical models used

Full derivation: [`docs/mathematical_model.md`](docs/mathematical_model.md).

$$P_0^{\text{sel}}(N)=\left[\cos^2\tfrac{\Omega\tau}{2}\right]^N,\qquad p_0^{\text{ens}}(N)=\tfrac12\left[1+\cos^N(\Omega\tau)\right],\qquad \tau=T/N$$

$$R(\tau)=2\pi\int G(\omega)\,F_\tau(\omega-\omega_a)\,d\omega,\quad F_\tau(x)=\tfrac{\tau}{2\pi}\,\mathrm{sinc}^2\tfrac{x\tau}{2},\quad S(t)=e^{-R(\tau)t},\quad \Gamma=2\pi G(\omega_a)$$

Regime: $R/\Gamma<1$ Zeno, $R/\Gamma>1$ anti-Zeno.

## 7. Technology stack

Python 3.10 to 3.13, NumPy, SciPy (ODE solver, `expm`, quadrature, `eigh`), pandas, Plotly, Streamlit, pytest.

## 8. Repository structure

```
quantum-observer/
├── app.py                      Streamlit interface (widgets and layout only)
├── requirements.txt            Python dependencies
├── pytest.ini                  Test configuration
├── README.md
├── LICENSE                     MIT
├── .gitignore
├── .streamlit/config.toml      Theme colour, telemetry off
├── src/
│   ├── __init__.py
│   ├── models.py               Formulas: Rabi, Zeno, Gaussian-band decay rate
│   ├── simulation.py           Time series, sweeps, ODE/matrix checks, exact reservoir
│   ├── analysis.py             Tables, regime classification, interpretation text
│   └── plots.py                Plotly figure builders
├── tests/
│   ├── test_models.py
│   ├── test_simulation.py
│   ├── test_analysis.py
│   └── test_app_smoke.py
└── docs/
    ├── mathematical_model.md
    ├── methodology.md
    ├── validation.md
    ├── execution_guide_windows.md
    ├── github_publishing.md
    ├── code_explanation.md
    └── demonstration_plan.md
```

There is no `assets/` or `sample_data/` folder because the application needs neither.

**What GitHub stores vs what runs.** GitHub stores these files and their history. Your computer runs the app: `app.py` imports `src/`, `src/plots.py` builds the graphs, Streamlit serves the page at localhost.

## 9. Installation (Windows)

From the project folder in the VS Code terminal (PowerShell):

```powershell
python --version
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `venv\Scripts\activate.bat` in Command Prompt. Step-by-step with troubleshooting: [`docs/execution_guide_windows.md`](docs/execution_guide_windows.md).

## 10. Running the application

```powershell
streamlit run app.py
```

Open http://localhost:8501 if the browser does not open itself. Stop with Ctrl+C.

## 11. Running the tests

```powershell
python -m pytest
```

## 12. Example workflow

1. Open **Quantum Zeno effect**. Defaults: Ω = π, T = 1 (one π-pulse), N = 8, selective. Final survival 0.7331 against an unmeasured baseline of 0.
2. Raise N to 32 and 64: survival 0.926 and 0.962.
3. In comparison mode read the generated interpretation.
4. Set T = 2, N = 3: survival 0.0156 against a baseline of 1. Measurement lowered survival, and the text says so.
5. Open **Quantum anti-Zeno effect** (Δ = −2, σ = 1, g = 0.2). τ = 0.2, 1, 5 give R/Γ = 0.58, 2.01, 1.33.
6. Set Δ = 0: the anti-Zeno behaviour disappears.

## 13. Reading the graphs

* **Survival / transition vs time:** dashed grey is the unmeasured oscillation; blue ticks mark measurements.
* **Measured − unmeasured:** above zero means measurement kept the system in |0⟩ more often.
* **Final survival vs N or τ:** more measurements (shorter τ) to the right on the N axis.
* **Spectral overlap:** grey area is the reservoir spectrum, coloured curves are the measurement filters (shape only). Overlap explains the rate.
* **Decay rate vs τ:** blue line is the approximation, orange dots the exact finite-reservoir result, dashed line is the unmeasured rate. Above the line is anti-Zeno, below is Zeno.

## 14. Assumptions and limitations

Ideal instantaneous projective measurements; equal spacing; closed two-level system (Zeno page); single excitation, weak coupling, rotating-wave approximation and a Gaussian band (anti-Zeno page). The anti-Zeno model illustrates the mechanism and is not a prediction for any specific atom. No experimental data is reproduced or fitted. Tested on Linux with Python 3.13 only. See [`docs/validation.md`](docs/validation.md).

## 15. Documentation

[Mathematical model](docs/mathematical_model.md) · [Methodology](docs/methodology.md) · [Validation](docs/validation.md) · [Windows guide](docs/execution_guide_windows.md) · [GitHub publishing](docs/github_publishing.md) · [Code explanation](docs/code_explanation.md) · [Demonstration plan](docs/demonstration_plan.md)

## 16. References

1. B. Misra and E. C. G. Sudarshan, "The Zeno's paradox in quantum theory," *J. Math. Phys.* **18**, 756 (1977). https://doi.org/10.1063/1.523304
2. W. M. Itano, D. J. Heinzen, J. J. Bollinger and D. J. Wineland, "Quantum Zeno effect," *Phys. Rev. A* **41**, 2295 (1990). https://doi.org/10.1103/PhysRevA.41.2295
3. A. G. Kofman and G. Kurizki, "Acceleration of quantum decay processes by frequent observations," *Nature* **405**, 546 (2000). https://doi.org/10.1038/35014537 (preprint: https://arxiv.org/abs/quant-ph/9912077)
4. M. C. Fischer, B. Gutiérrez-Medina and M. G. Raizen, "Observation of the quantum Zeno and anti-Zeno effects in an unstable system," *Phys. Rev. Lett.* **87**, 040402 (2001). https://doi.org/10.1103/PhysRevLett.87.040402 (preprint: https://arxiv.org/abs/quant-ph/0104035)
5. P. Facchi and S. Pascazio, "Quantum Zeno dynamics: mathematical and physical aspects," *J. Phys. A* **41**, 493001 (2008). https://doi.org/10.1088/1751-8113/41/49/493001 (preprint: https://arxiv.org/abs/0903.3297)

Reference check: web searches confirmed that papers with these titles and authors exist (the arXiv preprint links came from those searches). The DOI strings are given from standard citation records; the publisher pages were not opened, so confirm them before submitting. The model equations here are the author's implementation of standard results; they are not copied from these papers, and no numerical result of these papers is claimed to be reproduced.

## 17. Future improvements

* Non-equal measurement spacing and finite-duration measurements.
* Dephasing during the drive (a noisy two-level system).
* Other spectral densities (Lorentzian, power-law with cutoff).
* Fit to experimental data once the slides' numbers are available.
* Optional deployment to Streamlit Community Cloud.

## Licence

MIT. See [`LICENSE`](LICENSE).
