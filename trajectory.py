"""
Trajectory simulation for RP-1 / 98% H2O2 rocket.

Integrates 1D vertical equations of motion from launch to target altitude
(or propellant exhaustion, whichever comes first).

Thruster inputs are read directly from a CEA study results.csv row —
no manual re-entry of performance numbers needed.

Usage:
    python trajectory.py

Outputs:
    - Console summary (burn time, propellant mass, peak altitude, burnout velocity)
    - trajectory_results.csv   — full time-history
    - plots saved to ./trajectory_results/
"""
from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ── Optional: use ambiance for standard atmosphere.
# If not installed, falls back to a built-in US Standard Atmosphere 1976 model.
try:
    from ambiance import Atmosphere
    _USE_AMBIANCE = True
except ImportError:
    _USE_AMBIANCE = False

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — the only things you should need to edit
# ════════════════════════════════════════════════════════════════════════════

# ── CEA run to load ───────────────────────────────────────────────────────────
RUN_ID          = "run_20260223_111121"     # folder name inside ./outputs/
RESULTS_CSV     = f"./outputs/{RUN_ID}/results.csv"

# ── Design point selector — pick the row you want ────────────────────────────
DESIGN_PC_BAR   = 5      # [bar]  must match a Pc_bar value in the CSV
DESIGN_OF       = 5.75       # [-]    must match an OF value in the CSV

# ── Mission ───────────────────────────────────────────────────────────────────
TARGET_ALTITUDE_M   = 1500  # [m]

# ── Vehicle ───────────────────────────────────────────────────────────────────
M_LAUNCH_KG         = 30.0      # [kg]  total wet mass at launch
M_PROP_REQ       = 5        # [-]   fraction that is dry structure (everything except propellant)
M_PROP_UNUSED = 0.2         # [kg] mass of propellant that gets stuck in pipes/tank
M_PROP_AVAILABLE = M_PROP_REQ + M_PROP_UNUSED- M_PROP_UNUSED

# ── Drag ──────────────────────────────────────────────────────────────────────
DRAG                = True
CD                  = 0.7       # [-]   drag coefficient
BODY_DIAMETER_M     = 0.3      # [m]   reference body diameter for frontal area

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
G0      = 9.80665       # [m/s²]
R_EARTH = 6_371_000.0   # [m]
P_SL_PA = 101_325.0     # [Pa]
FIGSIZE = (7, 4.5)
DPI     = 200


# ════════════════════════════════════════════════════════════════════════════
# LOAD DESIGN POINT FROM CEA RESULTS CSV
# ════════════════════════════════════════════════════════════════════════════

def load_design_point(csv_path: str, Pc_bar: float, OF: float) -> pd.Series:
    """
    Read the CEA results CSV and return the row matching (Pc_bar, OF).
    Raises a clear error if the point isn't found or is ambiguous.
    """
    df = pd.read_csv(csv_path)

    mask = (
        np.isclose(df["Pc_bar"].astype(float), Pc_bar, rtol=1e-4) &
        np.isclose(df["OF"].astype(float),     OF,     rtol=1e-4)
    )
    matches = df[mask]

    if len(matches) == 0:
        available_pc = sorted(df["Pc_bar"].unique())
        available_of = sorted(df["OF"].unique())
        raise ValueError(
            f"No row found for Pc_bar={Pc_bar}, OF={OF} in {csv_path}.\n"
            f"  Available Pc_bar values : {available_pc}\n"
            f"  Available OF values     : {[f'{v:.3f}' for v in available_of]}"
        )
    if len(matches) > 1:
        # In ae_at mode there may be multiple rows for the same (Pc, OF) with different ae/at.
        # Take the one whose ae_at most closely matches the pip-mode ideal expansion.
        # If only one nozzle mode is present, just take the first.
        matches = matches.iloc[[0]]
        print(f"  Warning: multiple rows matched (Pc={Pc_bar}, OF={OF}), using first.")

    row = matches.iloc[0]
    print(f"  Design point loaded:")
    print(f"    Pc = {row['Pc_bar']} bar,  O/F = {row['OF']}")
    print(f"    cstar = {row['cstar_m_s']:.1f} m/s,  Cf = {row['cf']:.4f}")
    print(f"    At = {float(row['At_m2'])*1e4:.4f} cm²,  Ae/At = {row['ae_at']:.3f}")
    print(f"    mdot = {row['mdot_kg_s']:.5f} kg/s,  F_req = {row['F_req_N']:.1f} N")
    print(f"    Isp(SL) = {row['isp_s']:.1f} s,  Ivac = {row['ivac_s']:.1f} s")
    return row


# ── Load at module level so all functions below can reference these globals ──
_dp             = load_design_point(RESULTS_CSV, DESIGN_PC_BAR, DESIGN_OF)

OF              = float(_dp["OF"])
Pc_Pa           = float(_dp["Pc_bar"]) * 1e5
Pe_Pa           = float(_dp["Pe_bar"]) * 1e5
At_m2           = float(_dp["At_m2"])
Ae_m2           = float(_dp["Ae_m2"])
mdot_kg_s       = float(_dp["mdot_kg_s"])
mdot_fuel       = float(_dp["mdot_fuel_kg_s"])
mdot_ox         = float(_dp["mdot_ox_kg_s"])
THRUST_N        = float(_dp["F_req_N"])         # sea-level design thrust
ISP_VAC_S       = float(_dp["ivac_s"])

# ── Derived vehicle quantities ────────────────────────────────────────────────
m_prop_kg       = M_PROP_AVAILABLE
m_dry_kg        = M_LAUNCH_KG - M_PROP_REQ
burn_time_s     = m_prop_kg / mdot_kg_s
A_ref_m2        = math.pi * (BODY_DIAMETER_M / 2.0) ** 2


# ════════════════════════════════════════════════════════════════════════════
# ATMOSPHERE MODEL
# ════════════════════════════════════════════════════════════════════════════

def ambient_pressure_pa(h_m: float) -> float:
    """Return ambient pressure [Pa] at geometric altitude h_m [m]."""
    h_m = max(0.0, h_m)
    if _USE_AMBIANCE:
        atm = Atmosphere(h_m)
        return np.asarray(atm.pressure).reshape(-1)[0].item()
    else:
        # US Standard Atmosphere 1976 — piecewise analytic
        if h_m <= 11_000.0:
            T = 288.15 - 0.0065 * h_m
            return P_SL_PA * (T / 288.15) ** 5.2561
        elif h_m <= 20_000.0:
            p11 = P_SL_PA * (216.65 / 288.15) ** 5.2561
            return p11 * math.exp(-G0 * (h_m - 11_000.0) / (287.058 * 216.65))
        elif h_m <= 32_000.0:
            p20 = ambient_pressure_pa(20_000.0)
            T = 216.65 + 0.001 * (h_m - 20_000.0)
            return p20 * (T / 216.65) ** -34.1632
        else:
            p32 = ambient_pressure_pa(32_000.0)
            return p32 * math.exp(-G0 * (h_m - 32_000.0) / (287.058 * 228.65))


def ambient_density_kg_m3(h_m: float) -> float:
    """Return air density [kg/m³] at altitude h_m [m] (ideal gas)."""
    h_m = max(0.0, h_m)
    if _USE_AMBIANCE:
        atm = Atmosphere(h_m)
        return np.asarray(atm.density).reshape(-1)[0].item()
    else:
        if h_m <= 11_000.0:
            T = 288.15 - 0.0065 * h_m
        elif h_m <= 20_000.0:
            T = 216.65
        elif h_m <= 32_000.0:
            T = 216.65 + 0.001 * (h_m - 20_000.0)
        else:
            T = 228.65
        p = ambient_pressure_pa(h_m)
        return p / (287.058 * T)

# ════════════════════════════════════════════════════════════════════════════
# THRUST MODEL
# ════════════════════════════════════════════════════════════════════════════

def thrust_n(h_m: float, burning: bool) -> float:
    """
    Delivered thrust at altitude h_m.

    F = (mdot * cstar_eff * cf_ideal_vac) + (Pe - Pamb) * Ae
      = F_vac + (Pe - Pamb) * Ae

    Equivalently written as the sea-level thrust corrected for ambient pressure:
      F(h) = F_sl + (P_sl - Pamb(h)) * Ae

    This is exact for a choked, ideally-expanded-at-SL nozzle.
    """
    if not burning:
        return 0.0
    P_amb = ambient_pressure_pa(h_m)
    F = THRUST_N + (P_SL_PA - P_amb) * Ae_m2
    return F


def isp_s(h_m: float) -> float:
    """Delivered Isp at altitude h_m [s]."""
    F = thrust_n(h_m, burning=True)
    return F / (mdot_kg_s * G0)


# ════════════════════════════════════════════════════════════════════════════
# GRAVITY MODEL
# ════════════════════════════════════════════════════════════════════════════

def gravity_m_s2(h_m: float) -> float:
    """Local gravitational acceleration [m/s²]."""
    return G0 * (R_EARTH / (R_EARTH + h_m)) ** 2


# ════════════════════════════════════════════════════════════════════════════
# ODE SYSTEM
# ════════════════════════════════════════════════════════════════════════════

def odes(t: float, y: list[float]) -> list[float]:
    """
    State vector y = [h, v, m]
      h : altitude [m]
      v : vertical velocity [m/s]
      m : vehicle mass [kg]

    Returns dy/dt.
    """
    h, v, m = y

    burning = (m > m_dry_kg + 1e-6)   # stop burning when propellant exhausted

    F   = thrust_n(h, burning)
    g   = gravity_m_s2(h)
    rho = ambient_density_kg_m3(h)

    # Drag — acts opposite to velocity direction
    if DRAG and abs(v) > 0.0:
        F_drag = 0.5 * rho * v * abs(v) * CD * A_ref_m2  # signed (opposes motion)
    else:
        F_drag = 0.0

    dm_dt = -mdot_kg_s if burning else 0.0
    dv_dt = (F - F_drag) / m - g
    dh_dt = v

    return [dh_dt, dv_dt, dm_dt]


# ════════════════════════════════════════════════════════════════════════════
# EVENTS
# ════════════════════════════════════════════════════════════════════════════

def event_target_altitude(t, y):
    return y[0] - TARGET_ALTITUDE_M
event_target_altitude.terminal  = True
event_target_altitude.direction = +1

def event_apogee(t, y):
    return y[1]   # velocity = 0 at apogee
event_apogee.terminal  = True
event_apogee.direction = -1

def event_ground(t, y):
    return y[0]   # h = 0
event_ground.terminal  = True
event_ground.direction = -1


# ════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_trajectory() -> pd.DataFrame:
    """
    Integrate the trajectory in two phases:
      Phase 1 — powered ascent (until propellant exhausted or target reached)
      Phase 2 — unpowered coast (if target not yet reached after burnout)

    Returns a DataFrame with the full time history.
    """
    t_max_powered  = burn_time_s
    t_max_coast    = 600.0                 # 10 min coast max

    y0 = [0.0, 0.0, M_LAUNCH_KG]

    # ── Phase 1: powered ────────────────────────────────────────────────────
    sol1 = solve_ivp(
        odes,
        t_span=(0.0, t_max_powered),
        y0=y0,
        method="RK45",
        events=[event_target_altitude, event_apogee],
        max_step=0.5,
        rtol=1e-8,
        atol=1e-10,
    )

    t1 = sol1.t
    h1, v1, m1 = sol1.y

    target_reached = (
        sol1.status == 1 and
        len(sol1.t_events[0]) > 0
    )

    if target_reached:
        print("Target altitude reached during powered phase.")
        df = _build_dataframe(t1, h1, v1, m1)
        return df

    # ── Phase 2: coast ───────────────────────────────────────────────────────
    y0_coast = [h1[-1], v1[-1], m1[-1]]
    t0_coast = t1[-1]

    sol2 = solve_ivp(
        odes,
        t_span=(t0_coast, t0_coast + t_max_coast),
        y0=y0_coast,
        method="RK45",
        events=[event_target_altitude, event_apogee, event_ground],
        max_step=1.0,
        rtol=1e-8,
        atol=1e-10,
    )

    t2 = sol2.t
    h2, v2, m2 = sol2.y

    t_all = np.concatenate([t1, t2])
    h_all = np.concatenate([h1, h2])
    v_all = np.concatenate([v1, v2])
    m_all = np.concatenate([m1, m2])

    df = _build_dataframe(t_all, h_all, v_all, m_all)
    return df


def _build_dataframe(t, h, v, m) -> pd.DataFrame:
    """Augment raw ODE output with derived quantities."""
    burning = m > m_dry_kg + 1e-6

    thrust   = np.array([thrust_n(hi, bi) for hi, bi in zip(h, burning)])
    isp      = np.where(burning, thrust / (mdot_kg_s * G0), 0.0)
    p_amb    = np.array([ambient_pressure_pa(hi) for hi in h])
    accel    = np.gradient(v, t)

    return pd.DataFrame({
        "t_s":          t,
        "h_m":          h,
        "v_m_s":        v,
        "m_kg":         m,
        "thrust_N":     thrust,
        "isp_s":        isp,
        "p_amb_Pa":     p_amb,
        "accel_m_s2":   accel,
        "burning":      burning.astype(int),
    })


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_trajectory(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    def _save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved: {path}")

    burnout_t = df.loc[df["burning"] == 1, "t_s"].max() if df["burning"].any() else 0.0

    # 1. Altitude vs time
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["h_m"] / 1000.0, color="steelblue")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.axhline(TARGET_ALTITUDE_M / 1000.0, color="green", linestyle=":", alpha=0.7, label="Target")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Altitude  [km]")
    ax.set_title("Altitude vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "altitude_vs_time.png")

    # 2. Velocity vs time
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["v_m_s"], color="darkorange")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Velocity  [m/s]")
    ax.set_title("Velocity vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "velocity_vs_time.png")

    # 3. Thrust vs altitude
    fig, ax = plt.subplots(figsize=FIGSIZE)
    powered = df[df["burning"] == 1]
    ax.plot(powered["h_m"] / 1000.0, powered["thrust_N"], color="crimson")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Thrust  [N]")
    ax.set_title("Thrust vs Altitude\n(pressure thrust gain with altitude)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "thrust_vs_altitude.png")

    # 4. Isp vs altitude
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(powered["h_m"] / 1000.0, powered["isp_s"], color="purple")
    ax.axhline(ISP_VAC_S, color="gray", linestyle=":", alpha=0.7, label=f"Ivac = {ISP_VAC_S} s")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Specific impulse  Isp  [s]")
    ax.set_title("Delivered Isp vs Altitude")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "isp_vs_altitude.png")

    # 5. Vehicle mass vs time
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["m_kg"], color="teal")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.axhline(m_dry_kg, color="gray", linestyle=":", alpha=0.7, label=f"Dry mass = {m_dry_kg:.1f} kg")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Vehicle mass  [kg]")
    ax.set_title("Mass vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "mass_vs_time.png")

    # 6. Acceleration vs time
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["accel_m_s2"] / G0, color="goldenrod")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Acceleration  [g]")
    ax.set_title("Acceleration vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "acceleration_vs_time.png")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame) -> None:
    h_max       = df["h_m"].max()
    v_burnout   = df.loc[df["burning"] == 1, "v_m_s"].iloc[-1] if df["burning"].any() else 0.0
    t_burnout   = df.loc[df["burning"] == 1, "t_s"].max()      if df["burning"].any() else 0.0
    isp_mean    = df.loc[df["burning"] == 1, "isp_s"].mean()   if df["burning"].any() else 0.0
    target_hit  = h_max >= TARGET_ALTITUDE_M

    print("\n" + "═" * 50)
    print("  TRAJECTORY SUMMARY")
    print("═" * 50)
    print(f"  Launch mass          : {M_LAUNCH_KG:.2f} kg")
    print(f"  Dry mass             : {m_dry_kg:.2f} kg")
    print(f"  Propellant mass      : {m_prop_kg:.2f} kg")
    print(f"    Fuel  (RP-1)       : {m_prop_kg / (1.0 + OF):.2f} kg")
    print(f"    Ox   (98% H2O2)    : {m_prop_kg * OF / (1.0 + OF):.2f} kg")
    print(f"  Mass flow rate       : {mdot_kg_s:.4f} kg/s")
    print(f"  Burn time            : {t_burnout:.1f} s")
    print(f"  Burnout velocity     : {v_burnout:.1f} m/s")
    print(f"  Peak altitude        : {h_max/1000:.3f} km")
    print(f"  Target altitude      : {TARGET_ALTITUDE_M/1000:.3f} km  {'✓ REACHED' if target_hit else '✗ NOT REACHED'}")
    print(f"  Mean delivered Isp   : {isp_mean:.1f} s")
    print(f"  Sea-level thrust     : {THRUST_N:.1f} N")
    print(f"  Chamber pressure     : {float(_dp['Pc_bar']):.1f} bar")
    print(f"  O/F                  : {OF:.3f}")
    print(f"  Throat area          : {At_m2*1e4:.4f} cm²")
    print(f"  Exit area            : {Ae_m2*1e4:.4f} cm²")
    print(f"  Ae/At                : {float(_dp['ae_at']):.3f}")
    print("═" * 50 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Running trajectory simulation...")
    print(f"  Launch mass   : {M_LAUNCH_KG} kg")
    print(f"  Propellant    : {m_prop_kg:.2f} kg  (burn time ≈ {burn_time_s:.1f} s)")
    print(f"  Target alt    : {TARGET_ALTITUDE_M/1000:.1f} km")
    print(f"  Drag          : {'ON' if DRAG else 'OFF'}  (Cd={CD}, D={BODY_DIAMETER_M*100:.1f} cm)")

    df = run_trajectory()

    outdir = "trajectory_results"
    df.to_csv(f"./outputs/{outdir}/trajectory_results.csv", index=False)
    print(f"\nFull time history saved to: trajectory_results.csv")

    plot_trajectory(df, outdir)
    print_summary(df)


if __name__ == "__main__":
    main()
