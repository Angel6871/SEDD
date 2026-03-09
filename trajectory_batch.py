"""
Batch trajectory post-processing for results.csv.

- Runs trajectory for each row only when nozzle_mode == 'pip'.
- Skips non-pip rows safely.
- Appends altitude output columns to a new CSV.

Run separately from study_new.py:
    python3 trajectory_batch.py --results outputs/frozen_throat_many_pc/results.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Vehicle/model assumptions (same role as trajectory.py defaults)
M_LAUNCH_KG = 30.0
M_PROP_KG = 5.0
CD = 0.7
BODY_DIAMETER_M = 0.25
DRAG = True
TARGET_ALTITUDE_M = 1515.0

# Physics constants
G0 = 9.80665
G = 6.67430e-11
R_EARTH = 6_371_000.0
M_EARTH = 5.972e24
P_SL_PA = 101_325.0
R_AIR = 287.05
FIGSIZE = (7, 4.5)
DPI = 200

# ── File paths (edit these for quick use without CLI args) ────────────────
# Set to None to auto-pick newest outputs/*/results.csv
INPUT_RESULTS_CSV = "outputs/frozen_throat_1250/best_by_mdot.csv"
# Set to None to auto-write next to input as results_with_altitude.csv
OUTPUT_RESULTS_CSV = "outputs/frozen_throat_1250/best_by_mdot_with_altitude.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch trajectory evaluator for results.csv")
    p.add_argument("--results", default=INPUT_RESULTS_CSV, help="Path to input results.csv (default: INPUT_RESULTS_CSV)")
    p.add_argument("--out", default=OUTPUT_RESULTS_CSV, help="Path to output CSV (default: OUTPUT_RESULTS_CSV)")
    p.add_argument("--plots-dir", default=None, help="Output dir for best-config trajectory plots (default: <out_dir>/trajectory_plots_best)")
    p.add_argument("--t-max-coast", type=float, default=600.0, help="Max coast duration after burnout [s]")
    p.add_argument("--max-step", type=float, default=0.5, help="Integrator max step [s]")
    return p.parse_args()


def resolve_results_path(path: Optional[str]) -> str:
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"results file not found: {path}")
        return path

    candidates = [p for p in glob.glob("outputs/*/results.csv") if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError("No outputs/*/results.csv found. Provide --results.")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def ambient_pressure_pa(h_m: float) -> float:
    """ISA pressure up to ~20 km."""
    h = max(0.0, float(h_m))
    if h <= 11_000.0:
        T = 288.15 - 0.0065 * h
        return P_SL_PA * (T / 288.15) ** 5.255877

    # Lower stratosphere (isothermal approximation)
    T = 216.65
    P11 = 22_632.06
    return P11 * math.exp(-G0 * (h - 11_000.0) / (R_AIR * T))


def ambient_density_kg_m3(h_m: float) -> float:
    h = max(0.0, float(h_m))
    if h <= 11_000.0:
        T = 288.15 - 0.0065 * h
    else:
        T = 216.65
    return ambient_pressure_pa(h) / (R_AIR * T)


def gravity_m_s2(h_m: float) -> float:
    return G * M_EARTH / (R_EARTH + max(0.0, float(h_m))) ** 2


def run_trajectory_history(
    thrust_sl_n: float,
    ae_m2: float,
    mdot_kg_s: float,
    *,
    t_max_coast: float,
    max_step: float,
) -> tuple[pd.DataFrame, float, str]:
    """
    Returns:
      full history df, burn_time_s, status
    """
    if thrust_sl_n <= 0 or ae_m2 < 0 or mdot_kg_s <= 0:
        return pd.DataFrame(), math.nan, "invalid_inputs"

    m_dry = M_LAUNCH_KG - M_PROP_KG
    if m_dry <= 0:
        return pd.DataFrame(), math.nan, "invalid_mass_model"

    burn_time = M_PROP_KG / mdot_kg_s
    a_ref = math.pi * (BODY_DIAMETER_M / 2.0) ** 2

    def thrust_n(h_m: float, burning: bool) -> float:
        if not burning:
            return 0.0
        return thrust_sl_n + (P_SL_PA - ambient_pressure_pa(h_m)) * ae_m2

    def odes(_t: float, y: list[float]) -> list[float]:
        h, v, m = y
        burning = m > (m_dry + 1e-9)

        F = thrust_n(h, burning)
        g = gravity_m_s2(h)

        if DRAG and abs(v) > 0.0:
            rho = ambient_density_kg_m3(h)
            f_drag = 0.5 * rho * v * abs(v) * CD * a_ref
        else:
            f_drag = 0.0

        dm_dt = -mdot_kg_s if burning else 0.0
        dv_dt = (F - f_drag) / m - g
        dh_dt = v
        return [dh_dt, dv_dt, dm_dt]

    def event_apogee(_t: float, y: list[float]) -> float:
        return y[1]

    event_apogee.terminal = True
    event_apogee.direction = -1

    def event_ground(t: float, y: list[float]) -> float:
        # Avoid immediate termination at t=0 when starting at h=0.
        if t < 1e-8:
            return 1.0
        return y[0]

    event_ground.terminal = True
    event_ground.direction = -1

    t_end = burn_time + t_max_coast
    y0 = [0.0, 0.0, M_LAUNCH_KG]

    sol = solve_ivp(
        odes,
        t_span=(0.0, t_end),
        y0=y0,
        method="RK45",
        events=[event_apogee, event_ground],
        max_step=max_step,
        rtol=1e-7,
        atol=1e-9,
    )

    h = sol.y[0]
    v = sol.y[1]
    t = sol.t

    altitude_reached = float(np.nanmax(h)) if len(h) else math.nan

    # Burnout samples are those before/at burn time.
    mask_burn = t <= burn_time + 1e-9
    if np.any(mask_burn):
        h_burn = float(h[mask_burn][-1])
        v_burn = float(v[mask_burn][-1])
    else:
        h_burn = math.nan
        v_burn = math.nan

    if not math.isfinite(altitude_reached):
        status = "solver_failed"
    elif altitude_reached >= TARGET_ALTITUDE_M:
        status = "target_reached"
    else:
        status = "target_not_reached"

    m = sol.y[2]
    burning = m > (m_dry + 1e-9)
    thrust = np.array([thrust_n(hi, bool(bi)) for hi, bi in zip(h, burning)])
    isp = np.where(burning, thrust / (mdot_kg_s * G0), 0.0)
    p_amb = np.array([ambient_pressure_pa(hi) for hi in h])
    accel = np.gradient(v, t) if len(t) > 1 else np.array([0.0])

    hist = pd.DataFrame({
        "t_s": t,
        "h_m": h,
        "v_m_s": v,
        "m_kg": m,
        "thrust_N": thrust,
        "isp_s": isp,
        "p_amb_Pa": p_amb,
        "accel_m_s2": accel,
        "burning": burning.astype(int),
    })
    return hist, burn_time, status


def simulate_row(
    thrust_sl_n: float,
    ae_m2: float,
    mdot_kg_s: float,
    *,
    t_max_coast: float,
    max_step: float,
) -> tuple[float, float, float, str]:
    """
    Returns:
      altitude_reached_m, burnout_altitude_m, burnout_velocity_m_s, status
    """
    hist, burn_time, status = run_trajectory_history(
        thrust_sl_n,
        ae_m2,
        mdot_kg_s,
        t_max_coast=t_max_coast,
        max_step=max_step,
    )
    if hist.empty:
        return (math.nan, math.nan, math.nan, status)

    altitude_reached = float(hist["h_m"].max())
    mask_burn = hist["t_s"] <= burn_time + 1e-9
    if np.any(mask_burn):
        burnout_alt = float(hist.loc[mask_burn, "h_m"].iloc[-1])
        burnout_vel = float(hist.loc[mask_burn, "v_m_s"].iloc[-1])
    else:
        burnout_alt = math.nan
        burnout_vel = math.nan
    return altitude_reached, burnout_alt, burnout_vel, status


def plot_trajectory_like_single(df: pd.DataFrame, outdir: str, isp_vac_s: float) -> None:
    os.makedirs(outdir, exist_ok=True)

    def _save(fig: plt.Figure, name: str) -> None:
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Saved: {path}")

    burnout_t = df.loc[df["burning"] == 1, "t_s"].max() if (df["burning"] == 1).any() else 0.0
    powered = df[df["burning"] == 1]

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

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(powered["h_m"] / 1000.0, powered["thrust_N"], color="crimson")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Thrust  [N]")
    ax.set_title("Thrust vs Altitude\n(pressure thrust gain with altitude)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "thrust_vs_altitude.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(powered["h_m"] / 1000.0, powered["isp_s"], color="purple")
    ax.axhline(isp_vac_s, color="gray", linestyle=":", alpha=0.7, label=f"Ivac = {isp_vac_s:.1f} s")
    ax.set_xlabel("Altitude  [km]")
    ax.set_ylabel("Specific impulse  Isp  [s]")
    ax.set_title("Delivered Isp vs Altitude")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "isp_vs_altitude.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["m_kg"], color="teal")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.axhline(M_LAUNCH_KG - M_PROP_KG, color="gray", linestyle=":", alpha=0.7, label=f"Dry mass = {M_LAUNCH_KG - M_PROP_KG:.1f} kg")
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Vehicle mass  [kg]")
    ax.set_title("Mass vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "mass_vs_time.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["t_s"], df["accel_m_s2"] / G0, color="goldenrod")
    ax.axvline(burnout_t, color="tomato", linestyle="--", alpha=0.7, label="Burnout")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time  [s]")
    ax.set_ylabel("Acceleration  [g]")
    ax.set_title("Acceleration vs Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "acceleration_vs_time.png")


def main() -> None:
    args = parse_args()
    results_path = resolve_results_path(args.results)

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(results_path), "results_with_altitude.csv")

    df = pd.read_csv(results_path)

    n_total = len(df)
    print(f"Loaded: {results_path}")
    print(f"Rows: {n_total}")

    altitude = np.full(n_total, np.nan, dtype=float)
    h_burn = np.full(n_total, np.nan, dtype=float)
    v_burn = np.full(n_total, np.nan, dtype=float)
    status = np.array(["skipped_non_pip"] * n_total, dtype=object)

    # Per user requirement: only calculate trajectory when nozzle_mode == pip.
    if "nozzle_mode" in df.columns:
        is_pip = df["nozzle_mode"].astype(str).str.lower().eq("pip")
    else:
        # best_by_mdot.csv from older study runs may not include nozzle_mode.
        # In that case, treat all rows as candidates and rely on required-column checks.
        print("Warning: nozzle_mode column missing; assuming all rows are pip-compatible.")
        is_pip = pd.Series([True for _ in range(n_total)])

    pip_indices = np.where(is_pip.values)[0]
    print(f"pip rows to simulate: {len(pip_indices)}")

    required = ["F_req_N", "Ae_m2", "mdot_kg_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns for trajectory simulation: "
            f"{missing}. Regenerate best_by_mdot.csv with updated study_new.py."
        )

    for j, idx in enumerate(pip_indices, start=1):
        row = df.iloc[idx]
        try:
            thrust_sl = float(row["F_req_N"])
            ae_m2 = float(row["Ae_m2"])
            mdot = float(row["mdot_kg_s"])

            alt_i, hb_i, vb_i, st_i = simulate_row(
                thrust_sl,
                ae_m2,
                mdot,
                t_max_coast=float(args.t_max_coast),
                max_step=float(args.max_step),
            )
        except Exception:
            alt_i, hb_i, vb_i, st_i = (math.nan, math.nan, math.nan, "row_error")

        altitude[idx] = alt_i
        h_burn[idx] = hb_i
        v_burn[idx] = vb_i
        status[idx] = st_i

        if j % 25 == 0 or j == len(pip_indices):
            print(f"  simulated {j}/{len(pip_indices)} pip rows")

    df["altitude_reached_m"] = altitude
    df["burnout_altitude_m"] = h_burn
    df["burnout_velocity_m_s"] = v_burn
    df["trajectory_status"] = status
    df["is_best_config"] = False

    # Best config = reaches target and has lowest chamber temperature.
    df_target = df[df["trajectory_status"] == "target_reached"].copy()
    best_idx = None
    if len(df_target) > 0:
        if "Tc_K" in df_target.columns:
            df_target["Tc_K_num"] = pd.to_numeric(df_target["Tc_K"], errors="coerce")
            df_target = df_target.dropna(subset=["Tc_K_num"])
            if len(df_target) > 0:
                best_idx = int(df_target.sort_values(["Tc_K_num", "mdot_kg_s"], kind="mergesort").index[0])
        if best_idx is None:
            best_idx = int(df_target.index[0])
        df.loc[best_idx, "is_best_config"] = True

    df.to_csv(out_path, index=False)

    print("Done.")
    print(f"Output: {out_path}")
    print(f"Calculated rows: {(df['trajectory_status'] != 'skipped_non_pip').sum()} / {n_total}")

    if best_idx is None:
        print("No best config found: no pip row reached target altitude.")
        return

    best_row = df.loc[best_idx]
    print(
        "Best config:",
        f"idx={best_idx}",
        f"Pc={best_row.get('Pc_bar', 'n/a')}",
        f"OF={best_row.get('OF', 'n/a')}",
        f"Tc_K={best_row.get('Tc_K', 'n/a')}",
        f"altitude={best_row.get('altitude_reached_m', 'n/a')}",
    )

    plots_dir = args.plots_dir or os.path.join(os.path.dirname(results_path), "best_config_plots")
    best_hist, _, best_status = run_trajectory_history(
        float(best_row["F_req_N"]),
        float(best_row["Ae_m2"]),
        float(best_row["mdot_kg_s"]),
        t_max_coast=float(args.t_max_coast),
        max_step=float(args.max_step),
    )
    if best_hist.empty:
        print(f"Could not generate best-config trajectory plots (status={best_status}).")
        return
    plot_trajectory_like_single(best_hist, plots_dir, isp_vac_s=float(best_row.get("ivac_s", 0.0)))
    print(f"Best-config trajectory plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
