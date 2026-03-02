#!/usr/bin/env python3
"""
First-pass convective heat-transfer coefficients for rocket cooling studies.

What this script computes:
- Hot-gas-side h_g at throat using a simplified Bartz-style relation.
- Coolant-side h for fuel and oxidizer channels using internal-flow correlations.

Inputs:
- A study results.csv row (selected by Pc/OF or minimum mdot by default).
- Channel geometry + channel count for fuel/oxidizer.
- Fluid properties for fuel/oxidizer (defaults can be overridden).

Notes:
- This is a sizing-level estimator, not a CFD replacement.
- Hot-gas transport is approximated from gamma/MW/Tc unless user overrides.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

R_UNIV = 8314.462618  # J/(kmol*K)


@dataclass
class FluidProps:
    rho: float  # kg/m^3
    mu: float   # Pa*s
    cp: float   # J/(kg*K)
    k: float    # W/(m*K)


@dataclass
class ChannelGeom:
    width_m: float
    height_m: float
    length_m: float
    count: int


@dataclass
class ConvectiveResult:
    Re: float
    Pr: float
    Nu: float
    h: float
    v: float
    dh: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate hot-gas/fuel/oxidizer convective h coefficients")
    p.add_argument("--results", default="outputs/frozen_throat/results.csv", help="Path to study results.csv")
    p.add_argument("--pc", type=float, default=None, help="Select row by Pc_bar")
    p.add_argument("--of", type=float, default=None, help="Select row by OF")
    p.add_argument("--tol", type=float, default=1e-4, help="Float matching tolerance for Pc/OF")

    # Gas-side model controls
    p.add_argument("--pr-gas", type=float, default=0.83, help="Assumed hot-gas Prandtl number")
    p.add_argument("--mu-gas", type=float, default=None, help="Override hot-gas dynamic viscosity [Pa*s]")
    p.add_argument("--rc-over-rt", type=float, default=1.5, help="Assumed throat wall radius ratio Rc/Rt")

    # Fuel channel geometry (mm, mm, m, count)
    p.add_argument("--fuel-w-mm", type=float, default=1.0, help="Fuel channel width [mm]")
    p.add_argument("--fuel-h-mm", type=float, default=1.5, help="Fuel channel height [mm]")
    p.add_argument("--fuel-l-m", type=float, default=0.20, help="Fuel channel flow length [m]")
    p.add_argument("--fuel-ch", type=int, default=60, help="Fuel channel count")

    # Ox channel geometry
    p.add_argument("--ox-w-mm", type=float, default=1.0, help="Ox channel width [mm]")
    p.add_argument("--ox-h-mm", type=float, default=1.5, help="Ox channel height [mm]")
    p.add_argument("--ox-l-m", type=float, default=0.20, help="Ox channel flow length [m]")
    p.add_argument("--ox-ch", type=int, default=60, help="Ox channel count")

    # Fuel properties (defaults ~RP-1, moderate temperature)
    p.add_argument("--fuel-rho", type=float, default=810.0, help="Fuel density [kg/m^3]")
    p.add_argument("--fuel-mu", type=float, default=1.8e-3, help="Fuel viscosity [Pa*s]")
    p.add_argument("--fuel-cp", type=float, default=2100.0, help="Fuel cp [J/(kg*K)]")
    p.add_argument("--fuel-k", type=float, default=0.13, help="Fuel thermal conductivity [W/(m*K)]")

    # Ox properties (defaults ~98% H2O2/water, moderate temperature)
    p.add_argument("--ox-rho", type=float, default=1380.0, help="Ox density [kg/m^3]")
    p.add_argument("--ox-mu", type=float, default=1.2e-3, help="Ox viscosity [Pa*s]")
    p.add_argument("--ox-cp", type=float, default=2700.0, help="Ox cp [J/(kg*K)]")
    p.add_argument("--ox-k", type=float, default=0.50, help="Ox thermal conductivity [W/(m*K)]")

    p.add_argument("--export", default=None, help="Optional output CSV for computed coefficients")
    return p.parse_args()


def fget(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_row(rows: List[Dict[str, str]], pc: Optional[float], of: Optional[float], tol: float) -> Dict[str, str]:
    if pc is not None and of is not None:
        nearest: Optional[Dict[str, str]] = None
        nearest_dist = float("inf")
        for row in rows:
            pc_r = fget(row, "Pc_bar")
            of_r = fget(row, "OF")
            if pc_r is None or of_r is None:
                continue
            if abs(pc_r - pc) <= tol and abs(of_r - of) <= tol:
                return row
            dist = ((pc_r - pc) ** 2 + (of_r - of) ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = row
        if nearest is None:
            raise ValueError(f"No row found for Pc={pc}, OF={of}.")
        return nearest

    valid = [r for r in rows if fget(r, "mdot_kg_s") is not None]
    if not valid:
        raise ValueError("No rows with mdot_kg_s found.")
    return min(valid, key=lambda r: fget(r, "mdot_kg_s") or float("inf"))


def sutherland_mu(T: float, mu_ref: float = 1.716e-5, T_ref: float = 273.15, S: float = 110.4) -> float:
    """Simple gas-viscosity estimate when no CEA transport is available."""
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def coolant_h(mdot_total: float, fluid: FluidProps, geom: ChannelGeom, heating: bool) -> ConvectiveResult:
    if geom.count <= 0:
        raise ValueError("Channel count must be > 0")

    mdot_ch = mdot_total / geom.count
    area = geom.width_m * geom.height_m
    dh = 2.0 * geom.width_m * geom.height_m / (geom.width_m + geom.height_m)

    v = mdot_ch / (fluid.rho * area)
    Re = fluid.rho * v * dh / fluid.mu
    Pr = fluid.cp * fluid.mu / fluid.k

    # Turbulent: Gnielinski. Laminar/developing fallback: Sieder-Tate style.
    if Re >= 3000.0:
        f = (0.79 * math.log(Re) - 1.64) ** -2
        Nu = ((f / 8.0) * (Re - 1000.0) * Pr) / (1.0 + 12.7 * math.sqrt(f / 8.0) * (Pr ** (2.0 / 3.0) - 1.0))
    else:
        gzl = max(Re * Pr * dh / max(geom.length_m, 1e-6), 1e-12)
        Nu = 1.86 * (gzl ** (1.0 / 3.0))

    h = Nu * fluid.k / dh
    return ConvectiveResult(Re=Re, Pr=Pr, Nu=Nu, h=h, v=v, dh=dh)


def hot_gas_h_bartz(Pc_Pa: float, cstar: float, dt_m: float, Tc_K: float, gamma: float, mw: float, pr: float, rc_over_rt: float, mu_override: Optional[float]) -> float:
    """
    Simplified Bartz-style estimate at throat (A_t/A = 1, sigma = 1).

    h_g = 0.026 / D_t^0.2 * (mu^0.2 * cp / Pr^0.6) * (Pc/c*)^0.8 * (D_t/R_c)^0.1
    """
    R_g = R_UNIV / mw  # J/(kg*K), mw in kg/kmol
    cp_g = gamma * R_g / (gamma - 1.0)
    mu_g = mu_override if (mu_override is not None and mu_override > 0.0) else sutherland_mu(Tc_K)

    rc_over_rt = max(rc_over_rt, 1.0)
    rt = 0.5 * dt_m
    rc = rc_over_rt * rt

    h = (
        0.026
        * (mu_g ** 0.2)
        * (cp_g / (pr ** 0.6))
        * ((Pc_Pa / cstar) ** 0.8)
        * ((dt_m / rc) ** 0.1)
        / (dt_m ** 0.2)
    )
    return h


def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3


def export_csv(path: str, row: Dict[str, str], h_g: float, fuel: ConvectiveResult, ox: ConvectiveResult) -> None:
    fields = [
        "Pc_bar", "OF", "analysis_type", "nfz",
        "h_gas_W_m2K", "h_fuel_W_m2K", "h_ox_W_m2K",
        "Re_fuel", "Re_ox", "Nu_fuel", "Nu_ox", "Pr_fuel", "Pr_ox",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "Pc_bar": row.get("Pc_bar", ""),
            "OF": row.get("OF", ""),
            "analysis_type": row.get("analysis_type", ""),
            "nfz": row.get("nfz", ""),
            "h_gas_W_m2K": f"{h_g:.6g}",
            "h_fuel_W_m2K": f"{fuel.h:.6g}",
            "h_ox_W_m2K": f"{ox.h:.6g}",
            "Re_fuel": f"{fuel.Re:.6g}",
            "Re_ox": f"{ox.Re:.6g}",
            "Nu_fuel": f"{fuel.Nu:.6g}",
            "Nu_ox": f"{ox.Nu:.6g}",
            "Pr_fuel": f"{fuel.Pr:.6g}",
            "Pr_ox": f"{ox.Pr:.6g}",
        })


def main() -> None:
    a = parse_args()
    rows = read_rows(a.results)
    row = pick_row(rows, a.pc, a.of, a.tol)

    Pc_bar = fget(row, "Pc_bar")
    OF = fget(row, "OF")
    Tc = fget(row, "Tc_K")
    cstar = fget(row, "cstar_m_s")
    gamma = fget(row, "gamma")
    mw = fget(row, "mw_kg_kmol")
    dt = fget(row, "dt_m")
    mdot_fuel = fget(row, "mdot_fuel_kg_s")
    mdot_ox = fget(row, "mdot_ox_kg_s")

    required = {
        "Pc_bar": Pc_bar,
        "OF": OF,
        "Tc_K": Tc,
        "cstar_m_s": cstar,
        "gamma": gamma,
        "mw_kg_kmol": mw,
        "dt_m": dt,
        "mdot_fuel_kg_s": mdot_fuel,
        "mdot_ox_kg_s": mdot_ox,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Selected row missing required fields: {missing}")

    fuel_geom = ChannelGeom(mm_to_m(a.fuel_w_mm), mm_to_m(a.fuel_h_mm), a.fuel_l_m, a.fuel_ch)
    ox_geom = ChannelGeom(mm_to_m(a.ox_w_mm), mm_to_m(a.ox_h_mm), a.ox_l_m, a.ox_ch)

    fuel_props = FluidProps(a.fuel_rho, a.fuel_mu, a.fuel_cp, a.fuel_k)
    ox_props = FluidProps(a.ox_rho, a.ox_mu, a.ox_cp, a.ox_k)

    h_g = hot_gas_h_bartz(
        Pc_Pa=Pc_bar * 1e5,
        cstar=cstar,
        dt_m=dt,
        Tc_K=Tc,
        gamma=gamma,
        mw=mw,
        pr=a.pr_gas,
        rc_over_rt=a.rc_over_rt,
        mu_override=a.mu_gas,
    )

    h_fuel = coolant_h(mdot_total=mdot_fuel, fluid=fuel_props, geom=fuel_geom, heating=True)
    h_ox = coolant_h(mdot_total=mdot_ox, fluid=ox_props, geom=ox_geom, heating=True)

    print("=== Convective Coefficient Estimate ===")
    print(f"Row: Pc={Pc_bar:.4g} bar, OF={OF:.4g}, analysis={row.get('analysis_type','')} nfz={row.get('nfz','')}")
    print()
    print("Hot-gas side (throat, simplified Bartz)")
    print(f"  h_gas = {h_g:.3f} W/(m^2*K)")
    print(f"  inputs: Tc={Tc:.2f} K, gamma={gamma:.5g}, MW={mw:.5g} kg/kmol, dt={dt*1e3:.3f} mm, Pr={a.pr_gas:.3f}")
    print()
    print("Fuel-side coolant channels")
    print(f"  h_fuel = {h_fuel.h:.3f} W/(m^2*K)")
    print(f"  Re={h_fuel.Re:.3g}, Pr={h_fuel.Pr:.3g}, Nu={h_fuel.Nu:.3g}, v={h_fuel.v:.3f} m/s, Dh={h_fuel.dh*1e3:.3f} mm")
    print()
    print("Ox-side coolant channels")
    print(f"  h_ox = {h_ox.h:.3f} W/(m^2*K)")
    print(f"  Re={h_ox.Re:.3g}, Pr={h_ox.Pr:.3g}, Nu={h_ox.Nu:.3g}, v={h_ox.v:.3f} m/s, Dh={h_ox.dh*1e3:.3f} mm")
    print()
    print("Assumptions:")
    print("  - Gas-side h uses simplified Bartz at throat with estimated gas viscosity unless --mu-gas is provided.")
    print("  - Coolant-side h uses Gnielinski (turbulent) and Sieder-Tate style fallback (laminar/developing).")

    if a.export:
        export_csv(a.export, row, h_g, h_fuel, h_ox)
        print(f"\nWrote: {a.export}")


if __name__ == "__main__":
    main()
