"""
CEA_Wrap interface functions for RP-1 / 98% H2O2 rocket performance.

This module isolates all CEA_Wrap-specific code from the study logic.
"""
from __future__ import annotations

from typing import Any, Dict

from CEA_Wrap import RocketProblem, Fuel, Oxidizer


def build_materials_rp1_h2o2(
    *,
    fuel_T_K: float,
    ox_T_K: float,
    h2o2_mass_frac: float = 0.98,
    h2o_mass_frac: float = 0.02,
    fuel_name: str = "RP-1",
    h2o2_name: str = "H2O2(L)",
    h2o_name: str = "H2O(L)",
) -> Dict[str, object]:
    """
    Build reactants for:
      - Fuel: RP-1
      - Oxidizer mixture: 98% H2O2 + 2% H2O by mass

    IMPORTANT: CEA_Wrap requires reactants to be Fuel/Oxidizer objects (not base Material).
    """
    if abs(h2o2_mass_frac + h2o_mass_frac - 1.0) > 1e-9:
        raise ValueError("Oxidizer mass fractions must sum to 1.0")

    fuel = Fuel(fuel_name, temp=fuel_T_K, wt=1.0)

    # wt values are relative; they don't need to sum to 100, but it's convenient.
    ox_h2o2 = Oxidizer(h2o2_name, temp=ox_T_K, wt=100.0 * h2o2_mass_frac)
    ox_h2o = Oxidizer(h2o_name, temp=ox_T_K, wt=100.0 * h2o_mass_frac)

    return {"fuel": fuel, "ox_h2o2": ox_h2o2, "ox_h2o": ox_h2o}


def run_rocket_case(
    *,
    Pc_bar: float,
    OF: float,
    ae_at: float,
    analysis_type: str = "equilibrium",
    fuel_T_K: float = 298.15,
    ox_T_K: float = 298.15,
    h2o2_mass_frac: float = 0.98,
    h2o_mass_frac: float = 0.02,
    fuel_name: str = "RP-1",
    h2o2_name: str = "H2O2(L)",
    h2o_name: str = "H2O(L)",
    massf: bool = False,
) -> Dict[str, Any]:
    """
    Run one CEA RocketProblem case and return a flat dict of key outputs.
    """
    mats = build_materials_rp1_h2o2(
        fuel_T_K=fuel_T_K,
        ox_T_K=ox_T_K,
        h2o2_mass_frac=h2o2_mass_frac,
        h2o_mass_frac=h2o_mass_frac,
        fuel_name=fuel_name,
        h2o2_name=h2o2_name,
        h2o_name=h2o_name,
    )

    prob = RocketProblem(
        pressure=Pc_bar,
        pressure_units="bar",
        materials=[mats["fuel"], mats["ox_h2o2"], mats["ox_h2o"]],
        o_f=OF,
        ae_at=ae_at,
        analysis_type=analysis_type,
        massf=massf,
    )

    data = prob.run_cea()

    return {
        "Pc_bar": Pc_bar,
        "OF": OF,
        "ae_at": ae_at,
        "analysis_type": analysis_type,
        "Tc_K": getattr(data, "c_t", None),
        "cstar_m_s": getattr(data, "cstar", None),
        "cf": getattr(data, "cf", None),
        "ivac_s": getattr(data, "ivac", None),
        "isp_s": getattr(data, "isp", None),
        "mw_kg_kmol": getattr(data, "mw", None),
        "gammas": getattr(data, "gammas", None),
        "gamma": getattr(data, "gamma", None),
        "phi": getattr(data, "phi", None),
        "pip": getattr(data, "pip", None),
        "ae": getattr(data, "ae", None),
    }
