"""
CEA_Wrap interface functions for RP-1 / 98% H2O2 rocket performance.

This module isolates all CEA_Wrap-specific code from the study logic.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from CEA_Wrap import RocketProblem, Fuel, Oxidizer
from core.decomposition import decompose_h2o2_stream


def build_materials_rp1_h2o2(
    *,
    Pc_bar: float,
    fuel_T_K: float,
    ox_T_K: float,
    h2o2_mass_frac: float = 0.98,
    h2o_mass_frac: float = 0.02,
    fuel_name: str = "RP-1",
    h2o2_name: str = "H2O2(L)",
    h2o_name: str = "H2O(L)",
    o2_name: str = "O2",
    decomp_conversion: float = 1.0,
    eta_decomp: float = 1.0,
    delta_p_injector_bar: float = 0.0,
    delta_p_bed_bar: float = 0.0,
) -> Dict[str, object]:
    """
    Build reactants for:
      - Fuel: RP-1
      - Oxidizer decomposition products from H2O2/H2O feed.

    IMPORTANT: CEA_Wrap requires reactants to be Fuel/Oxidizer objects.
    """
    if abs(h2o2_mass_frac + h2o_mass_frac - 1.0) > 1e-9:
        raise ValueError("Oxidizer mass fractions must sum to 1.0")

    decomp = decompose_h2o2_stream(
        Pc_bar=Pc_bar,
        delta_p_injector_bar=delta_p_injector_bar,
        delta_p_bed_bar=delta_p_bed_bar,
        ox_temp_in_K=ox_T_K,
        h2o2_mass_frac_in=h2o2_mass_frac,
        h2o_mass_frac_in=h2o_mass_frac,
        decomp_conversion=decomp_conversion,
        eta_decomp=eta_decomp,
    )

    fuel = Fuel(fuel_name, temp=fuel_T_K, wt=1.0)

    def _de_liquefy(name: str) -> str:
        # CEA liquid tags like "(L)" are invalid at high decomposition temperatures.
        return name.replace("(L)", "").replace("(l)", "").strip()

    h2o2_mat_name = h2o2_name
    h2o_mat_name = h2o_name
    if decomp["ox_temp_out_K"] > 600.0:
        h2o2_mat_name = _de_liquefy(h2o2_name)
        h2o_mat_name = _de_liquefy(h2o_name)
    oxidizers = []
    if decomp["h2o2_mass_frac_out"] > 0.0:
        oxidizers.append(Oxidizer(h2o2_mat_name, temp=decomp["ox_temp_out_K"], wt=100.0 * decomp["h2o2_mass_frac_out"]))
    if decomp["h2o_mass_frac_out"] > 0.0:
        oxidizers.append(Oxidizer(h2o_mat_name, temp=decomp["ox_temp_out_K"], wt=100.0 * decomp["h2o_mass_frac_out"]))
    if decomp["o2_mass_frac_out"] > 0.0:
        oxidizers.append(Oxidizer(o2_name, temp=decomp["ox_temp_out_K"], wt=100.0 * decomp["o2_mass_frac_out"]))

    return {"fuel": fuel, "oxidizers": oxidizers}


def run_rocket_case(
    *,
    Pc_bar: float,
    OF: float,
    # Nozzle definition: specify either ae_at (geometry) OR pip (Pc/Pe).
    ae_at: Optional[float] = None,
    pip: Optional[float] = None,
    analysis_type: str = "equilibrium",
    nfz: Optional[int] = None,
    custom_nfz: Optional[float] = None,
    fuel_T_K: float = 298.15,
    ox_T_K: float = 298.15,
    h2o2_mass_frac: float = 0.98,
    h2o_mass_frac: float = 0.02,
    # Material names can differ depending on thermo library content.
    # Adjust here if needed.
    fuel_name: str = "RP-1",
    h2o2_name: str = "H2O2(L)",
    h2o_name: str = "H2O(L)",
    o2_name: str = "O2",
    decomp_conversion: float = 1.0,
    eta_decomp: float = 1.0,
    delta_p_injector_bar: float = 0.0,
    delta_p_bed_bar: float = 0.0,
    massf: bool = False,
) -> Dict[str, Any]:
    """
    Run one CEA RocketProblem case and return a flat dict of key outputs.

    Notes:
    - Sea-level ideally-expanded design: set pip = Pc/Pe with Pe≈1.01325 bar.
      CEA will compute the corresponding area ratio 'ae' (Ae/At).
    - CEA_Wrap output key 'ae' is the ratio Ae/At. (Not absolute area.)
    - analysis_type supports standard CEA_Wrap options such as:
      "equilibrium", "frozen", and frozen variants using nfz/custom_nfz.
    """
    if (ae_at is None) == (pip is None):
        raise ValueError("Specify exactly one of ae_at or pip (not both).")

    mats = build_materials_rp1_h2o2(
        Pc_bar=Pc_bar,
        fuel_T_K=fuel_T_K,
        ox_T_K=ox_T_K,
        h2o2_mass_frac=h2o2_mass_frac,
        h2o_mass_frac=h2o_mass_frac,
        fuel_name=fuel_name,
        h2o2_name=h2o2_name,
        h2o_name=h2o_name,
        o2_name=o2_name,
        decomp_conversion=decomp_conversion,
        eta_decomp=eta_decomp,
        delta_p_injector_bar=delta_p_injector_bar,
        delta_p_bed_bar=delta_p_bed_bar,
    )

    kwargs = dict(
        pressure=Pc_bar,
        pressure_units="bar",
        materials=[mats["fuel"], *mats["oxidizers"]],
        o_f=OF,
        analysis_type=analysis_type,
        massf=massf,
    )
    if ae_at is not None:
        kwargs["ae_at"] = float(ae_at)
    if pip is not None:
        kwargs["pip"] = float(pip)
    if nfz is not None:
        kwargs["nfz"] = int(nfz)
    if custom_nfz is not None:
        kwargs["custom_nfz"] = float(custom_nfz)

    prob = RocketProblem(**kwargs)
    data = prob.run_cea()

    # CEA_Wrap may return pip/ae as 0 for some frozen runs even when a valid
    # design pip input was provided. Recover geometry-driving values so sizing
    # does not collapse to Ae=0.
    def _as_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except (TypeError, ValueError):
            return None

    def _aeat_from_pip_gamma(pip_val: float, gamma_val: float) -> Optional[float]:
        """
        Compute Ae/At from pressure ratio and gamma (ideal isentropic relation).
        pip_val = Pc/Pe.
        """
        if pip_val <= 1.0 or gamma_val <= 1.0:
            return None
        pe_pc = 1.0 / pip_val
        base = (2.0 / (gamma_val + 1.0)) ** (1.0 / (gamma_val - 1.0))
        pr = pip_val ** (1.0 / gamma_val)
        bracket = ((gamma_val + 1.0) / (gamma_val - 1.0)) * (
            1.0 - pe_pc ** ((gamma_val - 1.0) / gamma_val)
        )
        if bracket <= 0.0:
            return None
        return base * pr * (bracket ** -0.5)

    out: Dict[str, Any] = {
        "Pc_bar": Pc_bar,
        "OF": OF,
        "analysis_type": analysis_type,
        "nfz": nfz,
        "custom_nfz": custom_nfz,
        "Tc_K": getattr(data, "c_t", None),
        "cstar_m_s": getattr(data, "cstar", None),
        "cf": getattr(data, "cf", None),
        "ivac_s": getattr(data, "ivac", None),
        "isp_s": getattr(data, "isp", None),
        "mw_kg_kmol": getattr(data, "mw", None),
        "gamma": getattr(data, "gamma", None),
        "gammas": getattr(data, "gammas", None),
        "phi": getattr(data, "phi", None),
        "pip": getattr(data, "pip", None),
        "ae_at": getattr(data, "ae", None),  # Ae/At ratio
    }

    returned_pip = _as_float(out.get("pip"))
    input_pip = _as_float(pip)
    if returned_pip is None or returned_pip <= 0.0:
        out["pip"] = input_pip
        returned_pip = input_pip

    returned_ae = _as_float(out.get("ae_at"))
    if returned_ae is None or returned_ae <= 0.0:
        gamma_for_nozzle = _as_float(getattr(data, "gammas", None))
        if gamma_for_nozzle is None or gamma_for_nozzle <= 1.0:
            gamma_for_nozzle = _as_float(getattr(data, "gamma", None))
        if returned_pip is not None and gamma_for_nozzle is not None:
            ae_recovered = _aeat_from_pip_gamma(returned_pip, gamma_for_nozzle)
            if ae_recovered is not None and math.isfinite(ae_recovered) and ae_recovered > 0.0:
                out["ae_at"] = ae_recovered

    if out["pip"] not in (None, 0):
        out["Pe_bar"] = Pc_bar / float(out["pip"])
    else:
        out["Pe_bar"] = None

    return out
