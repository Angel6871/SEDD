"""
Hydrogen peroxide catalytic decomposition utilities.

Model:
  2 H2O2 -> 2 H2O + O2

Temperature:
  - equilibrium decomposition temperature from CEA_Wrap HPProblem at
    P_decomp = Pc + 0.25*Pc + delta_p_bed
  - scaled with eta_decomp:
      T_out = T_in + eta_decomp * (T_eq - T_in)

Composition:
  - stoichiometric conversion model controlled by decomp_conversion.
"""
from __future__ import annotations

from CEA_Wrap import Fuel, HPProblem, Oxidizer

# Molar masses [kg/mol]
MW_H2O2 = 34.0147e-3
MW_H2O = 18.01528e-3
MW_O2 = 31.9988e-3
INJECTOR_DP_FRACTION_OF_PC = 0.25


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _de_liquefy(name: str) -> str:
    return name.replace("(L)", "").replace("(l)", "").strip()


def equilibrium_decomp_temperature_hp(
    *,
    pressure_bar: float,
    ox_temp_in_K: float,
    h2o2_mass_frac_in: float,
    h2o_mass_frac_in: float,
) -> float:
    """
    Equilibrium decomposition temperature from CEA_Wrap HPProblem.
    """
    p_bar = float(pressure_bar)
    if p_bar <= 0.0:
        raise ValueError("pressure_bar must be > 0 for HP decomposition.")

    h2o2_w = float(h2o2_mass_frac_in)
    h2o_w = max(float(h2o_mass_frac_in), 1e-9)  # HPProblem needs a non-zero ratio denominator.

    h2o_name = "H2O(L)" if ox_temp_in_K <= 600.0 else _de_liquefy("H2O(L)")
    h2o2_name = "H2O2(L)" if ox_temp_in_K <= 600.0 else _de_liquefy("H2O2(L)")

    mats = [
        Fuel(h2o_name, temp=float(ox_temp_in_K), wt=100.0 * h2o_w),
        Oxidizer(h2o2_name, temp=float(ox_temp_in_K), wt=100.0 * h2o2_w),
    ]

    out = HPProblem(
        pressure=p_bar,
        pressure_units="bar",
        materials=mats,
        o_f=h2o2_w / h2o_w,
        massf=True,
    ).run()
    return float(out.t)


def decompose_h2o2_stream(
    *,
    Pc_bar: float,
    delta_p_bed_bar: float,
    ox_temp_in_K: float,
    h2o2_mass_frac_in: float,
    h2o_mass_frac_in: float,
    decomp_conversion: float,
    eta_decomp: float,
) -> dict[str, float]:
    """
    Return oxidizer outlet composition and temperature after catalytic decomposition.
    """
    h2o2_in = float(h2o2_mass_frac_in)
    h2o_in = float(h2o_mass_frac_in)
    if abs((h2o2_in + h2o_in) - 1.0) > 1e-9:
        raise ValueError("h2o2_mass_frac_in + h2o_mass_frac_in must equal 1.0")

    conv = _clamp01(decomp_conversion)
    eta = _clamp01(eta_decomp)

    delta_p_injector_bar = INJECTOR_DP_FRACTION_OF_PC * float(Pc_bar)
    p_decomp_bar = float(Pc_bar) + delta_p_injector_bar + float(delta_p_bed_bar)
    t_eq_K = equilibrium_decomp_temperature_hp(
        pressure_bar=p_decomp_bar,
        ox_temp_in_K=float(ox_temp_in_K),
        h2o2_mass_frac_in=h2o2_in,
        h2o_mass_frac_in=h2o_in,
    )

    # Basis: 1 kg oxidizer stream.
    m_h2o2_in = h2o2_in
    m_h2o_in = h2o_in
    m_h2o2_reacted = m_h2o2_in * conv
    m_h2o2_left = m_h2o2_in - m_h2o2_reacted

    # Stoichiometric mass split of reacted H2O2:
    # H2O2 -> H2O + 0.5 O2 (masses per mol preserve total mass).
    m_h2o_prod = m_h2o2_reacted * (MW_H2O / MW_H2O2)
    m_o2_prod = m_h2o2_reacted * (0.5 * MW_O2 / MW_H2O2)

    m_h2o_out = m_h2o_in + m_h2o_prod
    m_o2_out = m_o2_prod
    m_total = m_h2o2_left + m_h2o_out + m_o2_out

    t_out_K = float(ox_temp_in_K) + eta * (t_eq_K - float(ox_temp_in_K))

    return {
        "ox_temp_out_K": t_out_K,
        "h2o2_mass_frac_out": m_h2o2_left / m_total,
        "h2o_mass_frac_out": m_h2o_out / m_total,
        "o2_mass_frac_out": m_o2_out / m_total,
        "p_decomp_bar": p_decomp_bar,
        "delta_p_injector_bar": delta_p_injector_bar,
        "ox_temp_eq_K": t_eq_K,
    }
