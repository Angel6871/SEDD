"""
Pressure-regulated tank and pressurant sizing for a single selected configuration row.

No argparse by design: edit the constants in CONFIG below and run:
    python analysis/tank_pressurant_sizing.py
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CONFIG (edit here)
# ---------------------------------------------------------------------------
RUN_FOLDER = "outputs/frozen_throat_750"
INPUT_RESULTS_CSV = os.path.join(RUN_FOLDER, "results.csv")
ROW_INDEX = 0  # row to size from INPUT_RESULTS_CSV

# Total feed losses from each propellant tank to chamber [bar].
# Injector drop is modeled as a fraction of chamber pressure for each line.
INJECTOR_DP_FRACTION_OF_PC = 0.25
# Additional non-injector losses by line [bar] (lines, valves, etc.).
DELTA_P_EXTRA_OX_BAR = 6.75
DELTA_P_EXTRA_FUEL_BAR = 4.75

# Total propellant mass for full mission burn [kg].
# Burn time is computed from this (not from trajectory-limited burn in other scripts).
M_PROP_TOTAL_KG = 5.0

# Propellant densities [kg/m^3]
RHO_OX_KG_M3 = 1400.0   # 98% H2O2 (approx)
RHO_FUEL_KG_M3 = 810.0  # RP-1 (approx)

# Tank ullage fraction at BOL (fraction of tank volume that is gas at start).
ULLAGE_FRAC_OX = 0.10
ULLAGE_FRAC_FUEL = 0.10

# Pressurant assumptions
PRESSURANT_NAME = "He"
R_PRESSURANT_J_KG_K = 2077.0
T_PRESSURANT_IN_TANKS_K = 298.15
T_PRESSURANT_BOTTLE_K = 298.15
P_BOTTLE_START_BAR = 300.0
REGULATOR_MIN_RATIO = 1.5

# Structural assumptions (editable, not from sweep config)
PROP_TANK_SHAPE = "cyl_hemis"  # "sphere" or "cyl_hemis"
PRESSURANT_BOTTLE_SHAPE = "sphere"  # "sphere" or "cyl_hemis"
CYL_L_OVER_D = 2.0  # used only for cyl_hemis

SF_PROP_TANK = 1.5
SF_PRESSURANT_BOTTLE = 2.0

# Generic aluminum-like placeholder until material is fixed
SIGMA_ALLOW_PROP_PA = 140e6
SIGMA_ALLOW_BOTTLE_PA = 180e6
RHO_TANK_MATERIAL_KG_M3 = 2700.0

# Output
OUTPUT_SUMMARY_CSV = os.path.join(RUN_FOLDER, "tank_pressurant_sizing_summary.csv")
OUTPUT_BOTTLE_CSV = os.path.join(RUN_FOLDER, "pressurant_bottle_options.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bar_to_pa(p_bar: float) -> float:
    return float(p_bar) * 1e5


def pa_to_bar(p_pa: float) -> float:
    return float(p_pa) / 1e5


@dataclass
class TankResult:
    name: str
    p_tank_bar: float
    mdot_kg_s: float
    m_prop_kg: float
    v_prop_m3: float
    v_tank_m3: float
    v_displaced_m3: float
    m_pressurant_needed_kg: float
    radius_m: float
    cyl_length_m: float
    thickness_cyl_m: float
    thickness_head_m: float
    thickness_used_m: float
    shell_mass_kg: float


def solve_geometry_for_volume(*, volume_m3: float, shape: str, cyl_l_over_d: float) -> tuple[float, float]:
    """
    Return (radius, cylinder_length). For sphere, cylinder_length = 0.
    """
    if volume_m3 <= 0.0:
        raise ValueError("volume_m3 must be > 0.")
    shape = shape.strip().lower()
    if shape == "sphere":
        r = (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        return r, 0.0
    if shape == "cyl_hemis":
        if cyl_l_over_d <= 0.0:
            raise ValueError("CYL_L_OVER_D must be > 0 for cyl_hemis.")
        k = float(cyl_l_over_d)
        # V = pi*r^2*L + 4/3*pi*r^3, with L = 2*k*r
        r = (volume_m3 / (math.pi * (2.0 * k + 4.0 / 3.0))) ** (1.0 / 3.0)
        l_cyl = 2.0 * k * r
        return r, l_cyl
    raise ValueError(f"Unknown shape: {shape}. Use 'sphere' or 'cyl_hemis'.")


def shell_mass_and_thickness(
    *,
    pressure_bar: float,
    volume_m3: float,
    shape: str,
    cyl_l_over_d: float,
    sigma_allow_pa: float,
    safety_factor: float,
    rho_material: float,
) -> dict[str, float]:
    """
    Thin-wall first-order sizing for pressure vessel.
    """
    if sigma_allow_pa <= 0.0:
        raise ValueError("sigma_allow_pa must be > 0.")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be > 0.")
    if rho_material <= 0.0:
        raise ValueError("rho_material must be > 0.")

    p = bar_to_pa(pressure_bar)
    r, l_cyl = solve_geometry_for_volume(volume_m3=volume_m3, shape=shape, cyl_l_over_d=cyl_l_over_d)
    shape_key = shape.strip().lower()

    if shape_key == "sphere":
        t_head = p * r * safety_factor / (2.0 * sigma_allow_pa)
        t_cyl = t_head
        t_use = t_head
        area = 4.0 * math.pi * r * r
    else:
        # Cylinder section hoop stress governs cylinder wall.
        t_cyl = p * r * safety_factor / sigma_allow_pa
        # Hemispherical heads follow sphere relation.
        t_head = p * r * safety_factor / (2.0 * sigma_allow_pa)
        t_use = max(t_cyl, t_head)
        area = 2.0 * math.pi * r * l_cyl + 4.0 * math.pi * r * r

    shell_volume = area * t_use
    shell_mass = shell_volume * rho_material
    return {
        "radius_m": r,
        "cyl_length_m": l_cyl,
        "thickness_cyl_m": t_cyl,
        "thickness_head_m": t_head,
        "thickness_used_m": t_use,
        "shell_mass_kg": shell_mass,
    }


def size_single_tank(
    *,
    name: str,
    p_tank_bar: float,
    mdot_kg_s: float,
    t_burn_s: float,
    rho_kg_m3: float,
    ullage_frac: float,
    r_pressurant: float,
    t_pressurant_k: float,
    shape: str,
    cyl_l_over_d: float,
    sigma_allow_pa: float,
    safety_factor: float,
    rho_material: float,
) -> TankResult:
    if not (0.0 < ullage_frac < 1.0):
        raise ValueError(f"{name}: ullage fraction must be in (0,1).")

    m_prop = mdot_kg_s * t_burn_s
    v_prop = m_prop / rho_kg_m3

    # Tank total volume from BOL liquid volume and ullage definition.
    # V_liquid_BOL = (1 - ullage) * V_tank
    v_tank = v_prop / (1.0 - ullage_frac)

    # In regulated mode, pressurant gas replaces expelled liquid volume.
    v_displaced = v_prop
    m_press = bar_to_pa(p_tank_bar) * v_displaced / (r_pressurant * t_pressurant_k)
    struct = shell_mass_and_thickness(
        pressure_bar=p_tank_bar,
        volume_m3=v_tank,
        shape=shape,
        cyl_l_over_d=cyl_l_over_d,
        sigma_allow_pa=sigma_allow_pa,
        safety_factor=safety_factor,
        rho_material=rho_material,
    )

    return TankResult(
        name=name,
        p_tank_bar=p_tank_bar,
        mdot_kg_s=mdot_kg_s,
        m_prop_kg=m_prop,
        v_prop_m3=v_prop,
        v_tank_m3=v_tank,
        v_displaced_m3=v_displaced,
        m_pressurant_needed_kg=m_press,
        radius_m=struct["radius_m"],
        cyl_length_m=struct["cyl_length_m"],
        thickness_cyl_m=struct["thickness_cyl_m"],
        thickness_head_m=struct["thickness_head_m"],
        thickness_used_m=struct["thickness_used_m"],
        shell_mass_kg=struct["shell_mass_kg"],
    )


def bottle_volume_from_mass(
    *,
    m_pressurant_kg: float,
    p_start_bar: float,
    p_end_bar: float,
    r_pressurant: float,
    t_bottle_k: float,
) -> float:
    p_start = bar_to_pa(p_start_bar)
    p_end = bar_to_pa(p_end_bar)
    if p_start <= p_end:
        raise ValueError("P_BOTTLE_START_BAR must be > bottle end pressure.")
    return m_pressurant_kg * r_pressurant * t_bottle_k / (p_start - p_end)


def main() -> None:
    if not os.path.isfile(INPUT_RESULTS_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_RESULTS_CSV}")

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(INPUT_RESULTS_CSV)
    if len(df) == 0:
        raise ValueError(f"No rows in input CSV: {INPUT_RESULTS_CSV}")
    if ROW_INDEX < 0 or ROW_INDEX >= len(df):
        raise IndexError(f"ROW_INDEX={ROW_INDEX} out of range [0, {len(df)-1}]")

    row = df.iloc[ROW_INDEX]
    required_cols = ["Pc_bar", "mdot_ox_kg_s", "mdot_fuel_kg_s"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_RESULTS_CSV}: {missing}")

    pc_bar = float(row["Pc_bar"])
    mdot_ox = float(row["mdot_ox_kg_s"])
    mdot_fuel = float(row["mdot_fuel_kg_s"])
    mdot_total = mdot_ox + mdot_fuel
    if mdot_total <= 0.0:
        raise ValueError("Total mass flow must be > 0.")

    t_burn_s = float(M_PROP_TOTAL_KG) / mdot_total

    delta_p_inj_bar = INJECTOR_DP_FRACTION_OF_PC * pc_bar
    p_tank_ox_bar = pc_bar + delta_p_inj_bar + float(DELTA_P_EXTRA_OX_BAR)
    p_tank_fuel_bar = pc_bar + delta_p_inj_bar + float(DELTA_P_EXTRA_FUEL_BAR)

    ox = size_single_tank(
        name="oxidizer",
        p_tank_bar=p_tank_ox_bar,
        mdot_kg_s=mdot_ox,
        t_burn_s=t_burn_s,
        rho_kg_m3=float(RHO_OX_KG_M3),
        ullage_frac=float(ULLAGE_FRAC_OX),
        r_pressurant=float(R_PRESSURANT_J_KG_K),
        t_pressurant_k=float(T_PRESSURANT_IN_TANKS_K),
        shape=PROP_TANK_SHAPE,
        cyl_l_over_d=float(CYL_L_OVER_D),
        sigma_allow_pa=float(SIGMA_ALLOW_PROP_PA),
        safety_factor=float(SF_PROP_TANK),
        rho_material=float(RHO_TANK_MATERIAL_KG_M3),
    )
    fuel = size_single_tank(
        name="fuel",
        p_tank_bar=p_tank_fuel_bar,
        mdot_kg_s=mdot_fuel,
        t_burn_s=t_burn_s,
        rho_kg_m3=float(RHO_FUEL_KG_M3),
        ullage_frac=float(ULLAGE_FRAC_FUEL),
        r_pressurant=float(R_PRESSURANT_J_KG_K),
        t_pressurant_k=float(T_PRESSURANT_IN_TANKS_K),
        shape=PROP_TANK_SHAPE,
        cyl_l_over_d=float(CYL_L_OVER_D),
        sigma_allow_pa=float(SIGMA_ALLOW_PROP_PA),
        safety_factor=float(SF_PROP_TANK),
        rho_material=float(RHO_TANK_MATERIAL_KG_M3),
    )

    p_reg_out_max_bar = max(p_tank_ox_bar, p_tank_fuel_bar)
    p_bottle_end_shared_bar = float(REGULATOR_MIN_RATIO) * p_reg_out_max_bar

    m_press_total = ox.m_pressurant_needed_kg + fuel.m_pressurant_needed_kg
    v_bottle_shared = bottle_volume_from_mass(
        m_pressurant_kg=m_press_total,
        p_start_bar=float(P_BOTTLE_START_BAR),
        p_end_bar=p_bottle_end_shared_bar,
        r_pressurant=float(R_PRESSURANT_J_KG_K),
        t_bottle_k=float(T_PRESSURANT_BOTTLE_K),
    )

    bottle_struct_shared = shell_mass_and_thickness(
        pressure_bar=float(P_BOTTLE_START_BAR),
        volume_m3=v_bottle_shared,
        shape=PRESSURANT_BOTTLE_SHAPE,
        cyl_l_over_d=float(CYL_L_OVER_D),
        sigma_allow_pa=float(SIGMA_ALLOW_BOTTLE_PA),
        safety_factor=float(SF_PRESSURANT_BOTTLE),
        rho_material=float(RHO_TANK_MATERIAL_KG_M3),
    )

    summary_rows = [
        {
            "run_folder": RUN_FOLDER,
            "row_index": ROW_INDEX,
            "Pc_bar": pc_bar,
            "delta_p_injector_bar": delta_p_inj_bar,
            "delta_p_extra_ox_bar": DELTA_P_EXTRA_OX_BAR,
            "delta_p_extra_fuel_bar": DELTA_P_EXTRA_FUEL_BAR,
            "delta_p_total_ox_bar": delta_p_inj_bar + DELTA_P_EXTRA_OX_BAR,
            "delta_p_total_fuel_bar": delta_p_inj_bar + DELTA_P_EXTRA_FUEL_BAR,
            "P_tank_ox_bar": p_tank_ox_bar,
            "P_tank_fuel_bar": p_tank_fuel_bar,
            "mdot_ox_kg_s": mdot_ox,
            "mdot_fuel_kg_s": mdot_fuel,
            "mdot_total_kg_s": mdot_total,
            "M_prop_total_kg": M_PROP_TOTAL_KG,
            "burn_time_s": t_burn_s,
            "m_ox_kg": ox.m_prop_kg,
            "m_fuel_kg": fuel.m_prop_kg,
            "V_ox_liquid_m3": ox.v_prop_m3,
            "V_fuel_liquid_m3": fuel.v_prop_m3,
            "V_tank_ox_total_m3": ox.v_tank_m3,
            "V_tank_fuel_total_m3": fuel.v_tank_m3,
            "prop_tank_shape": PROP_TANK_SHAPE,
            "tank_radius_ox_m": ox.radius_m,
            "tank_radius_fuel_m": fuel.radius_m,
            "tank_cyl_len_ox_m": ox.cyl_length_m,
            "tank_cyl_len_fuel_m": fuel.cyl_length_m,
            "tank_thickness_ox_mm": ox.thickness_used_m * 1e3,
            "tank_thickness_fuel_mm": fuel.thickness_used_m * 1e3,
            "tank_shell_mass_ox_kg": ox.shell_mass_kg,
            "tank_shell_mass_fuel_kg": fuel.shell_mass_kg,
            "m_pressurant_ox_kg": ox.m_pressurant_needed_kg,
            "m_pressurant_fuel_kg": fuel.m_pressurant_needed_kg,
            "m_pressurant_total_kg": m_press_total,
        }
    ]
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False)

    bottle_rows = [
        {
            "option": "shared_bottle_two_regulators",
            "pressurant": PRESSURANT_NAME,
            "P_bottle_start_bar": P_BOTTLE_START_BAR,
            "P_bottle_end_bar": p_bottle_end_shared_bar,
            "regulator_ratio_min": REGULATOR_MIN_RATIO,
            "V_bottle_m3": v_bottle_shared,
            "V_bottle_L": v_bottle_shared * 1000.0,
            "bottle_shape": PRESSURANT_BOTTLE_SHAPE,
            "bottle_radius_m": bottle_struct_shared["radius_m"],
            "bottle_cyl_len_m": bottle_struct_shared["cyl_length_m"],
            "bottle_thickness_mm": bottle_struct_shared["thickness_used_m"] * 1e3,
            "bottle_shell_mass_kg": bottle_struct_shared["shell_mass_kg"],
        }
    ]
    pd.DataFrame(bottle_rows).to_csv(OUTPUT_BOTTLE_CSV, index=False)

    print("Tank + pressurant sizing complete")
    print(f"Input row: {INPUT_RESULTS_CSV}  idx={ROW_INDEX}")
    print(f"Pc: {pc_bar:.3f} bar")
    print(
        f"Injector drop rule: delta_p_inj = {INJECTOR_DP_FRACTION_OF_PC:.2f} * Pc = {delta_p_inj_bar:.3f} bar"
    )
    print(f"P_tank_ox: {p_tank_ox_bar:.3f} bar, P_tank_fuel: {p_tank_fuel_bar:.3f} bar")
    print(f"Burn time (from M_prop_total/mdot_total): {t_burn_s:.3f} s")
    print(f"Prop tanks ({PROP_TANK_SHAPE}): t_ox={ox.thickness_used_m*1e3:.2f} mm, t_fuel={fuel.thickness_used_m*1e3:.2f} mm")
    print(f"Prop tank shell masses: ox={ox.shell_mass_kg:.3f} kg, fuel={fuel.shell_mass_kg:.3f} kg")
    print(f"Pressurant required: ox={ox.m_pressurant_needed_kg:.4f} kg, fuel={fuel.m_pressurant_needed_kg:.4f} kg, total={m_press_total:.4f} kg")
    print(
        f"Shared bottle ({PRESSURANT_BOTTLE_SHAPE}): {v_bottle_shared*1000.0:.3f} L, "
        f"t={bottle_struct_shared['thickness_used_m']*1e3:.2f} mm, "
        f"m_shell={bottle_struct_shared['shell_mass_kg']:.3f} kg "
        f"at Pstart={P_BOTTLE_START_BAR:.1f} bar, Pend={p_bottle_end_shared_bar:.3f} bar"
    )
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved: {OUTPUT_BOTTLE_CSV}")
    print(f"Config reference loaded from config.yaml (run_name={cfg.get('run_name', 'n/a')})")


if __name__ == "__main__":
    main()
