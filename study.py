"""
Main driver using YAML config file.
Each run automatically creates a dedicated results folder.

Run:
    python study.py
"""
from __future__ import annotations

import os
import math
import yaml
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from cea_runner import run_rocket_case
from plots import generate_all_plots


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def feasibility_from_coeffs(
    *,
    F_req_N: float,
    Pc_bar: float,
    OF: float,
    ae_at: float,
    cf: float,
    cstar_m_s: float,
    eta_cf: float = 1.0,
    eta_cstar: float = 1.0,
) -> Dict[str, float]:

    Pc_Pa = Pc_bar * 1e5

    cf_eff = eta_cf * cf
    cstar_eff = eta_cstar * cstar_m_s

    At = F_req_N / (cf_eff * Pc_Pa)
    dt = math.sqrt(4.0 * At / math.pi)

    Ae = ae_at * At
    de = math.sqrt(4.0 * Ae / math.pi)

    mdot = Pc_Pa * At / cstar_eff
    mdot_fuel = mdot / (1.0 + OF)
    mdot_ox = mdot - mdot_fuel

    return {
        "At_m2": At,
        "dt_m": dt,
        "Ae_m2": Ae,
        "de_m": de,
        "mdot_kg_s": mdot,
        "mdot_fuel_kg_s": mdot_fuel,
        "mdot_ox_kg_s": mdot_ox,
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    cfg = load_config("config.yaml")

    # === Create unique run folder ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(cfg["output_dir"], run_name)
    ensure_dir(run_dir)

    OF_list = np.linspace(cfg["OF_min"], cfg["OF_max"], cfg["OF_points"])

    rows: List[Dict[str, Any]] = []

    for Pc_bar in cfg["Pc_bar_list"]:
        for ae_at in cfg["ae_at_list"]:
            for OF in OF_list:

                r = run_rocket_case(
                    Pc_bar=float(Pc_bar),
                    OF=float(OF),
                    ae_at=float(ae_at),
                    analysis_type=cfg["analysis_type"],
                    fuel_T_K=cfg["fuel_T_K"],
                    ox_T_K=cfg["ox_T_K"],
                    h2o2_mass_frac=cfg["H2O2_mass_frac"],
                    h2o_mass_frac=cfg["H2O_mass_frac"],
                )

                r.update({
                    "F_req_N": float(cfg["F_req_N"]),
                    "eta_cf": float(cfg["eta_cf"]),
                    "eta_cstar": float(cfg["eta_cstar"]),
                })

                if r.get("cf") is not None and r.get("cstar_m_s") is not None:
                    feas = feasibility_from_coeffs(
                        F_req_N=float(cfg["F_req_N"]),
                        Pc_bar=float(Pc_bar),
                        OF=float(OF),
                        ae_at=float(ae_at),
                        cf=float(r["cf"]),
                        cstar_m_s=float(r["cstar_m_s"]),
                        eta_cf=float(cfg["eta_cf"]),
                        eta_cstar=float(cfg["eta_cstar"]),
                    )
                    r.update(feas)

                rows.append(r)

    df = pd.DataFrame(rows)

    # === Write outputs inside run folder ===
    results_path = os.path.join(run_dir, cfg["output_csv"])
    df.to_csv(results_path, index=False)

    # Summary file
    if "mdot_kg_s" in df.columns:
        summary = (
            df.dropna(subset=["mdot_kg_s"])
              .sort_values("mdot_kg_s")
              .groupby(["Pc_bar", "ae_at"], as_index=False)
              .first()[["Pc_bar", "ae_at", "OF", "Tc_K", "cf", "cstar_m_s", "ivac_s", "mdot_kg_s", "dt_m", "de_m"]]
        )
        summary.to_csv(os.path.join(run_dir, "best_by_mdot.csv"), index=False)

    # Generate plots inside run folder
    generate_all_plots(df, run_dir)

    print("Study completed.")
    print("Run folder created:", run_dir)
    print("Results file:", results_path)


if __name__ == "__main__":
    main()
