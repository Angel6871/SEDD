"""
Refactored study driver using YAML config file.
Keeps behavior aligned with study.py but is structured for readability.

Run:
    python study_new.py
"""
from __future__ import annotations

import os
import math
import yaml
from datetime import datetime

import numpy as np
import pandas as pd

from cea_runner import run_rocket_case
from plots import generate_all_plots


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    required = [
        "F_req_N", "OF_min", "OF_max", "OF_points",
        "analysis_type", "fuel_T_K", "ox_T_K", "H2O2_mass_frac",
        "H2O_mass_frac", "eta_cstar", "eta_cf", "output_csv", "output_dir",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    has_pc_range = all(k in cfg for k in ("Pc_min", "Pc_max", "Pc_points"))
    has_pc_list = "Pc_bar_list" in cfg and bool(cfg["Pc_bar_list"])

    if not has_pc_range and not has_pc_list:
        raise ValueError(
            "Define either Pc_min/Pc_max/Pc_points for a continuous sweep "
            "or Pc_bar_list for explicit pressure points."
        )

    if has_pc_range:
        if float(cfg["Pc_min"]) >= float(cfg["Pc_max"]):
            raise ValueError("Pc_min must be < Pc_max.")
        if int(cfg["Pc_points"]) < 2:
            raise ValueError("Pc_points must be >= 2.")

    if float(cfg["OF_min"]) >= float(cfg["OF_max"]):
        raise ValueError("OF_min must be < OF_max.")

    if int(cfg["OF_points"]) < 2:
        raise ValueError("OF_points must be >= 2.")

    if "Tc_max_K" in cfg and cfg["Tc_max_K"] is not None:
        if float(cfg["Tc_max_K"]) <= 0.0:
            raise ValueError("Tc_max_K must be > 0 when provided.")

    if abs(float(cfg["H2O2_mass_frac"]) + float(cfg["H2O_mass_frac"]) - 1.0) > 1e-9:
        raise ValueError("H2O2_mass_frac + H2O_mass_frac must equal 1.0.")

    analysis_type = str(cfg["analysis_type"]).strip().lower()
    if analysis_type not in {"equilibrium", "frozen"}:
        raise ValueError("analysis_type must be 'equilibrium' or 'frozen'.")

    nozzle_mode = str(cfg.get("nozzle_mode", "ae_at")).strip().lower()
    if nozzle_mode not in {"pip", "ae_at"}:
        raise ValueError("nozzle_mode must be 'pip' or 'ae_at'.")

    if nozzle_mode == "pip":
        if "Pe_bar" not in cfg:
            raise ValueError("Pe_bar is required when nozzle_mode='pip'.")
        if float(cfg["Pe_bar"]) <= 0.0:
            raise ValueError("Pe_bar must be > 0.")
    else:
        if "ae_at_list" not in cfg or not cfg["ae_at_list"]:
            raise ValueError("ae_at_list is required and non-empty when nozzle_mode='ae_at'.")


def reaction_mode_from_config(cfg: dict) -> str:
    analysis_type = str(cfg.get("analysis_type", "")).strip().lower()
    if analysis_type == "equilibrium":
        return "fulleq"
    if analysis_type != "frozen":
        return analysis_type or "unknown"

    nfz = cfg.get("nfz")
    try:
        nfz_i = int(nfz) if nfz is not None else None
    except (TypeError, ValueError):
        nfz_i = None

    if nfz_i == 1:
        return "frozen chamber"
    if nfz_i == 2:
        return "frozen throat"
    return "frozen"


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
) -> dict:
    """
    Convert CEA coefficients (cf, cstar) into geometric + flow requirements.

    Equations used:
      At = F / ( (eta_cf*cf) * Pc )
      mdot = Pc*At / (eta_cstar*cstar)

    Units:
      Pc_bar -> converted to Pa internally.
    """
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


def build_pc_list(cfg: dict) -> np.ndarray:
    if all(k in cfg for k in ("Pc_min", "Pc_max", "Pc_points")):
        return np.linspace(float(cfg["Pc_min"]), float(cfg["Pc_max"]), int(cfg["Pc_points"]))
    return np.asarray(cfg["Pc_bar_list"], dtype=float)


def build_nozzle_cases(cfg: dict, Pc_bar: float) -> list[tuple[float | None, float | None]]:
    nozzle_mode = str(cfg.get("nozzle_mode", "ae_at")).strip().lower()
    if nozzle_mode == "pip":
        pe_bar = float(cfg.get("Pe_bar", 1.01325))
        return [(None, float(Pc_bar) / pe_bar)]
    return [(float(ae), None) for ae in cfg["ae_at_list"]]


def run_single_case(cfg: dict, Pc_bar: float, OF: float, ae_at: float | None, pip: float | None) -> dict:
    nozzle_mode = str(cfg.get("nozzle_mode", "ae_at")).strip().lower()
    pe_bar = float(cfg.get("Pe_bar", 1.01325))

    r = run_rocket_case(
        Pc_bar=float(Pc_bar),
        OF=float(OF),
        ae_at=ae_at,
        pip=pip,
        analysis_type=cfg["analysis_type"],
        nfz=cfg.get("nfz"),
        custom_nfz=cfg.get("custom_nfz"),
        fuel_T_K=cfg["fuel_T_K"],
        ox_T_K=cfg["ox_T_K"],
        h2o2_mass_frac=cfg["H2O2_mass_frac"],
        h2o_mass_frac=cfg["H2O_mass_frac"],
    )

    r.update({
        "F_req_N": float(cfg["F_req_N"]),
        "eta_cf": float(cfg["eta_cf"]),
        "eta_cstar": float(cfg["eta_cstar"]),
        "nozzle_mode": nozzle_mode,
        "Pe_design_bar": (pe_bar if nozzle_mode == "pip" else None),
        "reaction_mode": reaction_mode_from_config(cfg),
    })

    # CEA gives cf and c*; sizing then computes At, Ae, and mdot.
    if r.get("cf") is not None and r.get("cstar_m_s") is not None and r.get("ae_at") is not None:
        r.update(
            feasibility_from_coeffs(
                F_req_N=float(cfg["F_req_N"]),
                Pc_bar=float(Pc_bar),
                OF=float(OF),
                ae_at=float(r["ae_at"]),
                cf=float(r["cf"]),
                cstar_m_s=float(r["cstar_m_s"]),
                eta_cf=float(cfg["eta_cf"]),
                eta_cstar=float(cfg["eta_cstar"]),
            )
        )

    return r


def write_outputs(df: pd.DataFrame, cfg: dict, run_dir: str) -> str:
    results_path = os.path.join(run_dir, cfg["output_csv"])
    df.to_csv(results_path, index=False)

    nozzle_mode = str(cfg.get("nozzle_mode", "ae_at")).strip().lower()
    if "mdot_kg_s" in df.columns:
        best_pool = df.dropna(subset=["mdot_kg_s"]).copy()
        tc_max_k = cfg.get("Tc_max_K")
        if tc_max_k is not None:
            tc_max_k = float(tc_max_k)
            if "Tc_K" not in best_pool.columns:
                raise ValueError("Tc_max_K is set, but Tc_K column is missing from results.")
            best_pool["Tc_K"] = pd.to_numeric(best_pool["Tc_K"], errors="coerce")
            best_pool = best_pool[best_pool["Tc_K"] <= tc_max_k]

        group_cols = ["Pc_bar"] if nozzle_mode == "pip" else ["Pc_bar", "ae_at"]
        keep = [
            "Pc_bar", "ae_at", "Pe_bar", "pip", "OF", "Tc_K", "cf",
            "cstar_m_s", "ivac_s", "mdot_kg_s", "dt_m", "de_m",
        ]
        if len(best_pool) == 0:
            empty_cols = [c for c in keep if c in df.columns]
            pd.DataFrame(columns=empty_cols).to_csv(os.path.join(run_dir, "best_by_mdot.csv"), index=False)
            print("Warning: no rows satisfy Tc_max_K; wrote empty best_by_mdot.csv")
        else:
            summary = (
                best_pool.sort_values("mdot_kg_s")
                .groupby(group_cols, as_index=False)
                .first()
            )
            keep = [c for c in keep if c in summary.columns]
            summary[keep].to_csv(os.path.join(run_dir, "best_by_mdot.csv"), index=False)

    generate_all_plots(df, run_dir)
    return results_path


def main() -> None:
    cfg = load_config("config.yaml")
    validate_config(cfg)

    nozzle_mode = str(cfg.get("nozzle_mode", "ae_at")).strip().lower()
    reaction_mode = reaction_mode_from_config(cfg)
    Pc_list = build_pc_list(cfg)
    OF_list = np.linspace(cfg["OF_min"], cfg["OF_max"], cfg["OF_points"])

    case_count = 0
    for Pc_bar in Pc_list:
        case_count += len(build_nozzle_cases(cfg, float(Pc_bar))) * len(OF_list)

    pc_sweep_text = (
        f"[{float(cfg['Pc_min'])}, {float(cfg['Pc_max'])}], points={int(cfg['Pc_points'])}"
        if all(k in cfg for k in ("Pc_min", "Pc_max", "Pc_points"))
        else f"list={len(Pc_list)} points"
    )

    print(
        "Config sweep:",
        f"mode={nozzle_mode}",
        f"reaction={reaction_mode}",
        f"Pc={pc_sweep_text}",
        f"OF=[{cfg['OF_min']}, {cfg['OF_max']}], points={cfg['OF_points']}",
        f"cases={case_count}",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    configured_run_name = str(cfg.get("run_name", "")).strip()
    run_name = configured_run_name or f"run_{timestamp}"
    run_dir = os.path.join(cfg["output_dir"], run_name)
    ensure_dir(run_dir)

    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    rows = []
    for Pc_bar in Pc_list:
        for ae_at, pip in build_nozzle_cases(cfg, float(Pc_bar)):
            for OF in OF_list:
                rows.append(run_single_case(cfg, float(Pc_bar), float(OF), ae_at, pip))

    df = pd.DataFrame(rows)
    results_path = write_outputs(df, cfg, run_dir)

    print("Study completed.")
    print("Run folder created:", run_dir)
    print("Results file:", results_path)


if __name__ == "__main__":
    main()
