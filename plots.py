"""
Plot helpers for the RP-1 / 98% H2O2 CEA feasibility study.

All functions save plots into the given output directory.
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_tc_vs_of(df: pd.DataFrame, outdir: str) -> str:
    """
    Tc vs O/F for each Pc (Tc doesn't depend on ae/at, so we collapse over ae/at).
    """
    _ensure_dir(outdir)
    d = df.groupby(["Pc_bar", "OF"], as_index=False)["Tc_K"].max()

    plt.figure()
    for Pc, g in d.groupby("Pc_bar"):
        plt.plot(g["OF"], g["Tc_K"], label=f"Pc={Pc:g} bar")
    plt.xlabel("O/F [-]")
    plt.ylabel("Chamber temperature Tc [K]")
    plt.title("Tc vs O/F (98% H2O2 / RP-1)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, "Tc_vs_OF.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_cstar_vs_of(df: pd.DataFrame, outdir: str) -> str:
    """
    c* vs O/F for each Pc (collapse over ae/at).
    """
    _ensure_dir(outdir)
    d = df.groupby(["Pc_bar", "OF"], as_index=False)["cstar_m_s"].max()

    plt.figure()
    for Pc, g in d.groupby("Pc_bar"):
        plt.plot(g["OF"], g["cstar_m_s"], label=f"Pc={Pc:g} bar")
    plt.xlabel("O/F [-]")
    plt.ylabel("Characteristic velocity c* [m/s]")
    plt.title("c* vs O/F (98% H2O2 / RP-1)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, "cstar_vs_OF.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_mdot_vs_of_by_pc(df: pd.DataFrame, outdir: str) -> list[str]:
    """
    Required mdot vs O/F, one figure per Pc, with curves for each ae/at.
    """
    _ensure_dir(outdir)
    paths = []

    for Pc, gPc in df.dropna(subset=["mdot_kg_s"]).groupby("Pc_bar"):
        plt.figure()
        for ae, g in gPc.groupby("ae_at"):
            plt.plot(g["OF"], g["mdot_kg_s"], label=f"ae/at={ae:g}")
        plt.xlabel("O/F [-]")
        plt.ylabel("Required total mass flow mdot [kg/s]")
        plt.title(f"Required mdot vs O/F (Pc={Pc:g} bar)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()

        path = os.path.join(outdir, f"mdot_vs_OF_Pc_{Pc:g}bar.png")
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    return paths


def plot_exit_diameter_vs_of_by_pc(df: pd.DataFrame, outdir: str) -> list[str]:
    """
    Exit diameter vs O/F, one figure per Pc, with curves for each ae/at.
    """
    _ensure_dir(outdir)
    paths = []

    for Pc, gPc in df.dropna(subset=["de_m"]).groupby("Pc_bar"):
        plt.figure()
        for ae, g in gPc.groupby("ae_at"):
            plt.plot(g["OF"], g["de_m"] * 1000.0, label=f"ae/at={ae:g}")  # mm
        plt.xlabel("O/F [-]")
        plt.ylabel("Exit diameter de [mm]")
        plt.title(f"Exit diameter vs O/F (Pc={Pc:g} bar)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()

        path = os.path.join(outdir, f"de_vs_OF_Pc_{Pc:g}bar.png")
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    return paths


def generate_all_plots(df: pd.DataFrame, outdir: str) -> dict[str, object]:
    """
    Generate the full standard plot set.
    Returns a dict of plot names -> file paths.
    """
    outputs = {
        "Tc_vs_OF": plot_tc_vs_of(df, outdir),
        "cstar_vs_OF": plot_cstar_vs_of(df, outdir),
        "mdot_vs_OF_by_Pc": plot_mdot_vs_of_by_pc(df, outdir),
        "de_vs_OF_by_Pc": plot_exit_diameter_vs_of_by_pc(df, outdir),
    }
    return outputs
