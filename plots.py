"""
Plot helpers for the RP-1 / 98% H2O2 CEA feasibility study.

All functions save plots into the given output directory.

Supports two nozzle modes:
  - 'ae_at'  : fixed area ratios swept; curves are grouped by ae/at value.
  - 'pip'    : CEA solves for ae/at to give ideal sea-level expansion;
               one curve per Pc (ae/at varies continuously with O/F).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Consistent style ────────────────────────────────────────────────────────
FIGSIZE = (7, 4.5)
DPI = 200


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _nozzle_mode(df: pd.DataFrame) -> str:
    """Detect which nozzle mode was used from the DataFrame."""
    if "nozzle_mode" in df.columns:
        modes = df["nozzle_mode"].dropna().unique()
        if len(modes) == 1:
            return str(modes[0]).strip().lower()
    return "ae_at"  # safe default


# ── Helpers shared by both modes ─────────────────────────────────────────────

def plot_tc_vs_of(df: pd.DataFrame, outdir: str) -> str:
    """Tc vs O/F for each Pc — independent of nozzle mode."""
    _ensure_dir(outdir)
    d = df.dropna(subset=["Tc_K"]).groupby(["Pc_bar", "OF"], as_index=False)["Tc_K"].max()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for Pc, g in d.groupby("Pc_bar"):
        ax.plot(g["OF"], g["Tc_K"], label=f"Pc = {Pc:g} bar")
    ax.set_xlabel("O/F  [–]")
    ax.set_ylabel("Chamber temperature  Tc  [K]")
    ax.set_title("Chamber temperature vs O/F\n(98 % H₂O₂ / RP-1)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, "Tc_vs_OF.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_cstar_vs_of(df: pd.DataFrame, outdir: str) -> str:
    """c* vs O/F for each Pc — independent of nozzle mode."""
    _ensure_dir(outdir)
    d = df.dropna(subset=["cstar_m_s"]).groupby(["Pc_bar", "OF"], as_index=False)["cstar_m_s"].max()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for Pc, g in d.groupby("Pc_bar"):
        ax.plot(g["OF"], g["cstar_m_s"], label=f"Pc = {Pc:g} bar")
    ax.set_xlabel("O/F  [–]")
    ax.set_ylabel("Characteristic velocity  c*  [m/s]")
    ax.set_title("c* vs O/F\n(98 % H₂O₂ / RP-1)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, "cstar_vs_OF.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ── ae_at mode ───────────────────────────────────────────────────────────────

def _plot_quantity_aeat_mode(
    df: pd.DataFrame,
    outdir: str,
    col: str,
    ylabel: str,
    title_template: str,   # e.g. "Required mdot vs O/F  (Pc = {Pc:g} bar)"
    fname_template: str,   # e.g. "mdot_vs_OF_Pc_{Pc:g}bar.png"
    scale: float = 1.0,    # multiply column by this (e.g. 1000 for m→mm)
) -> list[str]:
    """Generic per-Pc plot with one curve per ae/at (ae_at mode)."""
    _ensure_dir(outdir)
    paths = []

    for Pc, gPc in df.dropna(subset=[col]).groupby("Pc_bar"):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for ae, g in gPc.groupby("ae_at"):
            ax.plot(g["OF"], g[col] * scale, label=f"Aₑ/Aₜ = {ae:g}")
        ax.set_xlabel("O/F  [–]")
        ax.set_ylabel(ylabel)
        ax.set_title(title_template.format(Pc=Pc))
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        path = os.path.join(outdir, fname_template.format(Pc=Pc))
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        paths.append(path)

    return paths


# ── pip mode ─────────────────────────────────────────────────────────────────

def _plot_quantity_pip_mode(
    df: pd.DataFrame,
    outdir: str,
    col: str,
    ylabel: str,
    title: str,
    fname: str,
    scale: float = 1.0,
) -> str:
    """
    Single plot with one curve per Pc (pip mode).
    ae/at varies continuously with O/F so we never group by it.
    """
    _ensure_dir(outdir)
    d = df.dropna(subset=[col])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for Pc, g in d.groupby("Pc_bar"):
        g_sorted = g.sort_values("OF")
        ax.plot(g_sorted["OF"], g_sorted[col] * scale, label=f"Pc = {Pc:g} bar")
    ax.set_xlabel("O/F  [–]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_aeat_vs_of_pip(df: pd.DataFrame, outdir: str) -> str:
    """
    pip mode only: show the CEA-solved Ae/At vs O/F for each Pc.
    Useful sanity check — confirms what expansion ratio you're actually getting.
    """
    return _plot_quantity_pip_mode(
        df=df.dropna(subset=["ae_at"]),
        outdir=outdir,
        col="ae_at",
        ylabel="Nozzle area ratio  Aₑ/Aₜ  [–]",
        title="CEA-solved area ratio vs O/F\n(sea-level ideal expansion)",
        fname="aeat_vs_OF_pip.png",
    )


# ── Mode-dispatching wrappers ────────────────────────────────────────────────

def plot_mdot_vs_of(df: pd.DataFrame, outdir: str) -> list[str] | str:
    mode = _nozzle_mode(df)
    if mode == "pip":
        return _plot_quantity_pip_mode(
            df=df.dropna(subset=["mdot_kg_s"]),
            outdir=outdir,
            col="mdot_kg_s",
            ylabel="Required mass flow  ṁ  [kg/s]",
            title="Required mass flow vs O/F\n(sea-level ideal expansion)",
            fname="mdot_vs_OF_pip.png",
        )
    else:
        return _plot_quantity_aeat_mode(
            df=df,
            outdir=outdir,
            col="mdot_kg_s",
            ylabel="Required mass flow  ṁ  [kg/s]",
            title_template="Required mass flow vs O/F\n(Pc = {Pc:g} bar)",
            fname_template="mdot_vs_OF_Pc_{Pc:g}bar.png",
        )


def plot_exit_diameter_vs_of(df: pd.DataFrame, outdir: str) -> list[str] | str:
    mode = _nozzle_mode(df)
    if mode == "pip":
        return _plot_quantity_pip_mode(
            df=df.dropna(subset=["de_m"]),
            outdir=outdir,
            col="de_m",
            ylabel="Exit diameter  dₑ  [mm]",
            title="Exit diameter vs O/F\n(sea-level ideal expansion)",
            fname="de_vs_OF_pip.png",
            scale=1000.0,
        )
    else:
        return _plot_quantity_aeat_mode(
            df=df,
            outdir=outdir,
            col="de_m",
            ylabel="Exit diameter  dₑ  [mm]",
            title_template="Exit diameter vs O/F\n(Pc = {Pc:g} bar)",
            fname_template="de_vs_OF_Pc_{Pc:g}bar.png",
            scale=1000.0,
        )


def plot_throat_diameter_vs_of(df: pd.DataFrame, outdir: str) -> list[str] | str:
    """Throat diameter — useful in pip mode where At is the primary sizing output."""
    mode = _nozzle_mode(df)
    if mode == "pip":
        return _plot_quantity_pip_mode(
            df=df.dropna(subset=["dt_m"]),
            outdir=outdir,
            col="dt_m",
            ylabel="Throat diameter  dₜ  [mm]",
            title="Throat diameter vs O/F\n(sea-level ideal expansion)",
            fname="dt_vs_OF_pip.png",
            scale=1000.0,
        )
    else:
        return _plot_quantity_aeat_mode(
            df=df,
            outdir=outdir,
            col="dt_m",
            ylabel="Throat diameter  dₜ  [mm]",
            title_template="Throat diameter vs O/F\n(Pc = {Pc:g} bar)",
            fname_template="dt_vs_OF_Pc_{Pc:g}bar.png",
            scale=1000.0,
        )


def plot_isp_vs_of(df: pd.DataFrame, outdir: str) -> list[str] | str:
    """Delivered Isp (vacuum) vs O/F."""
    mode = _nozzle_mode(df)
    if mode == "pip":
        return _plot_quantity_pip_mode(
            df=df.dropna(subset=["isp_s"]),
            outdir=outdir,
            col="isp_s",
            ylabel="Specific impulse  Isp  [s]",
            title="Isp vs O/F\n(sea-level ideal expansion)",
            fname="isp_vs_OF_pip.png",
        )
    else:
        return _plot_quantity_aeat_mode(
            df=df,
            outdir=outdir,
            col="isp_s",
            ylabel="Specific impulse  Isp  [s]",
            title_template="Isp vs O/F\n(Pc = {Pc:g} bar)",
            fname_template="isp_vs_OF_Pc_{Pc:g}bar.png",
        )


# ── Master entry point ────────────────────────────────────────────────────────

def generate_all_plots(df: pd.DataFrame, outdir: str) -> dict[str, object]:
    """
    Generate the full standard plot set.
    Automatically adapts to 'pip' or 'ae_at' nozzle mode.
    Returns a dict of plot names -> file path(s).
    """
    mode = _nozzle_mode(df)

    outputs: dict[str, object] = {
        "Tc_vs_OF":        plot_tc_vs_of(df, outdir),
        "cstar_vs_OF":     plot_cstar_vs_of(df, outdir),
        "mdot_vs_OF":      plot_mdot_vs_of(df, outdir),
        "de_vs_OF":        plot_exit_diameter_vs_of(df, outdir),
        "dt_vs_OF":        plot_throat_diameter_vs_of(df, outdir),
        "isp_vs_OF":       plot_isp_vs_of(df, outdir),
    }

    if mode == "pip":
        outputs["aeat_vs_OF"] = plot_aeat_vs_of_pip(df, outdir)

    return outputs