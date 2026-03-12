"""
Regenerative Cooling Channel Design Tool
RP-1 / H2O2 Bipropellant Thruster — Inconel 718 Wall

Methodology:
  - Cervone AE4903 Lecture 7 (TU Delft): 1D thermal resistance network
  - Fagherazzi et al. Aerospace 2023: calibrated Nusselt for H2O2/hydrocarbon
  - Isentropic relations for axial gas conditions
  - Counter-flow coolant energy balance

Constraints checked:
  1. T_hot_wall  < T_wall_limit   (Inconel 718 hot-side wall)
  2. T_cold_wall < T_coking       (RP-1 coking on coolant-side wall)
  3. T_coolant   < T_bulk_limit   (coolant bulk temperature)
  4. Re_coolant  > Re_min         (Dittus-Boelter validity)

Pressure drop is NOT computed in this version.

====================================================================
  *** INPUTS TO UPDATE WHEN BETTER VALUES ARE AVAILABLE ***
  Search for the tag  # [UPDATE]  throughout the file
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq


# ====================================================================
# SECTION 1 — ENGINE / CEA PARAMETERS
# ====================================================================
# [UPDATE] Replace with final CEA output values

Pc          = 11.06e5        # Chamber pressure [Pa]
Tc          = 1973.97        # Adiabatic flame temperature [K]
gamma       = 1.2741          # Specific heat ratio [-]
Pr_g        = 0.60377          # Gas Prandtl number [-]
mu_g        = 4.8359e-5        # Gas dynamic viscosity [Pa·s] (visc)
cp_g        = 2236.2        # Gas specific heat [J/kg·K]
k_g         = mu_g * cp_g / Pr_g   # Gas thermal conductivity [W/m·K]
                                     # (derived from mu, cp, Pr for consistency)

mdot_total  = 0.5808          # Total propellant mass flow rate [kg/s]

# Calibrated Nusselt coefficient for hot gas side
# From Fagherazzi et al. 2023, eq.(19): Nu = C * Re^0.8 * Pr^0.4
# Calibrated for H2O2 / hydrocarbon bi-propellant: C = 0.0296
# This replaces the Bartz equation with an empirically-calibrated equivalent.
# [UPDATE] Can adjust C if you have experimental data for your specific engine
C_nusselt   = 0.0296


# ====================================================================
# SECTION 2 — COOLANT PROPERTIES (RP-1, liquid phase)
# ====================================================================
# [UPDATE] Properties are currently averaged constants.
#          Replace with temperature-dependent values if available, so these values will need to be found better

# O/F = 4.11 => fuel mass fraction = 1 / (1 + O/F)
OF_ratio        = 3
mdot_coolant    = mdot_total / (1.0 + OF_ratio)  # [kg/s]

T_coolant_in    = 298.0     # Coolant inlet temperature [K]
                             # (enters at nozzle exit in counter-flow)

rho_c           = 773.0     # Density [kg/m³]
mu_c            = 0.0021    # Dynamic viscosity [Pa·s]
k_c             = 0.145     # Thermal conductivity [W/m·K]
cp_c            = 1950.0    # Specific heat [J/kg·K]
Pr_c            = cp_c * mu_c / k_c    # Prandtl number [-]


# ====================================================================
# SECTION 3 — WALL MATERIAL (Inconel 718)
# ====================================================================
# [UPDATE] t_wall is a manufacturing/design choice — update when fixed.
#          k_wall range for Inconel 718: 11–16 W/m·K depending on temperature.

t_wall          = 0.001     # Inner wall thickness [m]  (1 mm)
k_wall          = 13.0      # Thermal conductivity [W/m·K]


# ====================================================================
# SECTION 4 — TEMPERATURE CONSTRAINTS
# ====================================================================
# [UPDATE] Adjust limits when final material specs and RP-1 conditions are known.

T_wall_limit    = 1573.0    # Inconel 718 max service temp (~1300 C) [K]

T_coking        = 700.0     # RP-1 coking limit on cold-wall surface [K]
                             # (~430 C = 703 K; conservative for turbulent flow)

T_bulk_limit    = 620.0     # RP-1 bulk temperature limit [K]
                             # At high pressure (Pc ~ 80 bar), RP-1 stays liquid
                             # well above the 1-atm boiling point (~490 K).
                             # [UPDATE] Verify with RP-1 phase diagram at your Pc.

Re_min          = 10000     # Minimum coolant Re for Dittus-Boelter validity


# ====================================================================
# SECTION 5 — NOZZLE GEOMETRY
# ====================================================================
# [UPDATE] Replace with final nozzle design dimensions.

D_t     = 2*0.00962             # Throat diameter [m]
D_c     = 0.06668             # Chamber diameter [m]
AR_exit = 2.399             # Exit area ratio A_exit / A_throat [-]
L_noz   = 0.04107+0.02125             # Nozzle (diverging section) length [m]
L_ch    = 0.13335             # Chamber (cylindrical section) length [m]  [UPDATE]


# Derived geometry
R_t     = D_t / 2
R_c     = D_c / 2
R_exit  = R_t * np.sqrt(AR_exit)
AR_c    = (R_c / R_t)**2


# ====================================================================
# SECTION 6 — CHANNEL OPTIMIZER SEARCH RANGE
# ====================================================================
# [UPDATE] Adjust based on manufacturing constraints (min feature size, etc.)

N_ch_min    = 10
N_ch_max    = 60
N_ch_step   = 2

s_min       = 0.0002        # Min channel side [m]  (0.2 mm)
s_max       = 0.0012        # Max channel side [m]  (1.2 mm)
s_steps     = 60


# ====================================================================
# GEOMETRY BUILDER
# ====================================================================

def generate_contour(N=300):
    """
    Smooth converging-diverging contour.
      - Chamber  : cylindrical at R_c
      - Converging: cosine blend from R_c to R_t over L_noz
      - Diverging : conical from R_t to R_exit over L_noz
    Throat at x = 0.

    [UPDATE] Replace body with:  return x_data, r_data
             when actual contour coordinates are available.
    """
    N_ch_pts = N // 4
    N_cv_pts = N // 4
    N_dv_pts = N // 2

    x_ch = np.linspace(-(L_ch + L_noz), -L_noz, N_ch_pts)
    r_ch = np.full(N_ch_pts, R_c)

    x_cv = np.linspace(-L_noz, 0.0, N_cv_pts)
    t    = (x_cv - x_cv[0]) / (x_cv[-1] - x_cv[0])
    r_cv = R_c + (R_t - R_c) * (1 - np.cos(np.pi * t)) / 2.0

    x_dv = np.linspace(0.0, L_noz, N_dv_pts)
    r_dv = R_t + (R_exit - R_t) * (x_dv / L_noz)

    x      = np.concatenate([x_ch,  x_cv[1:],  x_dv[1:]])
    radius = np.concatenate([r_ch,  r_cv[1:],  r_dv[1:]])
    return x, radius


# ====================================================================
# ISENTROPIC FLOW
# ====================================================================

def mach_from_area_ratio(AR, supersonic, gam=gamma):
    """Numerically solve A/A* = f(M)."""
    def eq(M):
        return (1/M)*((2/(gam+1))*(1+(gam-1)/2*M**2))**((gam+1)/(2*(gam-1))) - AR
    if AR <= 1.001:
        return 1.0
    return brentq(eq, 1.001, 20.0) if supersonic else brentq(eq, 0.001, 0.999)


def local_gas_conditions(radius):
    """
    Local Mach, static temperature, and adiabatic wall temperature at each station.
    T_aw = T_static + Pr^(1/3) * (T_total - T_static)   [Fagherazzi eq.15]
    """
    i_t = np.argmin(radius)
    Rt  = radius[i_t]
    M   = np.zeros(len(radius))
    Tst = np.zeros(len(radius))
    Taw = np.zeros(len(radius))
    for i, r in enumerate(radius):
        Mi      = mach_from_area_ratio((r/Rt)**2, i > i_t)
        M[i]    = Mi
        Tst[i]  = Tc / (1 + (gamma-1)/2 * Mi**2)
        Taw[i]  = Tst[i] + Pr_g**(1/3) * (Tc - Tst[i])
    return M, Tst, Taw


# ====================================================================
# HOT GAS HEAT TRANSFER COEFFICIENT
# ====================================================================

def hot_gas_htc(radius):
    """
    Calibrated Nusselt: Nu = C * Re^0.8 * Pr^0.4   (Fagherazzi 2023 eq.19)
    h_g = Nu * k_g / D_local
    G = mdot_total / A_local varies axially → correct throat peak.
    """
    h_g = np.zeros(len(radius))
    for i, r in enumerate(radius):
        D   = 2 * r
        G   = mdot_total / (np.pi * r**2)
        Re  = G * D / mu_g
        Nu  = C_nusselt * Re**0.8 * Pr_g**0.4
        h_g[i] = Nu * k_g / D
    return h_g


# ====================================================================
# COOLANT HEAT TRANSFER COEFFICIENT
# ====================================================================

def coolant_htc(s, N_ch):
    """
    Dittus-Boelter for square channels (Cervone slide 25):
    Nu = 0.023 * Re^0.8 * Pr^0.4,  Dh = s (square channel).
    Returns h_c [W/m²K], Re [-], V [m/s].
    """
    A_total = N_ch * s**2
    V       = mdot_coolant / (rho_c * A_total)
    Re      = rho_c * V * s / mu_c
    Nu      = 0.023 * Re**0.8 * Pr_c**0.4
    h_c     = Nu * k_c / s
    return h_c, Re, V


# ====================================================================
# 1D THERMAL RESISTANCE SOLVER
# ====================================================================

def solve_thermal(x, radius, h_g, h_c, T_aw):
    """
    At each axial station (Cervone §7.2):

        q = (T_aw - T_coolant) / (1/h_g + t_wall/k_wall + 1/h_c)

        T_hot_wall  = T_aw      - q / h_g   → vs T_wall_limit
        T_cold_wall = T_coolant + q / h_c   → vs T_coking

    Counter-flow: coolant enters at exit end (index N-1), flows toward
    chamber (decreasing index), heating up along the way.
    """
    N           = len(x)
    T_coolant   = np.zeros(N)
    T_hot_wall  = np.zeros(N)
    T_cold_wall = np.zeros(N)
    heat_flux   = np.zeros(N)

    T_coolant[-1] = T_coolant_in

    for i in range(N-1, 0, -1):
        R_tot           = 1/h_g[i] + t_wall/k_wall + 1/h_c
        q               = (T_aw[i] - T_coolant[i]) / R_tot
        heat_flux[i]    = q
        T_hot_wall[i]   = T_aw[i]      - q / h_g[i]
        T_cold_wall[i]  = T_coolant[i] + q / h_c
        perim           = 2 * np.pi * radius[i]
        dx              = abs(x[i] - x[i-1])
        T_coolant[i-1]  = T_coolant[i] + q * perim * dx / (mdot_coolant * cp_c)

    R_tot           = 1/h_g[0] + t_wall/k_wall + 1/h_c
    q               = (T_aw[0] - T_coolant[0]) / R_tot
    heat_flux[0]    = q
    T_hot_wall[0]   = T_aw[0]      - q / h_g[0]
    T_cold_wall[0]  = T_coolant[0] + q / h_c

    return T_hot_wall, T_cold_wall, T_coolant, heat_flux


# ====================================================================
# ENERGY BALANCE DIAGNOSTIC
# ====================================================================

def energy_balance_report(x, radius, h_g, T_aw):
    """
    Estimates total heat load vs coolant capacity before the optimizer runs.
    Flags immediately if the coolant flow is fundamentally insufficient.
    """
    Q_total = 0.0
    for i in range(len(x) - 1):
        R_min   = 1/h_g[i] + t_wall/k_wall     # best case: h_c -> infinity
        q       = (T_aw[i] - T_coolant_in) / R_min
        perim   = 2 * np.pi * radius[i]
        dx      = abs(x[i+1] - x[i])
        Q_total += q * perim * dx

    Q_max = mdot_coolant * cp_c * (T_bulk_limit - T_coolant_in)
    ratio = Q_total / Q_max

    print(f"\n  --- Energy Balance Diagnostic ---")
    print(f"  Total heat load (approx)         : {Q_total/1000:.1f} kW")
    print(f"  Coolant capacity (to {T_bulk_limit:.0f} K)  : {Q_max/1000:.1f} kW")
    print(f"  Load / capacity ratio            : {ratio:.2f}")
    if ratio > 1.0:
        print(f"  ⚠  Coolant flow insufficient by {ratio:.1f}x")
        print(f"     No fully valid design is possible with current parameters.")
        print(f"     Options: shorten cooled length, add film cooling,")
        print(f"     raise T_bulk_limit (verify RP-1 stays liquid at your Pc),")
        print(f"     or reconsider O/F ratio.")
    else:
        print(f"  ✓  Coolant flow sufficient — valid designs should exist.")


# ====================================================================
# CHANNEL OPTIMIZER
# ====================================================================

def optimize_channels(x, radius, h_g, T_aw):
    """
    Sweep (N_channels, channel_side s). Records all fully valid designs
    and the single best design (lowest peak T_cold_wall) regardless.
    """
    N_range = range(N_ch_min, N_ch_max + 1, N_ch_step)
    s_range = np.linspace(s_min, s_max, s_steps)
    total   = len(N_range) * len(s_range)

    print(f"\n  Running optimizer over {total} combinations...")

    valid_designs   = []
    best_score      = 1e12
    best_design     = None

    for N_ch in N_range:
        for s in s_range:

            h_c, Re, V = coolant_htc(s, N_ch)
            if Re < Re_min:
                continue

            T_hw, T_cw, T_cb, q = solve_thermal(x, radius, h_g, h_c, T_aw)

            result = {
                'N_ch'        : N_ch,
                's_mm'        : s * 1000,
                'Re'          : Re,
                'V'           : V,
                'h_c'         : h_c,
                'max_T_hw'    : np.max(T_hw),
                'max_T_cw'    : np.max(T_cw),
                'max_T_cb'    : np.max(T_cb),
                'T_cb_outlet' : T_cb[0],
                'mg_wall'     : T_wall_limit - np.max(T_hw),
                'mg_cok'      : T_coking     - np.max(T_cw),
                'T_hw_arr'    : T_hw,
                'T_cw_arr'    : T_cw,
                'T_cb_arr'    : T_cb,
                'q_arr'       : q,
            }

            ok = (np.max(T_hw) < T_wall_limit and
                  np.max(T_cw) < T_coking     and
                  np.max(T_cb) < T_bulk_limit)

            if ok:
                valid_designs.append(result)

            if np.max(T_cw) < best_score:
                best_score  = np.max(T_cw)
                best_design = result

    valid_designs.sort(key=lambda d: -d['mg_cok'])

    print(f"  Valid designs found : {len(valid_designs)}")
    if valid_designs:
        print(f"  Best coking margin  : {valid_designs[0]['mg_cok']:.0f} K")
    else:
        print(f"  Best achievable peak T_cold_wall : {best_design['max_T_cw']:.0f} K  "
              f"(coking limit = {T_coking} K)")

    return valid_designs, best_design


# ====================================================================
# PLOTTING
# ====================================================================

def plot_results(x, radius, d, n_valid):
    x_mm        = x * 1000
    i_t         = np.argmin(radius)
    fully_valid = (d['max_T_hw'] < T_wall_limit and
                   d['max_T_cw'] < T_coking     and
                   d['max_T_cb'] < T_bulk_limit)
    status = "VALID DESIGN" if fully_valid else "BEST ACHIEVABLE (constraints violated)"
    bg     = "lightyellow"  if fully_valid else "#fff0f0"

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"{status}  —  {d['N_ch']} ch × {d['s_mm']:.3f} mm square  |  "
        f"RP-1/H2O2, Inconel 718  |  Re = {d['Re']:.0f}  V = {d['V']:.1f} m/s",
        fontsize=11, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    def vline(ax): ax.axvline(x_mm[i_t], color='gray', ls='--', alpha=0.6)

    # Contour
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(x_mm, -radius*1000, radius*1000, alpha=0.12, color='steelblue')
    ax.plot(x_mm,  radius*1000, 'k-', lw=2)
    ax.plot(x_mm, -radius*1000, 'k-', lw=2)
    vline(ax); ax.plot([], [], 'k--', alpha=0.5, label='Throat')
    ax.set_xlabel("x [mm]"); ax.set_ylabel("Radius [mm]")
    ax.set_title("Nozzle Contour"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Heat flux
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(x_mm, d['q_arr']/1e6, 'r-', lw=1.8); vline(ax)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("MW/m²")
    ax.set_title(f"Wall Heat Flux  (peak = {np.max(d['q_arr'])/1e6:.2f} MW/m²)")
    ax.grid(alpha=0.3)

    # Hot wall
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(x_mm, d['T_hw_arr'], 'r-', lw=1.8, label='T_hot_wall')
    ax.axhline(T_wall_limit, color='darkred', ls='--', lw=1.5,
               label=f'Inconel limit  {T_wall_limit:.0f} K')
    vline(ax)
    ok = "✓" if d['max_T_hw'] < T_wall_limit else "⚠"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Hot-Wall Temp  {ok}  (max = {d['max_T_hw']:.0f} K)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Cold wall
    ax = fig.add_subplot(gs[1, 0])
    over = d['T_cw_arr'] > T_coking
    ax.plot(x_mm, d['T_cw_arr'], color='darkorange', lw=1.8, label='T_cold_wall')
    if np.any(over):
        ax.fill_between(x_mm, T_coking, d['T_cw_arr'], where=over,
                        alpha=0.3, color='red', label='Coking violation')
    ax.axhline(T_coking, color='saddlebrown', ls='--', lw=1.5,
               label=f'Coking limit  {T_coking:.0f} K')
    vline(ax)
    cw_str = "✓" if d['max_T_cw'] < T_coking else f"⚠ over by {d['max_T_cw']-T_coking:.0f} K"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Cold-Wall (Coking)  {cw_str}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Coolant bulk
    ax = fig.add_subplot(gs[1, 1])
    over2 = d['T_cb_arr'] > T_bulk_limit
    ax.plot(x_mm, d['T_cb_arr'], 'b-', lw=1.8, label='T_coolant (bulk)')
    if np.any(over2):
        ax.fill_between(x_mm, T_bulk_limit, d['T_cb_arr'], where=over2,
                        alpha=0.2, color='navy', label='Bulk limit exceeded')
    ax.axhline(T_bulk_limit, color='navy', ls='--', lw=1.5,
               label=f'Bulk limit  {T_bulk_limit:.0f} K')
    vline(ax)
    cb_str = "✓" if d['max_T_cb'] < T_bulk_limit else f"⚠ over by {d['max_T_cb']-T_bulk_limit:.0f} K"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Coolant Bulk Temp  {cb_str}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Summary
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    def fmt(val, lim, label):
        mg = lim - val
        return f"{val:.0f} K  {'✓ +'+str(int(mg))+' K' if mg>0 else '⚠ '+str(int(mg))+' K'}"
    txt = (f"DESIGN SUMMARY\n{'─'*36}\n"
           f"N channels        : {d['N_ch']}\n"
           f"Channel size      : {d['s_mm']:.3f} mm (square)\n"
           f"Coolant Re        : {d['Re']:.0f}\n"
           f"Coolant velocity  : {d['V']:.1f} m/s\n"
           f"h_coolant         : {d['h_c']:.0f} W/m²K\n"
           f"mdot_coolant      : {mdot_coolant:.4f} kg/s\n"
           f"{'─'*36}\n"
           f"Max T_hot_wall    : {fmt(d['max_T_hw'],  T_wall_limit, 'wall')}\n"
           f"Max T_cold_wall   : {fmt(d['max_T_cw'],  T_coking,     'cok')}\n"
           f"Max T_coolant     : {fmt(d['max_T_cb'],  T_bulk_limit, 'bulk')}\n"
           f"T_coolant outlet  : {d['T_cb_outlet']:.0f} K\n"
           f"{'─'*36}\n"
           f"Valid designs     : {n_valid}\n"
           f"Status            : {status}\n")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8.5, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', fc=bg, alpha=0.95))

    plt.savefig("cooling_design_result.png", dpi=150, bbox_inches='tight')
    print("  Saved: cooling_design_result.png")
    plt.show()


def plot_design_space(valid):
    if not valid:
        print("  No valid designs — design space plot skipped.")
        return
    N_arr  = np.array([d['N_ch']   for d in valid])
    s_arr  = np.array([d['s_mm']   for d in valid])
    mg_arr = np.array([d['mg_cok'] for d in valid])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(N_arr, s_arr, c=mg_arr, cmap='RdYlGn', s=60,
                    edgecolors='k', lw=0.3)
    plt.colorbar(sc, ax=ax, label='Coking margin [K]')
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Channel side [mm]")
    ax.set_title("Valid Design Space — Coking Safety Margin")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("design_space.png", dpi=150, bbox_inches='tight')
    print("  Saved: design_space.png")
    plt.show()


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Regenerative Cooling Design — RP-1 / H2O2 Thruster")
    print("=" * 60)
    print(f"  Pc             = {Pc/1e6:.2f} MPa")
    print(f"  Tc             = {Tc:.0f} K")
    print(f"  mdot_total     = {mdot_total:.3f} kg/s")
    print(f"  mdot_coolant   = {mdot_coolant:.4f} kg/s  (O/F = {OF_ratio})")
    print(f"  Throat D       = {D_t*1000:.1f} mm")
    print(f"  Chamber D      = {D_c*1000:.1f} mm")
    print(f"  Exit AR        = {AR_exit}")
    print(f"  Wall           = {t_wall*1000:.1f} mm Inconel 718  (k = {k_wall} W/m·K)")
    print(f"  Constraints    : T_hw < {T_wall_limit} K | T_cw < {T_coking} K | T_cb < {T_bulk_limit} K")

    # Build geometry
    x, radius = generate_contour(N=300)
    i_t = np.argmin(radius)

    # Gas conditions
    M, T_static, T_aw = local_gas_conditions(radius)
    print(f"\n  T_aw : throat = {T_aw[i_t]:.0f} K  |  chamber = {T_aw[0]:.0f} K")
    print(f"  Mach : exit   = {M[-1]:.2f}")

    # Hot gas HTC
    h_g = hot_gas_htc(radius)
    print(f"  h_g  : throat = {h_g[i_t]:.0f} W/m²K  |  chamber = {h_g[0]:.0f} W/m²K")

    # Energy balance check
    energy_balance_report(x, radius, h_g, T_aw)

    # Optimize
    valid, best = optimize_channels(x, radius, h_g, T_aw)

    # Print table
    show  = valid if valid else [best]
    label = "Top valid designs" if valid else "Best achievable (constraints shown)"
    print(f"\n  {label}:")
    print(f"  {'N':>4} {'s[mm]':>7} {'Re':>8} {'T_hw':>7} {'T_cw':>7} "
          f"{'T_cb':>7} {'V[m/s]':>8} {'cok_mg':>8}")
    print("  " + "─" * 62)
    for d in show[:8]:
        flag = "  ✓" if (d['max_T_cw']<T_coking and d['max_T_cb']<T_bulk_limit) else "  ⚠"
        print(f"  {d['N_ch']:>4} {d['s_mm']:>7.3f} {d['Re']:>8.0f} "
              f"{d['max_T_hw']:>7.0f} {d['max_T_cw']:>7.0f} "
              f"{d['max_T_cb']:>7.0f} {d['V']:>8.1f} {d['mg_cok']:>8.0f}{flag}")

    # Plot
    plot_results(x, radius, show[0], len(valid))
    plot_design_space(valid)
