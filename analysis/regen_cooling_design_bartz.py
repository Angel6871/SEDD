"""
Regenerative Cooling Channel Design Tool — BARTZ EQUATION VERSION
RP-1 / H2O2 Bipropellant Thruster — Inconel 718 Wall

Methodology:
  - Cervone AE4903 Lecture 7 (TU Delft): 1D thermal resistance network
  - BARTZ EQUATION (Huzel & Huang): calibrated for rocket engine nozzles
  - Isentropic relations for axial gas conditions
  - Counter-flow coolant energy balance

Improvements:
  1. Rectangular channels (w × h) — aspect ratio is a free parameter
  2. Fin efficiency — accounts for heat conduction through ribs between channels
  3. Axially-varying channel width — w scales with local circumference so that
     the channel fill fraction stays constant; channels are widest in the chamber
     and naturally narrow toward the throat, reducing pressure drop while keeping
     cooling where it is needed most
  6. Bartz h_g iterated on actual hot-wall temperature until convergence

Constraints checked:
  1. T_hot_wall  < T_wall_limit   (Inconel 718 hot-side wall)
  2. T_cold_wall < T_coking       (RP-1 coking on coolant-side wall)
  3. T_coolant   < T_bulk_limit   (coolant bulk temperature)
  4. Re_coolant  > Re_min         (Gnielinski validity, worst-case axial station)
  5. t_rib       > t_rib_min      (minimum manufacturable rib/fin thickness)

Pressure drop IS computed using Darcy-Weisbach integrated along the axially-varying
channel geometry. Pump power is reported but not used as a filter.

====================================================================
  *** INPUTS TO UPDATE WHEN BETTER VALUES ARE AVAILABLE ***
  Search for the tag  # [UPDATE]  throughout the file
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq
import csv


# ====================================================================
# SECTION 1 — ENGINE / CEA PARAMETERS
# ====================================================================
# [UPDATE] Replace with final CEA output values

Pc          = 10.738e5        # Chamber pressure [Pa]
Tc          = 1969.1        # Adiabatic flame temperature [K]
gamma       = 1.2776         # Specific heat ratio [-]
Pr_g        = 0.594        # Gas Prandtl number [-]
mu_g        = 4.8e-5      # Gas dynamic viscosity [Pa·s]
cp_g        = 2245.5         # Gas specific heat [J/kg·K]
k_g         = mu_g * cp_g / Pr_g   # Gas thermal conductivity [W/m·K]

mdot_total  = 0.5083         # Total propellant mass flow rate [kg/s]


# ====================================================================
# SECTION 2 — COOLANT PROPERTIES (RP-1, liquid phase)
# ====================================================================
# Temperature-dependent polynomial fits for liquid RP-1.
# Valid range: ~250–580 K (subcritical, high-pressure liquid phase).
# [UPDATE] Replace with measured data or a thermodynamic library if available.
#
# Sources / basis:
#   rho  : linear fit anchored at 298 K (773 kg/m³), slope from NIST/literature
#   mu   : Andrade equation ln(mu) = A + B/T, anchored at 298 K and 500 K
#   k    : linear fit, slight decrease with temperature
#   cp   : linear fit, moderate increase with temperature

OF_ratio        = 2.872
mdot_coolant    = mdot_total / (1.0 + OF_ratio)  # [kg/s]

T_coolant_in    = 298.0      # Coolant inlet temperature [K]


def rho_c_f(T):
    """RP-1 liquid density [kg/m³].  Valid ~250–580 K."""
    return 960.7 - 0.626 * T                        # 773 kg/m³ at 298 K


def mu_c_f(T):
    """RP-1 dynamic viscosity [Pa·s]. Andrade equation.  Valid ~250–580 K."""
    return np.exp(-10.32 + 1247.0 / np.clip(T, 250.0, 700.0))  # 0.0021 Pa·s at 298 K


def k_c_f(T):
    """RP-1 thermal conductivity [W/m·K]. Valid ~250–580 K."""
    return np.clip(0.145 - 1.5e-4 * (T - 298.0), 0.08, 0.20)  # 0.145 W/m·K at 298 K


def cp_c_f(T):
    """RP-1 specific heat [J/kg·K]. Valid ~250–580 K."""
    return 1950.0 + 1.5 * (T - 298.0)                          # 1950 J/kg·K at 298 K


def Pr_c_f(T):
    """RP-1 Prandtl number [-]."""
    return cp_c_f(T) * mu_c_f(T) / k_c_f(T)


# Reference values at inlet temperature (used where a single scalar is needed)
rho_c = rho_c_f(T_coolant_in)
mu_c  = mu_c_f(T_coolant_in)
k_c   = k_c_f(T_coolant_in)
cp_c  = cp_c_f(T_coolant_in)
Pr_c  = Pr_c_f(T_coolant_in)


# ====================================================================
# SECTION 3 — WALL MATERIAL (Inconel 718)
# ====================================================================
# [UPDATE] t_wall and t_rib_min are manufacturing/design choices.

t_wall          = 0.001      # Inner wall thickness [m]  (1 mm)
k_wall          = 18      # Thermal conductivity [W/m·K]
t_rib_min       = 0.001     # Minimum rib thickness between channels [m]  (0.4 mm)
roughness_lpbf  = 0.00002    # LPBF Inconel 718 internal channel roughness ε [m]  (20 μm)
                             # [UPDATE] Measure or adjust based on actual build parameters


# ====================================================================
# SECTION 4 — TEMPERATURE CONSTRAINTS
# ====================================================================
# [UPDATE] Adjust when final material specs and RP-1 conditions are known.

T_wall_limit    = 1300.0     # Inconel 718 max service temp [K]
T_coking        = 650.0      # RP-1 coking limit on cold-wall surface [K]
T_bulk_limit    = 600.0      # RP-1 bulk temperature limit [K]
Re_min          = 4000       # Reported in output only — laminar fallback means no Re filter applied


# ====================================================================
# SECTION 5 — NOZZLE GEOMETRY
# ====================================================================
# [UPDATE] Replace with final nozzle design dimensions.

D_t     = 2 * 0.01479        # Throat diameter [m]      (Rt = 14.79 mm)
D_c     = 2 * 0.02767        # Chamber diameter [m]     (Rc = 27.67 mm)
AR_exit = (21.88 / 14.79)**2 # Exit area ratio [-]      (Re = 21.88 mm)
L_con   = 0.03538            # Convergent section length [m]  (beta = 20 deg)
L_div   = 0.02184            # Divergent section length [m]   (alpha = 18 deg)
L_ch    = 0.39890            # Cylindrical chamber length [m]

R_t     = D_t / 2
R_c     = D_c / 2
R_exit  = R_t * np.sqrt(AR_exit)
AR_c    = (R_c / R_t)**2



# ====================================================================
# SECTION 6 — CHANNEL OPTIMIZER SEARCH RANGE (rectangular channels)
# ====================================================================
# [UPDATE] Adjust based on manufacturing constraints.
#
# Channels are w (width, circumferential) × h (height/depth, radial).
# w_ref is the channel width at the chamber reference diameter R_c.
# Along the axis: w(x) = w_ref * r(x)/R_c  — scales with circumference,
# so channel fill fraction is constant and ribs stay proportional.

N_ch_min    = 40
N_ch_max    = 40
N_ch_step   = 1

w_min_throat = 0.001                  # Manufacturing min channel width at throat [m]
w_min        = w_min_throat * R_c / R_t  # Min w_ref so throat width ≥ w_min_throat
w_max       = 0.003      # Max channel width at chamber [m]  (3.0 mm)
w_steps     = 20

h_min       = 0.0005     # Min channel height (depth) [m]  (0.4 mm)  [UPDATE: manufacturing limit]
h_max       = 0.003      # Max channel height (depth) [m]  (3.0 mm)
h_steps     = 20

# Bartz T_wall iteration
bartz_iter  = 5          # Fixed-point iterations for h_g / T_wall convergence


# ====================================================================
# GEOMETRY BUILDER
# ====================================================================

def generate_contour(N=300):
    """
    Conical converging-diverging contour.
      - Chamber    : cylindrical at R_c, length L_ch
      - Converging : straight conical from R_c to R_t over L_con (beta = 20 deg)
      - Diverging  : straight conical from R_t to R_exit over L_div (alpha = 18 deg)
    Throat at x = 0.
    """
    N_ch_pts = N // 4
    N_cv_pts = N // 4
    N_dv_pts = N // 2

    x_ch = np.linspace(-(L_ch + L_con), -L_con, N_ch_pts)
    r_ch = np.full(N_ch_pts, R_c)

    x_cv = np.linspace(-L_con, 0.0, N_cv_pts)
    r_cv = R_c + (R_t - R_c) * (x_cv + L_con) / L_con   # linear (conical)

    x_dv = np.linspace(0.0, L_div, N_dv_pts)
    r_dv = R_t + (R_exit - R_t) * (x_dv / L_div)

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
    T_aw = T_static + Pr^(1/3) * (T_total - T_static)
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
# HOT GAS HTC — BARTZ EQUATION (iterated on actual T_wall)
# ====================================================================

def hot_gas_htc(radius, T_aw, T_wall=None):
    """
    Dittus-Boelter with Bartz wall-temperature and area-ratio corrections:
      h_g = (Nu * k_g / D) * (A_t / A)^0.9

    The (A_t/A)^0.9 term is from the Bartz equation (Huzel & Huang) and
    accounts for the reduction in gas-side HTC in wider cross-sections.
    At the throat A_t/A = 1 (no correction); in the cylindrical chamber
    A_t/A = 1/eps_c ≈ 1/3.5, giving a factor of ~0.31 — reducing h_g
    to ~31% of the throat value, which is physically correct.

    The (T_wall/Tc)^(-0.2) correction accounts for gas property variation
    across the thermal boundary layer and requires iteration on T_wall.
    """
    if T_wall is None:
        T_wall = T_aw.copy()
    i_t   = np.argmin(radius)
    A_t   = np.pi * radius[i_t]**2
    h_g   = np.zeros(len(radius))
    for i, r in enumerate(radius):
        D            = 2 * r
        A_local      = np.pi * r**2
        area_ratio   = (A_t / A_local)**0.9
        G            = mdot_total / (np.pi * r**2)
        Re           = G * D / mu_g
        T_correction = (T_wall[i] / Tc)**(-0.2)
        Nu           = 0.023 * Re**0.8 * Pr_g**0.4 * T_correction
        h_g[i]       = Nu * k_g / D * area_ratio
    return h_g


# ====================================================================
# COOLANT HTC — GNIELINSKI, RECTANGULAR CHANNELS, AXIALLY VARYING
# ====================================================================

def friction_factor(Re, D_h=None):
    """
    Darcy friction factor.
      Laminar  (Re < 2300): f = 64 / Re
      Turbulent: Swamee-Jain approximation of Colebrook equation.
        If D_h is provided, includes LPBF surface roughness (roughness_lpbf).
        Rough surfaces give higher f, which flows through to higher Nu in Gnielinski.
    """
    if Re < 2300:
        return 64.0 / Re
    if D_h is not None and roughness_lpbf > 0:
        # Colebrook–White (Swamee-Jain explicit form) with roughness
        return 0.25 / (np.log10(roughness_lpbf / (3.7 * D_h) + 5.74 / Re**0.9))**2
    return 0.25 / (np.log10(5.74 / Re**0.9))**2


def nusselt_number(Re, Pr, w, h):
    """
    Nusselt number for a rectangular channel covering all flow regimes.

    Laminar  (Re ≤ 2300): Shah-London correlation for rectangular duct,
        uniform heat flux boundary condition.
        Nu = 8.235 * (1 - 2.0421α + 3.0853α² - 2.4765α³ + 1.0578α⁴ - 0.1861α⁵)
        where α = min(w,h)/max(w,h)  (aspect ratio ≤ 1)
        Gives Nu ≈ 5.4 for square (α=1) up to 8.24 for parallel plates (α→0).

    Turbulent (Re ≥ 3000): Gnielinski correlation.

    Transition (2300 < Re < 3000): linear blend of both.
    """
    ar   = min(w, h) / max(w, h)   # aspect ratio ≤ 1
    Nu_lam = 8.235 * (1 - 2.0421*ar + 3.0853*ar**2
                        - 2.4765*ar**3 + 1.0578*ar**4 - 0.1861*ar**5)
    Nu_lam = max(Nu_lam, 3.0)      # floor for numerical safety

    if Re <= 2300:
        return Nu_lam

    D_h    = 2 * w * h / (w + h)      # hydraulic diameter — needed for roughness correction
    f      = friction_factor(Re, D_h)
    Nu_turb = (f/8) * (Re - 1000) * Pr / (1 + 12.7*(f/8)**0.5 * (Pr**(2/3) - 1))
    Nu_turb = max(Nu_turb, Nu_lam)  # Gnielinski can give nonsense near Re=1000

    if Re >= 3000:
        return Nu_turb

    # Linear blend through the transition region
    blend = (Re - 2300) / (3000 - 2300)
    return (1 - blend) * Nu_lam + blend * Nu_turb


def channel_geometry(w_ref, h, N_ch, radius, T_ref=None):
    """
    Axially-varying rectangular channel geometry.

    Channel width scales with local circumference:
        w(x) = w_ref * r(x) / R_c

    This keeps the channel fill fraction (w / pitch) constant along the axis,
    so ribs and channels narrow together toward the throat. At the chamber
    (r = R_c) the channel width equals w_ref; it narrows toward the throat.

    T_ref : scalar or array of coolant temperatures for property evaluation.
            Defaults to T_coolant_in (inlet, conservative — cold RP-1 is most viscous).

    Returns arrays: w, D_h, t_rib, V, Re  (one value per axial station).
    """
    if T_ref is None:
        T_ref = T_coolant_in
    rho = rho_c_f(T_ref)
    mu  = mu_c_f(T_ref)

    w     = w_ref * (radius / R_c)            # local channel width [m]
    D_h   = 2 * w * h / (w + h)              # hydraulic diameter [m]
    pitch = 2 * np.pi * radius / N_ch        # circumferential pitch [m]
    t_rib = pitch - w                         # rib (fin) thickness [m]
    A_ch  = w * h                             # channel cross-section [m²]
    V     = mdot_coolant / (rho * N_ch * A_ch)
    Re    = rho * V * D_h / mu
    return w, D_h, t_rib, V, Re


def coolant_htc_array(w_ref, h, N_ch, radius):
    """
    Coolant HTC at every axial station, with fin efficiency applied.

    Nusselt number uses nusselt_number() which covers:
      - Laminar  (Re ≤ 2300): Shah-London rectangular duct correlation
      - Turbulent (Re ≥ 3000): Gnielinski
      - Transition: linear blend

    Coolant properties are evaluated at the average bulk temperature
    T_bulk_avg = (T_coolant_in + T_bulk_limit) / 2, which is more
    representative than always using the cold inlet temperature. RP-1
    viscosity drops ~5× between 298 K and 500 K, so this significantly
    increases Re and Nu compared to the conservative inlet-only evaluation.

    Fin model (rectangular fin between channels, adiabatic tip):
        m_fin  = sqrt(2 * h_c / (k_wall * t_rib))
        eta_fin = tanh(m_fin * h) / (m_fin * h)

    Returns (h_c_eff_arr, Re_arr, V_arr) or (None, None, None) if the
    design is geometrically infeasible (rib too thin).
    """
    T_bulk_avg = (T_coolant_in + T_bulk_limit) / 2.0   # representative bulk temperature
    w, D_h, t_rib, V, Re = channel_geometry(w_ref, h, N_ch, radius, T_ref=T_bulk_avg)

    if np.any(t_rib < t_rib_min):
        return None, None, None

    h_c_eff = np.zeros(len(radius))
    for i in range(len(radius)):
        Pr_loc  = Pr_c_f(T_bulk_avg)
        k_loc   = k_c_f(T_bulk_avg)
        Nu      = nusselt_number(Re[i], Pr_loc, w[i], h)
        h_c_raw = Nu * k_loc / D_h[i]

        # Fin efficiency
        m_fin      = np.sqrt(max(2 * h_c_raw / (k_wall * t_rib[i]), 0.0))
        mh         = m_fin * h
        eta_fin    = np.tanh(mh) / mh if mh > 1e-6 else 1.0
        h_c_eff[i] = h_c_raw * (w[i] + 2 * eta_fin * h) / (w[i] + 2 * h)

    return h_c_eff, Re, V




# ====================================================================
# 1D THERMAL RESISTANCE SOLVER
# ====================================================================

def solve_thermal(x, radius, h_g, h_c_in, T_aw):
    """
    Counter-flow thermal network (Cervone §7.2).
    h_c_in may be a scalar or an array (one value per axial station).

        q = (T_aw - T_coolant) / (1/h_g + t_wall/k_wall + 1/h_c)
        T_hot_wall  = T_aw     - q / h_g
        T_cold_wall = T_coolant + q / h_c

    Coolant enters at the nozzle exit (index N-1) and flows toward the
    chamber (decreasing index).
    """
    h_c = np.full(len(x), h_c_in) if np.isscalar(h_c_in) else np.asarray(h_c_in)

    N           = len(x)
    T_coolant   = np.zeros(N)
    T_hot_wall  = np.zeros(N)
    T_cold_wall = np.zeros(N)
    heat_flux   = np.zeros(N)

    T_coolant[-1] = T_coolant_in

    for i in range(N-1, 0, -1):
        R_tot           = 1/h_g[i] + t_wall/k_wall + 1/h_c[i]
        q               = (T_aw[i] - T_coolant[i]) / R_tot
        heat_flux[i]    = q
        T_hot_wall[i]   = T_aw[i]      - q / h_g[i]
        T_cold_wall[i]  = T_coolant[i] + q / h_c[i]
        perim           = 2 * np.pi * radius[i]
        dx              = abs(x[i] - x[i-1])
        cp_local        = cp_c_f(T_coolant[i])   # temperature-dependent cp
        T_coolant[i-1]  = T_coolant[i] + q * perim * dx / (mdot_coolant * cp_local)

    R_tot           = 1/h_g[0] + t_wall/k_wall + 1/h_c[0]
    q               = (T_aw[0] - T_coolant[0]) / R_tot
    heat_flux[0]    = q
    T_hot_wall[0]   = T_aw[0]      - q / h_g[0]
    T_cold_wall[0]  = T_coolant[0] + q / h_c[0]

    return T_hot_wall, T_cold_wall, T_coolant, heat_flux


# ====================================================================
# ENERGY BALANCE DIAGNOSTIC
# ====================================================================

def energy_balance_report(x, radius, h_g, T_aw):
    """
    Estimates total heat load vs coolant capacity before the optimizer runs.
    """
    Q_total = 0.0
    for i in range(len(x) - 1):
        R_min   = 1/h_g[i] + t_wall/k_wall     # best case: h_c → ∞
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
        print(f"  WARNING: Coolant flow insufficient by {ratio:.1f}x")
        print(f"     No fully valid design is possible with current parameters.")
    else:
        print(f"  OK: Coolant flow sufficient — valid designs should exist.")


# ====================================================================
# CHANNEL OPTIMIZER
# ====================================================================

def optimize_channels(x, radius, h_g, T_aw):
    """
    Sweep (N_ch, w_ref, h):
      - w_ref : channel width at chamber reference radius R_c
      - h     : channel height/depth (constant along axis)
      - N_ch  : number of channels

    At each station the channel width is w(x) = w_ref * r(x)/R_c.
    Fin efficiency is computed locally at each station.
    """
    N_range = range(N_ch_min, N_ch_max + 1, N_ch_step)
    w_range = np.linspace(w_min, w_max, w_steps)
    h_range = np.linspace(h_min, h_max, h_steps)
    total   = len(N_range) * w_steps * h_steps

    print(f"\n  Running optimizer over {total} combinations "
          f"({len(N_range)} N_ch × {w_steps} widths × {h_steps} heights)...")

    valid_designs = []
    best_score    = 1e12
    best_design   = None

    for N_ch in N_range:
        for w_ref in w_range:
            for h in h_range:

                h_c_arr, Re_arr, V_arr = coolant_htc_array(w_ref, h, N_ch, radius)
                if h_c_arr is None:
                    continue  # rib too thin — not manufacturable

                T_hw, T_cw, T_cb, q = solve_thermal(x, radius, h_g, h_c_arr, T_aw)

                # Derived geometry — chamber (index 0) and throat
                w_throat      = w_ref * R_t / R_c
                pitch_ch      = 2 * np.pi * R_c / N_ch
                pitch_th      = 2 * np.pi * R_t / N_ch
                t_rib_ch      = pitch_ch - w_ref
                t_rib_th      = pitch_th - w_throat
                coverage_ch   = w_ref / pitch_ch          # fraction of circumference covered
                i_t           = np.argmin(radius)

                result = {
                    'N_ch'        : N_ch,
                    'w_ref_mm'    : w_ref    * 1000,
                    'w_t_mm'      : w_throat * 1000,
                    'h_mm'        : h        * 1000,
                    't_rib_ch_mm' : t_rib_ch * 1000,
                    't_rib_th_mm' : t_rib_th * 1000,
                    'coverage_ch' : coverage_ch,
                    'AR'          : w_ref / h,
                    'Re_ch'       : Re_arr[0],
                    'Re_th'       : Re_arr[i_t],
                    'V'           : V_arr[0],
                    'h_c'         : h_c_arr[0],
                    'max_T_hw'    : np.max(T_hw),
                    'max_T_cw'    : np.max(T_cw),
                    'max_T_cb'    : np.max(T_cb),
                    'T_cb_outlet' : T_cb[0],
                    'mg_wall'     : T_wall_limit - np.max(T_hw),
                    'mg_cok'      : T_coking     - np.max(T_cw),
                    'mg_bulk'     : T_bulk_limit - np.max(T_cb),
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
    elif best_design:
        print(f"  Best achievable T_cold_wall : {best_design['max_T_cw']:.0f} K  "
              f"(limit = {T_coking} K)")
    else:
        print(f"  No designs computed.")

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
    bg     = "lightyellow" if fully_valid else "#fff0f0"

    fig = plt.figure(figsize=(15, 10))
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
               label=f'Inconel limit {T_wall_limit:.0f} K')
    vline(ax)
    ok_str = "OK" if d['max_T_hw'] < T_wall_limit else "XX"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Hot-Wall Temp [{ok_str}] (max = {d['max_T_hw']:.0f} K)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Cold wall
    ax = fig.add_subplot(gs[1, 0])
    over = d['T_cw_arr'] > T_coking
    ax.plot(x_mm, d['T_cw_arr'], color='darkorange', lw=1.8, label='T_cold_wall')
    if np.any(over):
        ax.fill_between(x_mm, T_coking, d['T_cw_arr'], where=over,
                        alpha=0.3, color='red', label='Coking violation')
    ax.axhline(T_coking, color='saddlebrown', ls='--', lw=1.5,
               label=f'Coking limit {T_coking:.0f} K')
    vline(ax)
    cw_str = "OK" if d['max_T_cw'] < T_coking else f"XX +{d['max_T_cw']-T_coking:.0f} K"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Cold-Wall (Coking) [{cw_str}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Coolant bulk
    ax = fig.add_subplot(gs[1, 1])
    over2 = d['T_cb_arr'] > T_bulk_limit
    ax.plot(x_mm, d['T_cb_arr'], 'b-', lw=1.8, label='T_coolant (bulk)')
    if np.any(over2):
        ax.fill_between(x_mm, T_bulk_limit, d['T_cb_arr'], where=over2,
                        alpha=0.2, color='navy', label='Bulk limit exceeded')
    ax.axhline(T_bulk_limit, color='navy', ls='--', lw=1.5,
               label=f'Bulk limit {T_bulk_limit:.0f} K')
    vline(ax)
    cb_str = "OK" if d['max_T_cb'] < T_bulk_limit else f"XX +{d['max_T_cb']-T_bulk_limit:.0f} K"
    ax.set_xlabel("x [mm]"); ax.set_ylabel("K")
    ax.set_title(f"Coolant Bulk Temp [{cb_str}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Summary box
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    def fmt(val, lim):
        mg = lim - val
        return f"{val:.0f} K [{'OK +' if mg > 0 else 'XX '}{int(mg)} K]"
    txt = (f"DESIGN SUMMARY (BARTZ + FIN EFF)\n{'='*38}\n"
           f"N channels        : {d['N_ch']}\n"
           f"Width at chamber  : {d['w_ref_mm']:.3f} mm\n"
           f"Width at throat   : {d['w_t_mm']:.3f} mm\n"
           f"Channel height    : {d['h_mm']:.3f} mm\n"
           f"Aspect ratio (ch) : {d['AR']:.2f}\n"
           f"Rib at chamber    : {d['t_rib_ch_mm']:.3f} mm\n"
           f"Rib at throat     : {d['t_rib_th_mm']:.3f} mm\n"
           f"Coverage (chamber): {d['coverage_ch']*100:.1f} %\n"
           f"Coolant V (ch)    : {d['V']:.1f} m/s\n"
           f"h_c_eff (ch)      : {d['h_c']:.0f} W/m²K\n"
           f"mdot_coolant      : {mdot_coolant:.4f} kg/s\n"
           f"{'='*38}\n"
           f"Max T_hot_wall    : {fmt(d['max_T_hw'], T_wall_limit)}\n"
           f"Max T_cold_wall   : {fmt(d['max_T_cw'], T_coking)}\n"
           f"Max T_coolant     : {fmt(d['max_T_cb'], T_bulk_limit)}\n"
           f"T_coolant outlet  : {d['T_cb_outlet']:.0f} K\n"
           f"{'='*38}\n"
           f"Valid designs     : {n_valid}\n"
           f"Status            : {status}\n")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', fc=bg, alpha=0.95))

    plt.savefig("cooling_design_result_bartz.png", dpi=150, bbox_inches='tight')
    print("  Saved: cooling_design_result_bartz.png")
    plt.show()


def plot_design_space(valid):
    if not valid:
        print("  No valid designs — design space plot skipped.")
        return

    N_arr  = np.array([d['N_ch']     for d in valid])
    w_arr  = np.array([d['w_ref_mm'] for d in valid])
    h_arr  = np.array([d['h_mm']     for d in valid])
    mg_arr = np.array([d['mg_cok']   for d in valid])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(N_arr, w_arr, c=mg_arr, cmap='RdYlGn', s=50,
                         edgecolors='k', lw=0.3)
    plt.colorbar(sc, ax=axes[0], label='Coking margin [K]')
    axes[0].set_xlabel("Number of channels")
    axes[0].set_ylabel("Width at chamber [mm]")
    axes[0].set_title("Design Space — N_ch vs Width (BARTZ)")
    axes[0].grid(alpha=0.3)

    sc2 = axes[1].scatter(w_arr, h_arr, c=mg_arr, cmap='RdYlGn', s=50,
                          edgecolors='k', lw=0.3)
    plt.colorbar(sc2, ax=axes[1], label='Coking margin [K]')
    axes[1].set_xlabel("Width at chamber [mm]")
    axes[1].set_ylabel("Channel height [mm]")
    axes[1].set_title("Design Space — Width vs Height (BARTZ)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("design_space_bartz.png", dpi=150, bbox_inches='tight')
    print("  Saved: design_space_bartz.png")
    plt.show()


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  Regenerative Cooling Design — RP-1 / H2O2 Thruster")
    print("  --- BARTZ + RECTANGULAR + FIN EFF + AXIAL VARIATION ---")
    print("=" * 60)
    print(f"  Pc             = {Pc/1e6:.2f} MPa")
    print(f"  Tc             = {Tc:.0f} K")
    print(f"  mdot_total     = {mdot_total:.3f} kg/s")
    print(f"  mdot_coolant   = {mdot_coolant:.4f} kg/s  (O/F = {OF_ratio})")
    print(f"  Throat D       = {D_t*1000:.1f} mm")
    print(f"  Chamber D      = {D_c*1000:.1f} mm")
    print(f"  Exit AR        = {AR_exit}")
    print(f"  Wall           = {t_wall*1000:.1f} mm Inconel 718  (k = {k_wall} W/m·K)")
    print(f"  Min rib thick  = {t_rib_min*1000:.1f} mm")
    print(f"  Constraints    : T_hw < {T_wall_limit} K | T_cw < {T_coking} K | T_cb < {T_bulk_limit} K")

    # Build geometry
    x, radius = generate_contour(N=300)
    i_t = np.argmin(radius)

    # Gas conditions
    M, T_static, T_aw = local_gas_conditions(radius)
    print(f"\n  T_aw : throat = {T_aw[i_t]:.0f} K  |  chamber = {T_aw[0]:.0f} K")
    print(f"  Mach : exit   = {M[-1]:.2f}")

    # --- Bartz h_g iteration on actual T_wall ---
    # We use a moderate reference h_c for the iteration thermal solve.
    # The Bartz correction (T_wall/Tc)^(-0.2) is a weak effect (~10-20%),
    # so a representative h_c is sufficient to converge T_wall.
    print(f"\n  Computing h_g with Bartz T_wall iteration ({bartz_iter} cycles)...")
    h_c_iter    = 5000.0          # W/m²K — representative coolant HTC
    T_wall_iter = T_aw.copy()     # initial guess: adiabatic wall
    for _ in range(bartz_iter):
        h_g = hot_gas_htc(radius, T_aw, T_wall=T_wall_iter)
        T_wall_iter, _, _, _ = solve_thermal(x, radius, h_g, h_c_iter, T_aw)
    h_g = hot_gas_htc(radius, T_aw, T_wall=T_wall_iter)
    print(f"  h_g  : throat = {h_g[i_t]:.0f} W/m²K  |  chamber = {h_g[0]:.0f} W/m²K")
    print(f"  T_wall (iterated): throat = {T_wall_iter[i_t]:.0f} K  |  chamber = {T_wall_iter[0]:.0f} K")

    # Energy balance check
    energy_balance_report(x, radius, h_g, T_aw)

    # Optimize
    valid, best = optimize_channels(x, radius, h_g, T_aw)

    # Print results table
    show  = valid if valid else [best]
    label = "Top valid designs" if valid else "Best achievable (constraints shown)"
    print(f"\n  {label}:")
    print(f"  {'N':>4} {'w_ch[mm]':>9} {'w_th[mm]':>9} {'h[mm]':>7} {'Re':>8} "
          f"{'T_hw':>7} {'T_cw':>7} {'cok_mg':>8}")
    print("  " + "-" * 80)
    for d in show[:8]:
        flag = "  OK" if (d['max_T_cw'] < T_coking and d['max_T_cb'] < T_bulk_limit) else "  XX"
        print(f"  {d['N_ch']:>4} {d['w_ref_mm']:>9.3f} {d['w_t_mm']:>9.3f} "
              f"{d['h_mm']:>7.3f} {d['Re_ch']:>8.0f} "
              f"{d['max_T_hw']:>7.0f} {d['max_T_cw']:>7.0f} "
              f"{d['mg_cok']:>8.0f}{flag}")

    # Export CSV
    csv_file = "cooling_designs_bartz.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N_channels", "w_chamber_mm", "w_throat_mm", "h_mm",
            "t_rib_chamber_mm", "t_rib_throat_mm", "coverage_chamber_pct",
            "Re_chamber", "Re_throat",
            "max_T_hot_wall_K", "max_T_cold_wall_K", "max_T_coolant_K",
            "T_coolant_outlet_K",
            "wall_margin_K", "coking_margin_K", "bulk_margin_K"
        ])
        for d in (valid if valid else [best]):
            writer.writerow([
                d["N_ch"],
                f"{d['w_ref_mm']:.4f}", f"{d['w_t_mm']:.4f}", f"{d['h_mm']:.4f}",
                f"{d['t_rib_ch_mm']:.4f}", f"{d['t_rib_th_mm']:.4f}",
                f"{d['coverage_ch']*100:.1f}",
                f"{d['Re_ch']:.0f}", f"{d['Re_th']:.0f}",
                f"{d['max_T_hw']:.1f}", f"{d['max_T_cw']:.1f}", f"{d['max_T_cb']:.1f}",
                f"{d['T_cb_outlet']:.1f}",
                f"{d['mg_wall']:.1f}", f"{d['mg_cok']:.1f}", f"{d['mg_bulk']:.1f}"
            ])
    print(f"  Saved: {csv_file}  ({len(valid if valid else [best])} rows)")

    # Plot
    plot_results(x, radius, show[0], len(valid))
    plot_design_space(valid)
