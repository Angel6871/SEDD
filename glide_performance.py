"""
Glide altitude estimator + plot: required altitude vs CL/CD (i.e., L/D)

Basic still-air relation (no wind):
    horizontal_distance = (CL/CD) * altitude_loss
=>  required_altitude = distance / (CL/CD)

Notes:
- This assumes a steady glide at best L/D for each CL/CD value and ignores wind, turns,
  thermals/sink, and speed effects. Add a safety margin if needed.
"""

import numpy as np
import matplotlib.pyplot as plt


def required_altitude_m(distance_m, cl_cd, safety_margin=1.0):
    """
    distance_m: horizontal distance to glide [m]
    cl_cd: scalar or array of CL/CD values (L/D)
    safety_margin: multiply altitude by this factor (e.g., 1.3 for 30% margin)
    """
    cl_cd = np.asarray(cl_cd, dtype=float)
    if np.any(cl_cd <= 0):
        raise ValueError("All CL/CD values must be > 0.")
    return safety_margin * (distance_m / cl_cd)


def plot_altitude_vs_clcd(
    distance_m,
    cl_cd_min=14,
    cl_cd_max=25,
    n=200,
    safety_margin=1.0,
    highlight_clcd_range=None,
    save_path="glide_altitude_vs_clcd.png",
):
    cl_cd = np.linspace(cl_cd_min, cl_cd_max, n)
    alt_m = required_altitude_m(distance_m, cl_cd, safety_margin=safety_margin)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=120)

    # --- Main curve (default style) ---
    ax.plot(cl_cd, alt_m, label="Required altitude")

    if highlight_clcd_range is not None:
        # Parse and sanitize highlight tuple/list: (cl_lo, cl_hi)
        try:
            cl_lo_raw, cl_hi_raw = highlight_clcd_range
        except Exception as exc:
            raise ValueError("highlight_clcd_range must be a 2-value tuple/list, e.g. (14, 18).") from exc

        cl_lo, cl_hi = sorted([float(cl_lo_raw), float(cl_hi_raw)])
        if cl_lo <= 0:
            raise ValueError("highlight_clcd_range values must be > 0.")

        cl_lo = max(cl_lo, cl_cd_min)
        cl_hi = min(cl_hi, cl_cd_max)

        if cl_lo < cl_hi:
            alt_hi = required_altitude_m(distance_m, cl_lo, safety_margin=safety_margin)
            alt_lo = required_altitude_m(distance_m, cl_hi, safety_margin=safety_margin)

            # --- Region 1: Under curve for selected CL/CD ---
            mask = (cl_cd >= cl_lo) & (cl_cd <= cl_hi)
            ax.fill_between(
                cl_cd,
                0,
                alt_m,
                where=mask,
                alpha=0.25,
                color="royalblue",
                label=f"CL/CD range [{cl_lo:.1f}, {cl_hi:.1f}]",
            )

            # --- Region 2: Left of curve for altitude band ---
            y_band = np.linspace(alt_lo, alt_hi, 400)

            # x on curve from exact inverse relation
            x_on_curve = safety_margin * (distance_m / y_band)

            ax.fill_betweenx(
                y_band,
                cl_cd_min,
                x_on_curve,
                alpha=0.20,
                color="crimson",
                label=f"Altitude range [{alt_lo:.0f}, {alt_hi:.0f}] m",
            )

    # Axis limits (tight & clean)
    y_min = alt_m.min()
    y_max = alt_m.max()
    ax.set_xlim(cl_cd_min, cl_cd_max)
    ax.set_ylim(0.95 * y_min, 1.05 * y_max)

    ax.set_xlabel("CL/CD (L/D)")
    ax.set_ylabel("Required altitude loss [m]")
    #ax.set_title(
    #    f"Required altitude to glide {distance_m/1000:.2f} km "
    #    f"(margin {safety_margin:.2f}×)"
    #)

    ax.grid(True, alpha=0.3)

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # ---- User inputs ----
    distance_km = 20.0          # target glide distance [km]
    safety_margin = 1.25        # e.g., 1.25 = 25% extra altitude

    # Plot settings
    cl_cd_min = 10
    cl_cd_max = 25
    n_points = 300
    highlight_range = (16.5, 20)  # CL/CD interval to highlight on the plot
    # ---------------------

    distance_m = distance_km * 1000.0

    # Example: print required altitude for a single CL/CD
    cl_cd_example = 17.2
    alt_needed = required_altitude_m(distance_m, cl_cd_example, safety_margin=safety_margin)
    print(f"Altitude required for {distance_km:.1f} km at CL/CD={cl_cd_example:.1f}: {alt_needed:.1f} m")

    # Plot altitude vs CL/CD
    plot_altitude_vs_clcd(
        distance_m,
        cl_cd_min=cl_cd_min,
        cl_cd_max=cl_cd_max,
        n=n_points,
        safety_margin=safety_margin,
        highlight_clcd_range=highlight_range,
        save_path="glide_altitude_vs_clcd.png",
    )
