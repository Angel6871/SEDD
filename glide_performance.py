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


def plot_altitude_vs_clcd(distance_m, cl_cd_min=5, cl_cd_max=25, n=200, safety_margin=1.0):
    cl_cd = np.linspace(cl_cd_min, cl_cd_max, n)
    alt_m = required_altitude_m(distance_m, cl_cd, safety_margin=safety_margin)

    plt.figure()
    plt.plot(cl_cd, alt_m)
    plt.xlabel("CL/CD (L/D)")
    plt.ylabel("Required altitude loss [m]")
    plt.title(f"Required altitude to glide {distance_m/1000:.2f} km (margin {safety_margin:.2f}×)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ---- User inputs ----
    distance_km = 20.0          # target glide distance [km]
    safety_margin = 1.25        # e.g., 1.25 = 25% extra altitude

    # Plot settings
    cl_cd_min = 6
    cl_cd_max = 22
    n_points = 300
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
    )
