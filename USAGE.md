# RP-1 / 98% H2O2 CEA Feasibility Study (Low Altitude 0--5000 m)

## Overview

This tool performs a structured propulsion feasibility study using NASA
CEA via CEA_Wrap.

It sweeps:

-   O/F from 6.5 to 7.5
-   Chamber pressure (5--15 bar by default)
-   Expansion ratio (ae/at = 4--20, suitable for low altitude operation)

For each combination it computes:

### CEA Outputs

-   Chamber temperature (Tc)
-   Characteristic velocity (c\*)
-   Thrust coefficient (Cf)
-   Vacuum Isp
-   Molecular weight and gamma

### Feasibility Outputs

For a required thrust:

-   Required throat area and diameter
-   Required exit area and diameter
-   Total mass flow
-   Fuel and oxidizer mass flow split

------------------------------------------------------------------------

## Configuration (No Python knowledge required)

All user-editable parameters are in:

    config.yaml

You can modify:

-   Required thrust
-   Chamber pressure sweep
-   O/F range
-   Expansion ratio range
-   Oxidizer purity
-   Efficiencies
-   Output directory

Example:

``` yaml
F_req_N: 800.0
Pc_bar_list: [6, 10, 14]
ae_at_list: [6, 8, 10, 12]
```

------------------------------------------------------------------------

## How to Run

From the project directory:

``` bash
python study.py
```

------------------------------------------------------------------------

## Output Files

Results are written to:

    outputs/

You will find:

-   results.csv → All cases
-   (optional plots if enabled)

Each row corresponds to one (Pc, O/F, ae/at) combination.

------------------------------------------------------------------------

## If You Get Material Name Errors

CEA material names must match your installed thermo library.

If names like "RP-1" or "H2O2(L)" fail:

Edit the default names inside:

    cea_runner.py

Common alternatives may be: - RP1 - H2O2 - H2O

------------------------------------------------------------------------

## Recommended Engineering Workflow

1.  Run baseline sweep
2.  Identify O/F giving minimum required mass flow
3.  Check throat and exit diameters for packaging feasibility
4.  Apply realistic efficiencies (\<1.0) for real engine sizing
5.  Iterate Pc and expansion ratio until geometry and flow are
    acceptable

------------------------------------------------------------------------

This structure keeps: - Configuration in YAML - CEA logic isolated -
Feasibility logic clear - Outputs reproducible
