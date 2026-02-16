import CEA_Wrap
import numpy as np

print(dir(CEA_Wrap))

from dataclasses import dataclass
from typing import Optional, Dict, Any

from CEA_Wrap import RocketProblem, Oxidizer, Fuel, CEA_Class


from CEA_Wrap import RocketProblem, Oxidizer, Fuel, CEA_Class

cea = CEA_Class()

ox = Oxidizer(
    name="H2O2(L)",
    wt_percent=98.0
)

fuel = Fuel(name="RP-1")

prob = RocketProblem(
    oxidizer=ox,
    fuel=fuel,
    Pc=30.0,
    OF=7.0,
    eps=40.0,
    pressure_units="bar"
)

out = cea.solve(prob)

print(out)
