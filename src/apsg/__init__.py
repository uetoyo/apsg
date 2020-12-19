# -*- coding: utf-8 -*-


from apsg.core import Cluster, StereoGrid

# These objects are imported from package private modules.
from apsg._vector import Vec3
from apsg._tensor import DefGrad, VelGrad, Stress, Tensor, Ortensor, Ellipsoid
from apsg._feature import Fol, Lin, Pair, Fault, G, Group, PairSet, FaultSet
from apsg._settings import settings

from apsg._helpers import sind, cosd, tand, acosd, asind, atand, atan2d

from apsg.plotting import (
    StereoNet,
    VollmerPlot,
    RamsayPlot,
    FlinnPlot,
    HsuPlot,
    RosePlot,
)

from apsg.database import SDB


__all__ = (
    "Vec3",
    "Fol",
    "Lin",
    "Pair",
    "Fault",
    "Group",
    "PairSet",
    "FaultSet",
    "Cluster",
    "StereoGrid",
    "G",
    "settings",
    "SDB",
    "DefGrad",
    "VelGrad",
    "Stress",
    "Tensor",
    "Ortensor",
    "Ellipsoid",
    "sind",
    "cosd",
    "tand",
    "acosd",
    "asind",
    "atand",
    "atan2d",
    "StereoNet",
    "VollmerPlot",
    "RamsayPlot",
    "FlinnPlot",
    "HsuPlot",
    "RosePlot",
)

__version__ = "0.7.0"
__author__ = "Ondrej Lexa"
__email__ = "lexa.ondrej@gmail.com"
