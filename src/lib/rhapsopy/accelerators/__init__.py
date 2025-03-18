#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:21:22 2023

Acceleration methods for the fixed-point problem

@author: lfrancoi
"""
from .IQN import IQNSolver
from .aitken_dynamicrelaxation import AitkenUnderrelaxationSolver
from .explicit import ExplicitSolver
from .fixed_point import FixedPointSolver, AitkenScalarSolver
from .newton import NewtonSolver
from .damped_newton import DampedNewtonSolver
from .anderson import AndersonSolver
from .base import BaseFixedPointSolver
from .JFNK import JFNKSolver
from .Broyden import Broyden1Solver, Broyden2Solver

# from src.lib.rhapsopy.coupling.IQN import IQNSolver
# from src.lib.rhapsopy.coupling.aitken_dynamicrelaxation import AitkenUnderrelaxationSolver
# from src.lib.rhapsopy.coupling.explicit import ExplicitSolver
# from src.lib.rhapsopy.coupling.fixed_point import FixedPointSolver, AitkenScalarSolver
# from src.lib.rhapsopy.coupling.newton import NewtonSolver
# from src.lib.rhapsopy.coupling.damped_newton import DampedNewtonSolver
# from src.lib.rhapsopy.coupling.anderson import AndersonSolver
# from src.lib.rhapsopy.coupling.base import BaseFixedPointSolver
# from src.lib.rhapsopy.coupling.JFNK import JFNKSolver
# from src.lib.rhapsopy.coupling.Broyden import Broyden1Solver, Broyden2Solver
