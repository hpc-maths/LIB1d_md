#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:40:59 2025

@author: lfrancoi
"""

class WRnonConvergence(Exception):
    """ Exception class used in cased of convergence failure for implicit code-coupling """
    pass

class ExceptionWhichMayDisappearWhenLoweringDeltaT(Exception):
    """ Exception class used for handling other issues (e.g. failures within subsystems) """
    pass