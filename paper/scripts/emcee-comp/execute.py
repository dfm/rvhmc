#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import subprocess as sp

cpus = os.cpu_count()
if cpus is None:
    cpus = 1

if len(sys.argv) > 1:
    n_planets = int(sys.argv[1])
else:
    n_planets = 1

exe = [
    sys.executable,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py"),
    "{0}".format(n_planets),
]
procs = []
for version in range(cpus):
    procs.append(sp.Popen(exe + ["{0}".format(version)],
                          stdout=sp.PIPE, stderr=sp.PIPE))

for proc in procs:
    results = proc.communicate()
