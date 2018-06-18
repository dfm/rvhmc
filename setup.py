#!/usr/bin/env python

import os
import tensorflow as tf
from setuptools import setup, Extension

compile_flags = tf.sysconfig.get_compile_flags()
compile_flags += ["-std=c++11", "-O2", "-mmacosx-version-min=10.9"]
link_flags = tf.sysconfig.get_link_flags()

extensions = [
    Extension(
        "rvhmc.kepler_op",
        sources=[os.path.join("rvhmc", "kepler_op.cc")],
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    name="rvhmc",
    license="MIT",
    packages=["rvhmc"],
    ext_modules=extensions,
    zip_safe=True,
)
