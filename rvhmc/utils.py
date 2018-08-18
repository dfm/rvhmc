# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["join_names"]


def join_names(*names):
    names = [n for n in names if n is not None]
    if len(names):
        return ":".join(names)
    return ""
