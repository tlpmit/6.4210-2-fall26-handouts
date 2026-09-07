"""Code shared between psets, rather than belonging to any one of them.

    viz.py         one Meshcat server per session, and HEADLESS
    dtsystems.py   DTSystem: a system described by two functions

Scripts live one directory down (ps2/, ps3/) and are run from there, so each
of them puts this package's parent on `sys.path` before importing from it.
"""
