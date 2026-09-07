######################################################################
## Code for nobody but Python to read
######################################################################

"""matplotlib, with a backend chosen so that it survives running with Drake.

Import `plt` from here rather than from matplotlib:

    from utils.plotting import plt

Why this file exists.  Drake's camera rendering goes through VTK, and on macOS
VTK starts up Cocoa's one `NSApplication`.  matplotlib's default backend there
is Tk, and Tk assumes the running NSApplication is its own `TKApplication`
subclass: the first time it opens a window it sends that object
`-macOSVersion`, VTK's application has no such method, and the process dies at
the first `plt.figure()` with

    *** Terminating app due to uncaught exception 'NSInvalidArgumentException',
    reason: '-[NSApplication macOSVersion]: unrecognized selector sent to
    instance ...'

That is an abort inside Objective-C, not a Python exception, so no try/except
can catch it and no amount of care in our own code avoids it -- the backend has
to be settled before the first figure is drawn.  Hence a module to import.

What it actually changes, and where.  The clash needs all three of macOS, Tk,
and Drake, so this touches nothing else:

  - Linux and Windows have no NSApplication.  matplotlib's own choice stands,
    including its existing fallback to a non-interactive backend when there is
    no display to draw on.
  - A backend that is not Tk -- Qt, a notebook's inline backend, whatever a
    student has configured -- is already clear of the problem and is left be.
  - MPLBACKEND set in the environment is a deliberate choice and is honoured,
    even on macOS.  A student who sets it to Tk there gets the crash back; that
    is the price of it meaning what it says.

Only when all of that leaves us on macOS with Tk does this switch: to the
native "macosx" backend, which draws real windows and does not involve Tk at
all, or -- if that is missing, as on a stripped or headless build -- to "Agg",
where nothing opens on screen and figures have to be written out with
`plt.savefig()`.  The one time it makes a choice a student might be surprised
by, it says so.
"""

import os
import sys

import matplotlib


def _choose_backend() -> None:
    if sys.platform != "darwin":
        return
    if os.environ.get("MPLBACKEND"):
        return
    # Safe to ask: resolving the name imports the backend module but does not
    # yet build a Tk application, which is the step that would abort.
    if "tk" not in matplotlib.get_backend().lower():
        return

    for candidate in ("macosx", "Agg"):
        try:
            matplotlib.use(candidate)
        except Exception:
            continue
        if candidate == "Agg":
            print(
                "No usable interactive matplotlib backend: drawing to files only. "
                "plt.show() will do nothing; use plt.savefig('name.png')."
            )
        return


_choose_backend()

import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt"]
