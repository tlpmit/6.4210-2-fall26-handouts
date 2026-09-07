######################################################################
## Code for nobody but Python to read
######################################################################

"""One Meshcat server for the whole of a session, started on the first call.

`ps2/pick_and_place.py` and each of ps3's seven programs carried their own copy
of this at one point, all thirty lines of it.  It lives here instead and they
all import it.

Everything that wants to draw calls `get_meshcat()` rather than being handed a
visualizer, so there is nothing to thread through the code and no way to end up
with a diagram that quietly has no visualizer in it.

    HEADLESS=1 python door_opening.py

runs with no server, no waiting for a browser and no real-time pacing.  Use it
to check that a change actually works -- a run is seconds instead of a minute
-- then run without it to watch.
"""

import os
import time

from pydrake.all import StartMeshcat

# Run without any visualization: no server, no waiting for a browser, and no
# pacing the simulation to wall-clock time.  Set it here, or from the shell as
#     HEADLESS=1 python <file>.py
HEADLESS = bool(os.environ.get("HEADLESS"))

# Colab runs the kernel on a remote VM, so the Meshcat server's localhost URL is
# unreachable from the browser and nothing can ever connect to its websocket.
# Where that changes what the code should do, it tests this.
#
# Importing rather than looking in sys.modules: the module is only in sys.modules
# once something has imported it, so the cheaper test comes out False on Colab
# whenever this file is imported first -- and then hangs for the whole of
# MESHCAT_BROWSER_WAIT on a browser that is never going to arrive.
try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# How long the very first get_meshcat() waits for a browser to connect, so that
# a freshly-opened tab does not miss the beginning of a simulation.
MESHCAT_BROWSER_WAIT = 15.0

_meshcat = None


def get_meshcat():
    """The session's one Meshcat instance, started on first use, or None.

    When HEADLESS, this is None -- which is exactly what MakeHardwareStation
    and MeshcatVisualizer.AddToBuilder already take to mean "add no
    visualizer", so callers do not have to test for it.

    On Colab there is no browser to wait for, so it does not: the scene is
    drawn after the fact by `show_meshcat()`.
    """
    global _meshcat
    if HEADLESS:
        return None
    if _meshcat is not None:
        return _meshcat

    _meshcat = StartMeshcat()
    if IN_COLAB:
        return _meshcat

    print(f"Check your Meshcat window ({_meshcat.web_url()})")
    deadline = time.time() + MESHCAT_BROWSER_WAIT
    while time.time() < deadline and _meshcat.GetNumActiveConnections() == 0:
        time.sleep(0.1)
    if _meshcat.GetNumActiveConnections() == 0:
        print("   ...no browser connected; running anyway.")
    return _meshcat


def current_meshcat():
    """The instance if one has already been started, else None.

    Unlike `get_meshcat()` this never starts a server, so it is the right way
    to ask "is anybody watching?" -- which is what code that only wants to
    record an animation, rather than draw one, needs to know.
    """
    return _meshcat


def start_recording(**kwargs):
    """Begin recording an animation, if there is a visualizer to record into.

    Keyword arguments go straight through to `Meshcat.StartRecording`, so a
    caller that wants `frames_per_second` or `set_visualizations_while_recording`
    can still say so.

    This and `publish_recording()` exist because the pair of them is otherwise
    four lines of `if meshcat is not None` around every simulation in the
    course.  Nothing is recorded when nobody is watching, and no caller has to
    write that down.
    """
    meshcat = current_meshcat()
    if meshcat is not None:
        meshcat.StartRecording(**kwargs)


def publish_recording():
    """Stop recording and hand the animation to the browser, if recording.

    Stopping first is what makes the animation a finished thing rather than
    one still accumulating frames; publishing is what puts a scrubber on the
    Meshcat page.
    """
    meshcat = current_meshcat()
    if meshcat is not None:
        meshcat.StopRecording()
        meshcat.PublishRecording()


def show_meshcat(height=500):
    """Draw the scene, and any recorded animation, inline in a notebook cell.

    `StaticHtml()` is a complete self-contained page -- the geometry and the
    animation are baked into it -- so this needs no connection back to the
    server and works where a live Meshcat window cannot.  Sliders and buttons
    are dropped, since those do rely on the server.

    A snapshot, not a live view: call it after the simulation has run.  Safe to
    call from anywhere: outside a notebook it does nothing, because there the
    live Meshcat window is already showing the same thing.
    """
    import html

    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        return
    if get_ipython() is None:
        return

    if _meshcat is None:
        print("Nothing to show: no simulation has been run with a visualizer.")
        return
    display(
        HTML(
            f'<iframe srcdoc="{html.escape(_meshcat.StaticHtml())}" '
            f'width="100%" height="{height}" style="border:1px solid #ccc">'
            f"</iframe>"
        )
    )


def keep_meshcat_open():
    """Block at the end of __main__ so the finished animation stays on screen.

    Meshcat lives inside this process; when it exits the page goes dead.  If
    the run never started a server -- HEADLESS, or a test that only prints
    numbers -- there is nothing to keep open and this returns at once.

    On Colab there is no live window to hold open, and a cell waiting on
    `input()` would only sit there, so this does nothing there either.
    """
    if _meshcat is not None and not IN_COLAB:
        input("Press Enter to quit.")


def sim_duration(duration: float, headless_duration: float = 0.1) -> float:
    """How long to simulate for: the real thing, or a token step when HEADLESS.

    This is the notebooks' `5.0 if running_as_notebook else 0.1` idiom.  The
    short run still builds the diagram and evaluates every system once, which
    is what catches the wiring mistakes; only the watching is skipped.
    """
    return headless_duration if HEADLESS else duration
