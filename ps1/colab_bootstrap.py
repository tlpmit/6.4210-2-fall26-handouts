######################################################################
## Code for nobody but Python to read
######################################################################

"""Set up a Google Colab session for this pset.

Cell 1 of the generated notebook (ps1_colab.ipynb) is the body of
`bootstrap()` below, lifted out verbatim by tools/make_handout.py -- it cannot
be imported from here, because it is the code that fetches this file.  Cell 2
imports this module and calls `setup()`.  After that, the only helper you
need is

    refresh()    # after editing a .py file: forget it, so re-import re-reads

The bootstrap clones the course code into Google Drive (MyDrive/6.4210/repo),
so the checkout -- and your edits -- survive runtime resets.  With the Colab
secrets GH_TOKEN and PRIVATE_REPO set (see the course README), it clones your
private homework repo and wires the public handouts repo as `upstream`; with
no secrets it clones the public repo directly, and everything works except
pushing a backup to GitHub.

Git is deliberately not wrapped here.  Colab's terminal is a paid feature,
but every cell runs shell commands with a `!` prefix, so use git directly:

    %cd /content/drive/MyDrive/6.4210/repo
    !git add -A && git commit -m "ps1 progress"
    !git pull upstream main
    !git push origin main
"""

import importlib.util
import pathlib
import subprocess
import sys

# This file lives at <repo>/<pset>/colab_bootstrap.py wherever the repo was
# cloned, so everything below is derived from its own location and nothing
# has to be kept in sync with the bootstrap cell's clone path.
ROOT = pathlib.Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    """
    The notebook's first cell: the marked region is lifted out verbatim.  It
    runs before this module exists on the machine, so it is fully
    self-contained -- its own imports, no names from the rest of this file.
    """
    # Fetch the course code into Google Drive (where it survives runtime
    # resets) and put it on the import path.  With the Colab secrets GH_TOKEN
    # and PRIVATE_REPO set (see the course README), this uses your private
    # homework repo; without them, a plain clone of the public handouts repo.
    import pathlib
    import subprocess
    import sys

    from google.colab import drive, userdata

    drive.mount("/content/drive")
    root = pathlib.Path("/content/drive/MyDrive/6.4210/repo")
    public = "https://github.com/tlpmit/6.4210-2-fall26-handouts.git"
    try:
        origin = f"https://github.com/{userdata.get('PRIVATE_REPO')}.git"
        # The token goes in a credential file outside the repo, so it never
        # ends up in .git/config or in what git prints when a fetch fails.
        creds = pathlib.Path("/root/.git-credentials")
        creds.write_text(
            f"https://x-access-token:{userdata.get('GH_TOKEN')}@github.com\n"
        )
        creds.chmod(0o600)
        subprocess.run(
            ["git", "config", "--global",
             "credential.helper", f"store --file={creds}"],
            check=True,
        )
    except Exception:
        origin = None  # no secrets configured: read-only public clone
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", origin or public, str(root)], check=True)
        if origin:
            subprocess.run(
                ["git", "-C", str(root), "remote", "add", "upstream", public],
                check=True,
            )
    for p in (str(root), str(root / "ps1")):
        if p not in sys.path:
            sys.path.insert(0, p)


def setup() -> None:
    """
    Install Drake if this session does not have it yet, and configure git so
    that `!git commit` and `!git pull` work on a fresh runtime: an identity if
    it has none, and merge-style pulls (without that, git refuses to pull once
    you have local commits and upstream has moved).

    Drake is a ~1 GB wheel, so this is the slow cell; it survives Restart
    session but not Disconnect and delete runtime.  The iiwa meshes are not in
    the wheel --- Drake fetches `package://drake_models` on first use, which
    adds a pause to the first test that draws the arm.
    """
    if importlib.util.find_spec("pydrake") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "drake"],
            check=True,
        )
    if subprocess.run(
        ["git", "config", "--global", "user.name"], capture_output=True
    ).returncode:
        subprocess.run(
            ["git", "config", "--global", "user.name", "Colab student"], check=True
        )
        subprocess.run(
            ["git", "config", "--global", "user.email", "student@colab"], check=True
        )
    subprocess.run(
        ["git", "config", "--global", "pull.rebase", "false"], check=True
    )
    print(f"ready: {ROOT} at", end=" ")
    subprocess.run(
        ["git", "-C", str(ROOT), "--no-pager", "log", "--oneline", "-1"], check=True
    )


def refresh() -> None:
    """
    Forget our modules, so the next import re-reads the edited files.

    `%autoreload` would be the usual answer and is not available: Colab's
    IPython still opens autoreload.py with `from imp import reload`, and `imp`
    was removed in Python 3.13, which is what Colab now runs.
    """
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and pathlib.Path(f).is_relative_to(ROOT):
            del sys.modules[name]
