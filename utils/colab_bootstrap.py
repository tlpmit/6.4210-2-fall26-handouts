######################################################################
## Code for nobody but Python to read
######################################################################

"""Set up a Google Colab session, for any pset.

Each pset's generated notebook (psN_colab.ipynb) starts with the marked
regions of `bootstrap()` below, lifted out verbatim and in order by
tools/make_handout.py -- they cannot be imported from here, because they are
the code that fetches this file.  The next cell imports this module and calls
`setup("psN")`; nothing in this file is specific to one pset.  After that,
the only helper you need is

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

# This file lives at <repo>/utils/colab_bootstrap.py wherever the repo was
# cloned, so everything below is derived from its own location and nothing
# has to be kept in sync with the bootstrap cell's clone path.
ROOT = pathlib.Path(__file__).resolve().parents[1]


def bootstrap() -> None:
    """
    The notebook's first cells: each marked region below is lifted out
    verbatim, in order -- the Drive mount alone, then the fetch.  They run
    before this module exists on the machine, so they are fully
    self-contained: their own imports, no names from the rest of this file.
    """
    # Mount Google Drive. This is where we will/have clone(d) your course repo to.
    from google.colab import drive

    drive.mount("/content/drive")

    # Fetch the course code into Drive and put it on the import path.  With
    # the Colab secrets GH_TOKEN and PRIVATE_REPO set (see the course
    # README), this uses your private homework repo; without them, a plain
    # read-only clone of the public handouts repo.
    import pathlib
    import subprocess
    import sys

    from google.colab import userdata

    root = pathlib.Path("/content/drive/MyDrive/6.4210/repo")
    public = "https://github.com/tlpmit/6.4210-2-fall26-handouts.git"
    try:
        private = userdata.get("PRIVATE_REPO")
        origin = f"https://github.com/{private}.git"
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
        print(f"Colab secrets found: your repo is {private}")
    except Exception:
        origin = None
        print("no Colab secrets configured: "
              "using the public repo, read-only")
    if root.exists():
        print(f"course code already in Drive at {root}")
    else:
        print(f"cloning {'your private repo' if origin else 'the public repo'}: "
              f"{origin or public}")
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", origin or public, str(root)], check=True)
        if origin:
            subprocess.run(
                ["git", "-C", str(root), "remote", "add", "upstream", public],
                check=True,
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    print("done: the course code is on the import path")


def setup(pset: str) -> None:
    """
    Make this session ready to work on the given pset (e.g. setup("ps1")):
    put <repo>/<pset> on the import path so its modules import as top-level
    names, install Drake if this session does not have it yet, and configure
    git so that `!git commit` and `!git pull` work on a fresh runtime: an
    identity if it has none, and merge-style pulls (without that, git refuses
    to pull once you have local commits and upstream has moved).

    Drake is a ~1 GB wheel, so this is the slow cell; it survives Restart
    session but not Disconnect and delete runtime.  The iiwa meshes are not in
    the wheel --- Drake fetches `package://drake_models` on first use, which
    adds a pause to the first test that draws the arm.
    """
    pset_dir = ROOT / pset
    if not pset_dir.is_dir():
        raise ValueError(f"no pset directory {pset_dir}; is the clone healthy?")
    if str(pset_dir) not in sys.path:
        sys.path.insert(0, str(pset_dir))
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
    head = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--oneline", "-1"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"ready: {ROOT} at commit: {head}")


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
