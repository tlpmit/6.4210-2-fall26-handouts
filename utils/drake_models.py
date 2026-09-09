######################################################################
## Code for nobody but Python to read
######################################################################

"""What to say when Drake cannot download its robot models.

The first `package://drake_models/...` load of a session downloads the models
(about 170 MB, once, into ~/.cache/drake).  Drake does that by handing a
command line to `/bin/sh` with the path of its own helper script pasted in
unquoted -- so a venv under a path that the shell reads as anything other than
one word breaks it, and all the student sees is

    PackageMap: when downloading 'drake_models', the downloader experienced
    an error: returncode == 512

which says nothing about paths.  Seen with Drake 1.56.0 (September 2026); a
later Drake may quote the path, so if this fires on a newer one, first check
that it still doesn't.  A space is the usual culprit -- a clone under
~/Robotic Manipulation/ on any OS, or under /mnt/c/Users/<First Last>/ on WSL,
with the venv inside it.  Quotes, parentheses, `&`, `;`, `$` and backticks do
it too; `.`, `-`, `_` and non-ASCII letters are fine.

`explain_model_download_error` is for the `except` around a pset's first model
load: it says the above, with the two ways out, when that is what happened,
and does nothing otherwise.
"""

import importlib.metadata
import os
import shlex
import subprocess
from pathlib import Path

from pydrake.all import FindResourceOrThrow

# The script Drake runs; the path that ends up unquoted on its command line.
DOWNLOADER = "drake/multibody/parsing/package_downloader.py"

# The Drake this was observed with and written against.
SEEN_WITH = "1.56.0"

_NAMES = {
    " ": "a space",
    "'": "an apostrophe",
    '"': "a double quote",
    "(": "parentheses",
    ")": "parentheses",
    "&": "an ampersand",
    ";": "a semicolon",
    "$": "a dollar sign",
    "`": "a backtick",
    "*": "an asterisk",
    "\\": "a backslash",
}


def shell_splits(path: str) -> bool:
    """Would /bin/sh, given this path unquoted, read it as anything but one
    word?

    Asked of the shell itself rather than of a list of special characters,
    because the list is not the whole story: `a*b` is fine when it matches only
    itself and not when it matches its neighbours.
    """
    result = subprocess.run(
        ["/bin/sh", "-c", f"printf '%s\\n' {path}"],
        capture_output=True,
        text=True,
    )
    return result.stdout != path + "\n"


def explain_model_download_error(e: Exception) -> None:
    """Exit with an explanation if `e` is Drake failing to download its models
    because this venv's path breaks the shell; otherwise return, for the
    caller to re-raise it.
    """
    if "the downloader experienced an error" not in str(e):
        return
    try:
        downloader = os.path.normpath(FindResourceOrThrow(DOWNLOADER))
    except RuntimeError:
        return
    if not shell_splits(downloader):
        return
    share = Path(downloader).parents[3]  # .../pydrake/share
    try:
        installed = importlib.metadata.version("drake")
    except importlib.metadata.PackageNotFoundError:
        installed = "an unknown version"
    culprits = {c for c in downloader if c.isascii() and shlex.quote(c) != c}
    what = " and ".join(sorted({_NAMES.get(c, repr(c)) for c in culprits}))
    raise SystemExit(f"""
{e}

Drake could not download its robot models, because this Python environment
lives at a path with {what} in it:

    {downloader}

Drake pastes that path into a shell command without quoting it (true of Drake
{SEEN_WITH}, which this was written against; you have {installed}), so the shell
does not read it as one word.  Two ways out (README, section 3.3):

  A. Make the venv somewhere with no spaces or special characters, e.g. your
     home directory, and activate that one instead of .venv:

         uv venv --python 3.13 ~/venv64210
         uv pip install --python ~/venv64210 drake
         source ~/venv64210/bin/activate

  B. Keep this venv and give Drake an alias for it at a plain path.  Run
     these once, then open a new terminal:

         ln -sfn {shlex.quote(str(share))} ~/drake-share
         echo 'export DRAKE_RESOURCE_ROOT=$HOME/drake-share' >> ~/.bashrc
""")
