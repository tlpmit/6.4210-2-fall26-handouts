# 6.4210/6.4212 — Robotic Manipulation, Fall 2026: problem set handouts

This repository holds the problem sets: for each pset, a directory `psN/` with
the assignment PDF (`psN.pdf`), the Python code you will read and extend, and
a Colab notebook (`psN_colab.ipynb`).
`utils/` holds code shared across psets.
New psets, and fixes to released ones, appear here as ordinary git
commits, and you get them with `git pull`.

You can work **locally** (recommended) or on **Google Colab**; both paths are
below.  Submission is always through Gradescope, per the pset PDF. 
GitHub is not used for grading.

## 1. Prerequisites (local)

You need `git` and a GitHub account ([install git](https://github.com/git-guides/install-git)),
plus two tools:

- the [uv](https://docs.astral.sh/uv/) package manager, which will manage
  Python and the packages for you:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- [Graphviz](https://graphviz.org/download/) (the program, not the Python
  package), to render system block diagrams: `brew install graphviz` (Mac) or
  `sudo apt install graphviz` (Ubuntu/WSL).

**Windows:** there are no Windows builds of Drake.  Install
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install) with its default
Ubuntu distribution (`wsl --install` in an administrator PowerShell), then
follow the Linux instructions inside it.

## 2. Get the code and keep your work in a *private* repo

> **Do not fork this repository.**  Forks of public repos are public forever
> and can never be made private, so a fork would publish your homework solutions publicly.

Instead, clone it and point it at a private repo of your own (one-time setup):

```sh
git clone https://github.com/tlpmit/6.4210-2-fall26-handouts.git 64210
cd 64210
git config pull.rebase false    # so `git pull` merges
git remote rename origin upstream
# on github.com: create an empty PRIVATE repository (no README), then:
git remote add origin git@github.com:<you>/<your-private-repo>.git
git push -u origin main
```

From then on, `git push` backs your work up to your private repo.  (If you
work on a single machine and don't want the backup, a plain `git clone` works
too — your commits just stay local.)

## 3. Create the Python environment

Inside the clone, make a virtual environment with the
[drake](https://drake.mit.edu/installation.html) package (the `.venv`
directory is gitignored, so it stays out of your commits):

```sh
cd 64210
uv python install 3.13
uv venv --python 3.13
uv pip install drake
```

## 4. Working on a pset

Activate the venv in each shell you work in, then run the pset's files from
its directory:

```sh
cd 64210
source .venv/bin/activate
cd ps1
python cartesian_2d_robot.py
```

Fill in the function bodies marked

```python
raise NotImplementedError("your code here")
```

and add helper functions and your own test calls in the same files.

## 5. Getting updates (new psets, fixes to released ones)

Commit your work first, then pull:

```sh
git add -A && git commit -m "wip"
git pull upstream main        # for a plain clone with no private repo use: git pull
```

## 6. Working on Google Colab

**First time**: open the pset's notebook straight from GitHub, e.g. for ps1:

> https://colab.research.google.com/github/tlpmit/6.4210-2-fall26-handouts/blob/main/ps1/ps1_colab.ipynb

Run the first two code cells: one mounts your **Google Drive**, the next fetches
the course code into it (`MyDrive/6.4210/repo`) — the code and your edits
live in Drive, so they survive Colab runtime resets.  Then **switch to the
notebook inside your Drive**: on [drive.google.com](https://drive.google.com)
open `My Drive → 6.4210 → repo → ps1` and double-click `ps1_colab.ipynb` —
it opens in Colab.  Work in that copy from then on: it lives inside your
repo, so everything you do in the notebook is saved to Drive and can be
committed with git.  The GitHub copy you started from is *not* part of your
repo, and notebook work done there is easy to lose.

**Every session after that**: open the notebook from your Drive (double-click
it on drive.google.com, or in Colab use File → Open notebook → Google Drive)
and run the setup cells at the top.  The mount and fetch are no-ops once
done; the `setup()` cell installs Drake and configures git (a default identity and merge-style pulls) so that the git commands below work.

**Recommended one-time setup** so Colab uses your private repo from §2: in
Colab, open the key icon (Secrets) in the left sidebar and add, with notebook
access enabled:

- `PRIVATE_REPO` — e.g. `yourname/your-private-repo`
- `GH_TOKEN` — a [fine-grained personal access
  token](https://github.com/settings/personal-access-tokens): Repository
  access limited to that repo, with Contents permission Read and write.

With the secrets set, the bootstrap clones *your* repo (do §2 first so it
exists) with the public repo wired as `upstream`, and pushing works from
Colab.  Without them, it clones the public repo — everything works, your work
still persists in Drive, there's just no GitHub backup.

Day to day on Colab:

- Edit the `.py` files in the file browser on the left (double-click a file,
  Ctrl-S saves to Drive).
- After editing, run `colab_bootstrap.refresh()` so the next import re-reads
  your changes, then re-run your cells.
- Git works the same as on a local machine.  Colab's terminal is a paid
  feature, but every cell runs shell commands with a `!` prefix.  Start by
  moving the session into the repo — `%cd` persists for the whole session,
  unlike `!cd`, which lasts only for its own line:

  ```
  %cd /content/drive/MyDrive/6.4210/repo
  ```

  Commit to save your work to git, and push if you set up the private repo (§2):

  ```
  !git add -A && git commit -m "ps1 progress"
  !git push origin main
  ```

  When a new pset comes out, or a released one is fixed, update the same way
  as in §5 — commit your work first, then:

  ```
  !git pull upstream main
  ```

---

The course code and materials in this repository are provided under the BSD
3-Clause license (see [LICENSE](LICENSE)).  That license does not change the
course policy above: keep your own solutions in a private repository.
