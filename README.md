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
git config pull.rebase false    # so `git pull` merges (see §5)
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

Open the pset's notebook straight from GitHub, e.g. for ps1:

> https://colab.research.google.com/github/tlpmit/6.4210-2-fall26-handouts/blob/main/ps1/ps1_colab.ipynb

The first cell clones the course code into your **Google Drive**
(`MyDrive/6.4210/repo`), so the code — and your edits — survive Colab runtime
resets.  The second cell installs Drake (the slow one, ~2 min on a fresh
runtime) and configures git — a default identity and merge-style pulls — so
the git commands below just work.

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

  If the pull reports a conflict, resolve it as in §5, editing the conflicted
  files in the file browser.

---

The course code and materials in this repository are provided under the BSD
3-Clause license (see [LICENSE](LICENSE)).  That license does not change the
course policy above: keep your own solutions in a private repository.
