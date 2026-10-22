"""Analyze packaged reference runs without Drake or policy training.

Evaluation goals are repeated measurements. Training-run inference uses one
mean paired difference per independently trained PPO policy, conditional on
one supplied BC checkpoint and one fixed benchmark of goals.
"""

from pathlib import Path
import json, hashlib
import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent / "data/cautious_push"
METRICS = ["success", "reward", "execution_seconds", "pushes", "distance"]


def read(p):
    return json.loads(Path(p).read_text())


def verify_data():
    checks = read(DATA / "checksums.json")
    for name, h in checks.items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != h:
            raise ValueError("Packaged file changed: " + name)
    return dict(files=len(checks), intact=True)


def reference(cohort="exploratory"):
    if cohort not in ["exploratory", "confirmatory"]:
        raise ValueError("Unknown cohort")
    folder = DATA / cohort
    plan = read(folder / "plan.json")
    status = read(folder / "status.json")
    if status["state"] != "complete":
        raise RuntimeError(
            "The fixed reference cohort is incomplete; do not pool or drop missing training runs. Use the completed exploratory study for interim analysis."
        )
    runs = [read(folder / f"seed{s}.json") for s in plan["seeds"]]
    baselines = {n: read(folder / f"{n}.json")["rows"] for n in ["bc", "bc_sampled"]}
    ids = plan["test_goals"]
    if any([r["seed"] for r in rows] != ids for rows in baselines.values()):
        raise ValueError("Unpaired baseline goals")
    for run in runs:
        if run["steps"] < plan["budget"]:
            raise ValueError("A training run ended early")
        for rows in run["evaluations"].values():
            if [r["seed"] for r in rows] != ids:
                raise ValueError("Unpaired policy goals")
    return dict(cohort=cohort, plan=plan, runs=runs, baselines=baselines)


def means(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in METRICS}


def best_monitoring_policy(study):
    """Choose an illustrative checkpoint using monitoring goals only.

    Maximize monitoring success, then reward; break ties by earlier training
    step and lower seed. This selection never changes the fixed-final primary
    statistical comparison. The demonstration goal is fixed independently.
    """
    candidates = []
    for run in study["runs"]:
        selected = min(
            run["history"], key=lambda h: (-h["success"], -h["reward"], h["steps"])
        )
        if selected["steps"] != run["selected_steps"]:
            raise ValueError("Saved checkpoint selection disagrees with monitoring")
        candidates.append(
            dict(
                seed=run["config"]["seed"],
                steps=selected["steps"],
                success=selected["success"],
                reward=selected["reward"],
                theta=selected["theta"],
            )
        )
    chosen = min(
        candidates, key=lambda h: (-h["success"], -h["reward"], h["steps"], h["seed"])
    )
    return dict(
        cohort=study["cohort"],
        **chosen,
        checkpoint=f"checkpoints/seed{chosen['seed']}_best.pt",
        goal_seed=9460000,
        selection="Best monitoring success then reward, earlier step then lower seed on ties; no test-goal or replay-outcome selection. Illustrative replay only; the primary comparison still uses all fixed final policies.",
    )


def score_table(study, variant="final"):
    import pandas as pd

    rows = [
        dict(method=n, training_seed=None, **means(v))
        for n, v in study["baselines"].items()
    ]
    rows += [
        dict(
            method="residual_ppo",
            training_seed=r["config"]["seed"],
            **means(r["evaluations"][variant]),
        )
        for r in study["runs"]
    ]
    return pd.DataFrame(rows)


def aggregate_table(study):
    import pandas as pd

    rows = [dict(policy=n, **means(v)) for n, v in study["baselines"].items()]
    for variant in ["midpoint", "final", "sampled", "selected", "midpoint_sampled"]:
        rows.append(
            dict(
                policy=variant,
                **{
                    k: float(
                        np.mean(
                            [means(r["evaluations"][variant])[k] for r in study["runs"]]
                        )
                    )
                    for k in METRICS
                },
            )
        )
    return pd.DataFrame(rows)


def training_comparison(study, *, resamples=5000, seed=815):
    """Final reward minus BC; independent unit = one PPO training run.

    The t-test concerns the mean difference (normal-reference exact under iid
    normal differences, approximate otherwise). The exact sign test concerns
    improvement direction. Neither test generalizes over new BC fits or goals.
    Do not use favorable checkpoints or select the cohort by the resulting p.
    """
    bc = means(study["baselines"]["bc"])["reward"]
    d = np.array(
        [means(r["evaluations"]["final"])["reward"] - bc for r in study["runs"]]
    )
    if len(d) < 2:
        raise ValueError("At least two independent training runs needed")
    t = stats.ttest_1samp(d, 0.0)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (resamples, len(d)))].mean(1)
    nz = d[d != 0]
    sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
    return dict(
        n=len(d),
        differences=d.tolist(),
        difference=float(d.mean()),
        training_seed_sd=float(d.std(ddof=1)),
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        t=float(t.statistic),
        df=len(d) - 1,
        p=float(t.pvalue),
        exact_sign_p=float(sign),
        interpretation="Across independent PPO training runs, conditional on the supplied BC and fixed test goals.",
    )


def paired_goal_interval(
    study, *, variant="final", metric="reward", resamples=5000, seed=701, compare="bc"
):
    """Paired GOAL interval for fixed fitted policies, not seed uncertainty.

    compare='midpoint' compares final/midpoint policies within each training seed;
    compare='bc' uses greedy BC, or matched sampled BC for sampled variants.
    """
    policy = np.array(
        [[row[metric] for row in r["evaluations"][variant]] for r in study["runs"]]
    )
    if compare == "midpoint":
        mid = "midpoint_sampled" if variant == "sampled" else "midpoint"
        other = np.array(
            [[row[metric] for row in r["evaluations"][mid]] for r in study["runs"]]
        ).mean(0)
    elif compare == "bc":
        other = np.array(
            [
                row[metric]
                for row in study["baselines"][
                    "bc_sampled" if variant in ["sampled", "midpoint_sampled"] else "bc"
                ]
            ]
        )
    else:
        raise ValueError("Compare against bc or midpoint")
    delta = policy.mean(0) - other
    rng = np.random.default_rng(seed)
    boots = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(1)
    return dict(
        difference=float(delta.mean()),
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        goals=len(delta),
        interpretation="Across paired goals, conditional on these fitted policies. These goals are not independent training runs.",
    )


def learning(study):
    import matplotlib.pyplot as plt
    from learning_curves import mean_band

    runs = study["runs"]
    end = min(r["history"][-1]["steps"] for r in runs)
    grid = np.linspace(0, end, 120)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    for ax, key, label in zip(
        axes,
        ["theta", "reward", "execution_seconds"],
        ["Residual theta", "Monitoring reward", "Monitoring simulated seconds"],
    ):
        matrix = []
        for r in runs:
            h = r["history"]
            xs = [x["steps"] for x in h]
            ys = [x[key] for x in h]
            ax.plot(xs, ys, alpha=0.3)
            matrix.append(np.interp(grid, xs, ys))
        mean, lo, hi = mean_band(matrix, resamples=5000, seed=816)
        ax.plot(grid, mean, color="black", label="Mean across training runs")
        ax.fill_between(
            grid,
            lo,
            hi,
            color="gray",
            alpha=0.2,
            label="Pointwise training-run bootstrap95% interval",
        )
        ax.axvline(4096, color="black", ls=":", alpha=0.4)
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Online training pushes")
    return fig
