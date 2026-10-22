"""Uncertainty and paired inference with TRAINING RUNS as sampling units.

Endpoint episodes within a trained policy are repeated measurements, not
independent algorithm replicates. All intervals here resample whole runs (or
whole paired runs); pointwise curve intervals are not simultaneous bands.
"""
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import stats


def mean_band(values, method='bootstrap', *, resamples=5000, seed=812):
    """Columnwise means and pointwise intervals, rows = independent runs.

    bootstrap: percentile 95% interval, resampling entire rows together.
    stderr: mean +/- ONE estimated standard error, not a 95% interval.
    With one run the uncertainty is undefined, represented by NaN bounds.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1: x = x[:, None]
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('values must be a finite, nonempty run-by-checkpoint array')
    if method not in ('bootstrap', 'stderr'):
        raise ValueError('method must be bootstrap or stderr')
    mean = x.mean(axis=0)
    if len(x) == 1:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    if method == 'stderr':
        se = x.std(axis=0, ddof=1) / np.sqrt(len(x))
        return mean, mean - se, mean + se
    if resamples < 1: raise ValueError('resamples must be positive')
    rng = np.random.default_rng(seed)
    boot = np.empty((resamples, x.shape[1]))
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        indices = rng.integers(len(x), size=(count, len(x)))
        boot[start:start+count] = x[indices].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975], axis=0)
    return mean, low, high


def paired_comparison(plain, baseline, *, permutations=99_999, seed=813):
    """Compare matched independent TRAINING RUN means, baseline minus plain.

    The randomization null requires within-pair label exchangeability; equality
    of means alone does not imply it. The paired t-test concerns mean difference
    zero, with an exact t reference for iid normal differences (approximate
    otherwise). We report both, without using a normality test to pick one.
    """
    a, b = np.asarray(plain, float), np.asarray(baseline, float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError('Need at least two complete pairs of run means')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('All run means must be finite')
    d = b-a
    n = len(d)
    observed = float(d.mean())
    _, low, high = mean_band(d, seed=seed)
    observed_abs = abs(observed)
    tolerance = 1e-12 * max(1., observed_abs)
    extreme, total = 0, 0
    exact = n <= 16
    rng = np.random.default_rng(seed)
    count = 2**n if exact else permutations
    if count < 1: raise ValueError('permutations must be positive')
    for start in range(0, count, 2048):
        size = min(2048, count-start)
        if exact:
            signs = 2*((np.arange(start, start+size)[:, None] >> np.arange(n)) & 1)-1
        else:
            signs = rng.integers(0, 2, (size, n))*2-1
        null = (signs*d).mean(axis=1)
        extreme += int(np.count_nonzero(abs(null) >= observed_abs-tolerance))
        total += size
    p = extreme/total if exact else (extreme+1)/(total+1)
    sd = float(d.std(ddof=1))
    if sd == 0:
        t, tp = (0., 1.) if observed == 0 else (None, None)
        note = 'Degenerate sample: no estimated variation; t approximation not informative.'
    else:
        t = observed/(sd/np.sqrt(n))
        tp = float(2*stats.t.sf(abs(t), df=n-1))
        note = 'Paired t reference: exact for normal differences, approximate otherwise.'
    return dict(n_pairs=n, mean_plain=float(a.mean()), mean_baseline=float(b.mean()),
                mean_difference=observed, paired_bootstrap_ci95=[float(low[0]), float(high[0])],
                difference_sd=sd, difference_standard_error=sd/np.sqrt(n),
                paired_permutation_p=float(p), permutation_draws=total, permutation_exact=exact,
                permutation_null='Within-pair labels exchangeable; not merely equal means.',
                paired_t_statistic=t, paired_t_p=tp, paired_t_note=note,
                differences=d.tolist(), zero_differences=int(np.count_nonzero(d == 0)),
                alternative='two-sided', alpha=.05)


def load_paired_runs(directory):
    """Reject incomplete, duplicate, or incompatible comparisons explicitly."""
    groups = {False: {}, True: {}}
    for path in sorted(Path(directory).glob('reinforce_*_seed*.json')):
        run = json.loads(path.read_text())
        group = groups[run['config']['baseline']]
        if run['seed'] in group: raise ValueError('Duplicate training seed')
        group[run['seed']] = run
    if not groups[False] or groups[False].keys() != groups[True].keys():
        raise ValueError('Require matching complete pairs of training seeds')
    seeds = sorted(groups[False])
    plan_path = Path(directory) / 'study_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if seeds != sorted(plan['training_seeds']):
            raise ValueError('Planned study is incomplete; do not test or silently drop missing runs')
    reference = None
    for seed in seeds:
        for baseline in [False, True]:
            run = groups[baseline][seed]
            cfg = {k:v for k,v in run['config'].items() if k != 'baseline'}
            signature = (cfg, run['environment'])
            if reference is None: reference = signature
            if signature != reference:
                raise ValueError('Do not pool different training or environment configurations')
            if run['learning'][-1]['update'] != cfg['updates']:
                raise ValueError('A run ended before its planned training endpoint')
        if groups[False][seed]['final_evaluation_seeds'] != groups[True][seed]['final_evaluation_seeds']:
            raise ValueError('Evaluation seeds must match within a pair')
    return seeds, [[groups[mode][s] for s in seeds] for mode in [False, True]]


def curve_arrays(groups, x='update', points=150):
    """Align on common checkpoints, or interpolate inside the shared x range.

    No extrapolation and no changing cohort as training runs finish. Interactions
    and elapsed time are descriptive interpolations; testing uses saved endpoint
    run means, never whichever plot location happens to separate most.
    """
    if x not in ('update', 'steps', 'seconds'):
        raise ValueError('x must be update, steps, or seconds')
    all_runs = [r for g in groups for r in g]
    if x == 'update':
        grid = np.asarray([r['update'] for r in all_runs[0]['learning']], float)
        for run in all_runs:
            if not np.array_equal(grid, [r['update'] for r in run['learning']]):
                raise ValueError('Update checkpoints must match')
    else:
        start = max(r['learning'][0][x] for r in all_runs)
        end = min(r['learning'][-1][x] for r in all_runs)
        if end <= start: raise ValueError('No common curve range')
        grid = np.linspace(start, end, points)
    arrays = []
    for group in groups:
        ys = []
        for run in group:
            xs = np.asarray([r[x] for r in run['learning']])
            if np.any(np.diff(xs) <= 0): raise ValueError('Curve x values must increase')
            ys.append(np.interp(grid, xs, [r['stochastic_mean'] for r in run['learning']]))
        arrays.append(np.asarray(ys))
    return grid, arrays


def plot_learning_curves(directory, *, band='bootstrap', individual=True, show=True, output_dir=None):
    """Plot evaluation return versus policy updates, and save the figure to PNG.

    All runs have the same update checkpoints. Thin lines show individual runs;
    thick lines show means, with bootstrap CI or +/-1 SE across training runs.
    A single training pair has no uncertainty band.
    Returns a matplotlib Figure. In a notebook, simply leave the call as the
    cell's last expression. Set show=False for batch scripts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    directory = Path(directory)
    output_dir = Path(output_dir or directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds, groups = load_paired_runs(directory)
    if len(seeds) == 1:
        warnings.warn('One training pair: no uncertainty band can be estimated across runs.')
    elif len(seeds) < 10:
        warnings.warn(f'Only {len(seeds)} independent run pairs: uncertainty estimates are exploratory.')
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout='constrained')
    colors = ['#2463a6', '#c65b19']
    labels = ['REINFORCE', '+ value baseline']
    batch = groups[0][0]['config']['batch_episodes']
    horizon = groups[0][0]['environment']['horizon']
    grid, arrays = curve_arrays(groups, 'update')
    for values, color, label in zip(arrays, colors, labels):
        mean, lo, hi = mean_band(values, band)
        if individual:
            for ys in values:
                ax.plot(grid, ys, color=color, alpha=.65 if len(seeds) == 1 else .22,
                        lw=.9, zorder=1)
        ax.plot(grid, mean, color=color, lw=2.3, label=label, zorder=4)
        if len(seeds) > 1:
            ax.fill_between(grid, lo, hi, color=color, alpha=.22, zorder=3)
    ax.set(xlabel=f'Policy updates ({batch} episodes each)',
           ylabel='Mean stochastic capped episode length', ylim=(0, 1.03*horizon))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    if len(seeds) == 1:
        fig.suptitle('1 training pair; no confidence band (one run per method)')
    else:
        kind = 'pointwise 95% bootstrap CI' if band == 'bootstrap' else '+/- 1 standard error (not 95% CI)'
        fig.suptitle(f'{len(seeds)} independent training pairs; {kind}')
    if individual and len(seeds) > 1:
        fig.supxlabel('Thin lines: individual training runs; thick lines: means.', fontsize=9)
    fig.savefig(output_dir / 'learning_uncertainty.png', dpi=170)
    if not show: plt.close(fig)
    return fig


def compare_runs(directory, *, show=True, output_dir=None):
    """Save inference, per-run CSV, and plots exposing capping and clustering."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    output_dir = Path(output_dir or directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds, groups = load_paired_runs(directory)
    scores = [np.asarray([r['final']['stochastic'] for r in group], float) for group in groups]
    means = [s.mean(axis=1) for s in scores]
    result = paired_comparison(*means)
    result.update(training_seeds=seeds, unit='Mean of evaluation episodes from one independently trained policy',
                  episodes_per_run=[scores[0].shape[1], scores[1].shape[1]],
                  endpoint_updates=groups[0][0]['config']['updates'],
                  fraction_at_cap=[float((s == groups[0][0]['environment']['horizon']).mean()) for s in scores])
    (output_dir / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    table = np.column_stack([seeds, *means, means[1]-means[0]])
    np.savetxt(output_dir / 'run_means.csv', table, delimiter=',',
               header='training_seed,plain_mean,baseline_mean,baseline_minus_plain', comments='',
               fmt=['%d', '%.8f', '%.8f', '%.8f'])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), layout='constrained')
    colors = ['#2463a6', '#c65b19']
    labels = ['REINFORCE', '+ value baseline']
    for score, mean, color, label in zip(scores, means, colors, labels):
        x = np.sort(score.ravel())
        axes[0].step(x, np.arange(1, len(x)+1)/len(x), where='post', color=color, label=label)
        axes[1].scatter(np.arange(len(seeds)), mean, s=22, color=color, label=label)
    for i in range(len(seeds)):
        axes[1].plot([i, i], [means[0][i], means[1][i]], color='gray', lw=.7, zorder=0)
    d = means[1]-means[0]
    axes[2].hist(d, bins=min(12, len(seeds)), color='#526777', edgecolor='white')
    axes[2].axvline(0, color='black', lw=1)
    axes[0].set(xlabel='Capped episode length', ylabel='Fraction at or below length',
                title='Episode ECDF (descriptive; clustered)')
    axes[1].set(xlabel='Independent training pair', ylabel='Mean of held-out episodes',
                title=f'{len(seeds)} training pairs, not {scores[0].size} independent runs')
    axes[2].set(xlabel='Baseline minus plain run mean', ylabel='Number of pairs',
                title='Distribution relevant to paired t-test')
    axes[0].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(output_dir / 'score_distributions.png', dpi=170)
    if not show: plt.close(fig)
    return result
