"""Small, explicit HW6 runs. Importing this module never starts an experiment."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from run_cache import run_cached

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
RESULTS = HERE / 'results'
PUSH_RESULTS = RESULTS / 'zero_minimum_push'
PUSH_REFERENCE = DATA / 'zero_minimum_push_reference'
BC_CHECKPOINT = DATA / 'zero_minimum_push/bc_seed130.pt'
LOCAL_PUSH_SEED = 150
LOCAL_GOALS = list(range(122000, 122008))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def code_inputs():
    """Run dependencies, excluding separate plotting/replay/instructor helpers."""
    display_only = {'learning_curves.py', 'visualization.py', 'watch_push.py',
                    'solution_support.py'}
    return [p for p in HERE.glob('*.py') if p.name not in display_only]


def _read_older_cartpole(path, checkpoint, *, seed, config, environment, monitor, endpoint):
    """Read a pre-cache result after checking settings and its saved actor.

    These runs have no source fingerprint. Report that limitation explicitly;
    never stamp today's source hashes onto a historical training run.
    """
    import torch
    from reinforce import Actor
    try:
        result = read(path)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected = dict(seed=seed, config=asdict(config), environment=asdict(environment),
                        monitoring_seeds=monitor, final_evaluation_seeds=endpoint)
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError('saved run settings do not match this call')
        if (saved['seed'] != seed or saved['config'] != expected['config'] or
                saved['env_config'] != expected['environment']):
            raise ValueError('checkpoint settings do not match the result')
        history = result['learning']
        updates = list(range(0, config.updates + 1, config.eval_every))
        if updates[-1] != config.updates:
            updates.append(config.updates)
        if [row['update'] for row in history] != updates:
            raise ValueError('training history is incomplete')
        if history[-1]['steps'] < config.updates * config.batch_episodes:
            raise ValueError('training transition count is incomplete')
        scores = np.asarray(result['final']['stochastic'], dtype=float)
        if (scores.shape != (len(endpoint),) or not np.isfinite(scores).all() or
                np.any((scores < 1) | (scores > environment.horizon) | (scores != np.floor(scores)))):
            raise ValueError('final evaluation is incomplete or invalid')
        if not all(torch.isfinite(value).all() for value in saved['actor'].values()):
            raise ValueError('checkpoint contains nonfinite weights')
        # Validate layer shapes without changing the notebook's random state.
        with torch.random.fork_rng(devices=[]):
            Actor(config.hidden).load_state_dict(saved['actor'])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise RuntimeError(
            f'Cannot reuse older result {path}: {error}. '
            'Use retrain=True to archive it and run again, or choose another output folder.'
        ) from error
    print(f'Using older saved run: {path}\n'
          'Settings and checkpoint checked; original training-code version was not recorded. '
          'No training performed. Use retrain=True if you want a new run with the current code.', flush=True)
    return result


def run_cartpole(baseline, *, seed=100, output=None, retrain=False):
    """Train/reuse one policy; pre-cache results load with a provenance notice."""
    from reinforce import TrainConfig, train, evaluate
    from cartpole import Config
    output = Path(output or RESULTS / 'cartpole')
    tag = f'reinforce_{"baseline" if baseline else "plain"}_seed{seed}'
    path = output / f'{tag}.json'
    cfg = TrainConfig(updates=60, baseline=baseline)
    monitor = list(range(1_000_000 + seed*1000, 1_000_000 + seed*1000 + 20))
    endpoint = list(range(2_100_000, 2_100_020))
    if not retrain and path.exists() and not (output / f'.{tag}.cache.json').exists():
        return _read_older_cartpole(path, output / f'{tag}.pt', seed=seed, config=cfg,
                                   environment=Config(), monitor=monitor, endpoint=endpoint)
    def run():
        actor, learning = train(seed, cfg, output=output / f'{tag}.pt', eval_seeds=monitor)
        start = time.perf_counter()
        scores = evaluate(actor, Config(), endpoint, stochastic=True)
        result = dict(seed=seed, config=asdict(cfg), environment=asdict(Config()),
            learning=learning, monitoring_seeds=monitor, final_evaluation_seeds=endpoint,
            final=dict(stochastic=scores), final_evaluation_seconds=time.perf_counter()-start)
        save(path, result)
        return result
    return run_cached(output, name=tag, settings=dict(seed=seed, config=asdict(cfg),
        environment=asdict(Config()), monitoring=monitor, endpoint=endpoint),
        inputs=code_inputs(), outputs=[f'{tag}.*'], run=run, retrain=retrain)


def clone_push(*, output=None, retrain=False):
    """Current Q12: quick cautious-teacher BC fit, requiring no simulator."""
    from residual_push import clone_cautious
    return clone_cautious(output=output, retrain=retrain)


def run_residual_update(*, output=None, retrain=False):
    """Current Q12: one PPO iteration from the recorded on-policy checkpoint."""
    from residual_push import update_from_batch
    return update_from_batch(output=output, retrain=retrain)


def run_ppo(method, *, seed=LOCAL_PUSH_SEED, budget=1024, output=None, retrain=False):
    """Historical full-policy experiment; current Q12 uses run_residual_update.

    Kept for old notebooks/results. This is optional simulator training and is
    not the scalar residual method or the required short coding exercise.
    """
    from train_hybrid_push import train
    from push_physics import PushPhysics
    if method not in ('ppo_raw', 'ppo_finetune'):
        raise ValueError('Choose ppo_raw or ppo_finetune')
    plan = read(DATA / 'push/plan.json')
    out = Path(output or PUSH_RESULTS / method)
    inputs = code_inputs() + [DATA / 'push/plan.json']
    if method == 'ppo_finetune':
        inputs.append(BC_CHECKPOINT)
    return run_cached(out, name=f'hybrid_seed{seed}',
        settings=dict(method=method, seed=seed, budget=budget, goals=LOCAL_GOALS),
        inputs=inputs, outputs=[f'hybrid_seed{seed}.*', f'hybrid_seed{seed}_*.pt'],
        retrain=retrain, run=lambda: train(seed=seed, budget=budget, curriculum=False, out=out,
        monitoring_start=60000, monitoring_count=4, test_start=LOCAL_GOALS[0], test_count=8,
        algorithm='ppo', horizon=16,
        initial_checkpoint=BC_CHECKPOINT if method=='ppo_finetune' else None,
        fresh_critic=True, training_seed_start=70000 + (seed-150)*10000,
        save_checkpoints=True, evaluate_best=True, timestep=.001,
        physics=PushPhysics(**plan['physics']), total_time_cost=.02, push_penalty=0.,
        free_time_cost=0., **plan['ppo']))


def push_replay(checkpoint, *, seed=122000, sampled=False):
    from train_hybrid_push import load, evaluate
    from watch_push import watch_multi_push
    rows, env = evaluate(load(checkpoint), [seed], sampled=sampled, record=True)
    print(rows[0])
    return watch_multi_push(env.task, output_dir=RESULTS / 'replays')


def push_scores(variant='selected', *, local=False):
    """Supplied or local scores: method -> runs -> episode dictionaries.

    By default use the supplied three-seed study on 100 shared goals, with
    the current zero-minimum range and unweighted BC initializer.
    local=True uses new zero-minimum runs and their eight-goal baselines.
    Rewards and success describe observed prefixes of at most 16 commands;
    no critic continuation estimate is added to evaluation scores.
    """
    from push_reward import normalize
    if variant not in ('selected','final','sampled'):raise ValueError('Unknown policy variant')
    groups={}
    if local:
        for name in ['analytic','bc']:
            groups[name]=[read(DATA / 'zero_minimum_push' / f'{name}.json')['rows']]
    if not local:
        for name in ['analytic','bc']:
            groups[name]=[read(PUSH_REFERENCE / f'{name}.json')['rows']]
    for method in ['ppo_raw','ppo_finetune']:
        runs=[]
        paths=[PUSH_RESULTS / method / f'hybrid_seed{LOCAL_PUSH_SEED}.json'] if local else sorted((PUSH_REFERENCE / method).glob('seed*.json'))
        for p in paths:
            if not p.exists():continue
            d=read(p)
            rows=d['best_monitoring']['test'] if variant=='selected' else d['test' if variant=='final' else 'sampled_test']
            runs.append(normalize(rows,[r['seed'] for r in rows],free_time_cost=0.,total_time_cost=.02,push_penalty=0.))
        if runs:groups[method]=runs
    return groups


def score_table(groups):
    import pandas as pd
    rows=[]
    for method,runs in groups.items():
        for index,run in enumerate(runs):
            rows.append(dict(method=method,training_seed=run[0].get('training_seed',150+index if 'ppo' in method else None),**{key:np.mean([r[key] for r in run])
                for key in ['success','reward','execution_seconds','pushes','free_seconds','travel_m','distance'] if key in run[0]}))
    return pd.DataFrame(rows)


def paired_goal_interval(groups, first, second, *, resamples=5000, seed=701):
    """First minus second. Goal bootstrap conditional on these fixed policies."""
    ids=[r['seed'] for r in groups[first][0]]
    def matrix(name):
        for run in groups[name]:
            if [r['seed'] for r in run]!=ids:raise ValueError('Goal IDs must match, in order')
        return np.array([[r['reward'] for r in run] for run in groups[name]])
    delta=matrix(first).mean(0)-matrix(second).mean(0)
    rng=np.random.default_rng(seed)
    boots=delta[rng.integers(len(ids),size=(resamples,len(ids)))].mean(1)
    return dict(difference=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
        goals=len(ids),interpretation='Conditional on these fitted policies; not training-seed uncertainty')


def reward_violins(groups):
    """Equal area PDFs, fixed smoothing bandwidth, failures retained; seed means are dots."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    densities=[]
    for runs in groups.values():
        values=np.array([r['reward'] for run in runs for r in run])
        grid=np.linspace(values.min()-.6,values.max()+.6,600)
        density=(gaussian_kde(values,bw_method=.15/values.std(ddof=1))(grid)
                 if len(values)>1 and values.std()>1e-9 else np.exp(-.5*((grid-values[0])/.15)**2))
        density/=np.trapezoid(density,grid)
        densities.append((grid,density))
    peak=max(d.max() for _,d in densities)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for i,((name,runs),(grid,density)) in enumerate(zip(groups.items(),densities)):
        width=.38*density/peak
        ax.fill_betweenx(grid,i-width,i+width,alpha=.45)
        ax.scatter(np.full(len(runs),i),[np.mean([r['reward'] for r in run]) for run in runs],c='black',s=22)
    ax.set_xticks(range(len(groups)),groups.keys(),rotation=15)
    ax.set_ylabel('Total-time reward (all outcomes)')
    ax.set_title('Equal-area violins; dots = fitted-policy means, not independent goals')
    return fig


def push_learning(*, local=False):
    """Plot local curves or the supplied zero-minimum three-seed study."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method,color in [('ppo_raw','#2463a6'),('ppo_finetune','#c65b19')]:
        paths=[PUSH_RESULTS / method / f'hybrid_seed{LOCAL_PUSH_SEED}.json'] if local else sorted((PUSH_REFERENCE/method).glob('seed*.json'))
        runs=[read(p)['learning'] for p in paths if p.exists()]
        if not runs:continue
        for ax,key in zip(axes,['mean_return','success']):
            for run in runs:
                ax.plot([r['steps'] for r in run],[r[key] for r in run],alpha=.35,color=color)
            # Interpolate within common range, fixed cohort throughout.
            grid=np.linspace(0,min(r[-1]['steps'] for r in runs),100)
            y=np.array([np.interp(grid,[v['steps'] for v in r],[v[key] for v in r]) for r in runs])
            line,=ax.plot(grid,y.mean(0),label=method,color=color)
            if len(runs)>1:
                se=y.std(0,ddof=1)/np.sqrt(len(runs))
                ax.fill_between(grid,y.mean(0)-se,y.mean(0)+se,color=line.get_color(),alpha=.15)
            ax.set(xlabel='Online pushes',ylabel=key)
    # Baselines on the SAME monitoring goals; no test-score baseline on a monitoring curve.
    baseline_paths=([PUSH_RESULTS/'ppo_finetune'/f'hybrid_seed{LOCAL_PUSH_SEED}.json'] if local else
                    sorted((PUSH_REFERENCE/'ppo_finetune').glob('seed*.json')))
    bc_initial=[read(p)['learning'][0] for p in baseline_paths if p.exists()]
    if bc_initial:
        # Temperature does not affect greedy initial BC behavior.
        for ax,key in zip(axes,['mean_return','success']):
            ax.axhline(np.mean([r[key] for r in bc_initial]),color='gray',ls='--',label='initial BC (monitoring)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Monitoring only; shading = one training-seed SE when multiple fits exist')
    return fig
