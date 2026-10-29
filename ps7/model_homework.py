"""Fitting the models and running the controllers for PS7's learned-model problem.

The simulator and the demonstration data come from the sibling ps6 folder.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PS6 = HERE.parent / 'ps6'
if not PS6.is_dir():
    raise FileNotFoundError('PS7 requires the sibling ps6 folder from the course checkout.')
if str(PS6) not in sys.path:
    sys.path.append(str(PS6))
from run_cache import run_cached
from homework import read, save, score_table, paired_goal_interval, reward_violins

DATA = HERE / 'data'
SHARED_DATA = PS6 / 'data'
RESULTS = HERE / 'results'
MODEL_RESULTS = RESULTS / 'cautious_model_based'


MODEL_CODE = ['model_based_math.py', 'model_based_push.py', 'model_homework.py', 'model_reference.py']


def code_inputs():
    """The files run_cached hashes to tell whether a saved result is stale.

    These are this problem's ps7 files and the ps6 Python files, so editing
    another problem's code (hybrid_force_control.py) keeps these results.
    """
    return [*(HERE / name for name in MODEL_CODE), *PS6.glob('*.py')]


def _cached(folder, *, name, retrain, **kwargs):
    """run_cached, except that a run which stopped with an error runs again.

    run_cached marks a run as running before it starts, and only marks it
    complete at the end, so a run that raised (for example on a function that
    is still a stub) leaves that mark behind. Such a run has no result worth
    keeping, so it is redone without asking for retrain=True.
    """
    receipt = Path(folder) / f'.{name}.cache.json'
    try:
        stopped = json.loads(receipt.read_text()).get('status') == 'running'
    except (OSError, ValueError):
        stopped = False
    return run_cached(folder, name=name, retrain=retrain or stopped, **kwargs)


def model_data():
    """Return (transitions, splits) from PS6's demonstrations; see model_reference.model_data."""
    from model_reference import model_data as shared_data
    return shared_data()


def fit_models(*, output=None, retrain=False):
    """Fit (or reuse) the affine calibration and the transition network.

    Returns (calibration dict, fitted Dynamics, fitting summary with seconds,
    selected_epoch, validation_loss and the loss history).
    """
    from model_based_push import calibrate, fit_dynamics, diagnostics, load_dynamics
    folder = Path(output or MODEL_RESULTS)
    def run():
        _, splits = model_data()
        calibration = calibrate(splits['train'])
        save(folder / 'calibration.json', calibration)
        model, fitting = fit_dynamics(splits['train'], splits['validation'], folder / 'dynamics.pt')
        save(folder / 'fit.json', dict(fitting=fitting, diagnostics=diagnostics(model, splits['diagnostic'])))
        return dict(calibration=calibration, fitting=fitting)
    result = _cached(folder, name='model_fit', settings={}, inputs=code_inputs() +
        [SHARED_DATA / 'cautious_push/train.npz', SHARED_DATA / 'cautious_push/validation.npz',
         DATA / 'cautious_model_based/plan.json'],
        outputs=['calibration.json', 'dynamics.pt', 'fit.json'], run=run, retrain=retrain)
    return result['calibration'], load_dynamics(folder / 'dynamics.pt'), result['fitting']


def evaluate_controller(method, *, folder=None, seeds=None, record=False, retrain=False):
    """Run (or reuse) the evaluation episodes and return (scores, environment).

    The default seeds are the first four PS6 test goals. The environment is
    None when the scores come from a saved run. `record=True` always runs the
    episodes again for a replay and returns that environment, leaving the
    saved scores unchanged. `retrain=True` archives this controller's saved
    scores and evaluates it again.
    """
    if method not in ('calibrated', 'mppi'):
        raise ValueError('Choose calibrated or mppi')
    folder = Path(folder or MODEL_RESULTS)
    seeds = list(range(9660000, 9660004)) if seeds is None else list(seeds)
    needed = ['calibration.json'] + (['dynamics.pt'] if method == 'mppi' else [])
    if not all((folder / name).exists() for name in needed):
        raise FileNotFoundError(f'No fitted models in {folder}. Run fit_models() first.')
    if record:
        return _evaluate_controller(method, folder=folder, seeds=seeds, record=True)
    inputs = code_inputs() + [DATA / 'cautious_model_based/plan.json', folder / 'calibration.json']
    if method == 'mppi':
        inputs.append(folder / 'dynamics.pt')
    environments = []
    def run():
        result, env = _evaluate_controller(method, folder=folder, seeds=seeds)
        environments.append(env)
        return result
    result = _cached(folder, name=method, settings=dict(method=method, seeds=seeds),
        inputs=inputs, outputs=[f'{method}_scores.json'], run=run, retrain=retrain)
    return result, environments[0] if environments else None


def _evaluate_controller(method, *, folder, seeds, record=False, progress_path=None):
    """Run the controller's episodes in Drake. MPPI's look-ahead uses the learned model."""
    from model_based_push import state_of, heuristic, MPPI, load_dynamics
    from push_env import FACES
    from residual_push import make_env
    from push_reward import reward_from_legacy
    folder = Path(folder or MODEL_RESULTS)
    seeds = list(range(9660000, 9660004)) if seeds is None else list(seeds)
    if method not in ('calibrated','mppi'):
        raise ValueError('Choose calibrated or mppi')
    path = folder / f'{method}_scores.json'
    plan = read(DATA / 'cautious_model_based/plan.json')
    calibration = read(folder / 'calibration.json')
    planner = MPPI(load_dynamics(folder / 'dynamics.pt'), calibration, **plan['mppi']) if method=='mppi' else None
    env = make_env(record=record)
    started = time.perf_counter(); rows=[]; transitions=[]
    for seed in seeds:
        initial_obs, _ = env.reset(seed=int(seed)); rng=np.random.default_rng(seed+9_000_000)
        total=0.;invalid=0;planning=0.
        while not env.task.done:
            state=state_of(env); begin=time.perf_counter()
            if planner is None:
                face,length=heuristic(state,env.task.goal,calibration)
            else:
                face,length,_=planner.act(state,env.task.goal,rng)
            planning+=time.perf_counter()-begin
            axis,sign=FACES[face]
            _,raw_reward,terminated,truncated,info=env.task.step(env.task.face_push(axis,sign,length))
            # PS6's reward_from_legacy turns the simulator's per-push reward
            # into this problem's total-time objective.
            total+=reward_from_legacy(raw_reward,int(info['valid']),0.,info['simulated_seconds'],
                                      total_time_cost=.02,push_penalty=0.)
            invalid+=int(not info['valid'])
            transitions.append(dict(seed=int(seed), episode=int(seed),
                step=env.task.count, state=state.tolist(), next_state=state_of(env).tolist(),
                face=int(face), length=float(length), seconds=float(info['simulated_seconds']),
                valid=bool(info['valid']), terminated=bool(terminated),
                truncated=bool(truncated)))
        rows.append(dict(seed=int(seed),success=bool(info['success']),reward=total,
            pushes=env.task.count,execution_seconds=float(env.task.context.get_time()),
            free_seconds=env.task.motion_seconds['free'],travel_m=sum(env.task.motion_length.values()),
            settling_seconds=env.task.motion_seconds['settle'],cutoff=not bool(info['success']) and info['reason']=='cutoff',
            invalid=invalid,planning_seconds=planning,
            distance=float(np.linalg.norm(env.task.box_pose()[:2]-env.task.goal)),
            initial_observation=initial_obs.tolist()))
        if progress_path:
            save(progress_path, dict(episodes=len(rows), last_goal=int(seed),
                commands=len(transitions), seconds=time.perf_counter()-started))
    result=dict(rows=rows,transitions=transitions,wall_seconds=time.perf_counter()-started)
    if not record:save(path,result)
    return result,env


def model_scores():
    """Supplied scores on PS6's 64 test goals: BC, the ten final PPO policies, the calibrated heuristic, and MPPI."""
    from model_reference import reference_scores
    return reference_scores()
