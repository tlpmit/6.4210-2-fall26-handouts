"""Reward accounting shared by PPO training and saved-trajectory rescoring.

The simulator's legacy return includes .02 per VALID push, plus progress,
success, invalid-command and workspace penalties. Keep that return available
for historical comparisons; explicitly replace its action/time costs here.
"""
import numpy as np


def reward_from_legacy(legacy, valid_pushes, free_seconds, total_seconds, *,
                       free_time_cost=0., total_time_cost=0., push_penalty=.02):
    costs=np.array([free_time_cost,total_time_cost,push_penalty],dtype=float)
    if not np.isfinite(costs).all() or (costs<0).any():
        raise ValueError('Reward costs must be finite and nonnegative')
    if free_time_cost and total_time_cost:
        raise ValueError('Choose free-space OR total time cost, not both')
    return (legacy+(.02-push_penalty)*valid_pushes
            -free_time_cost*free_seconds-total_time_cost*total_seconds)


def objective(plan):
    return dict(free_time_cost=plan.get('free_time_cost',.02),
                total_time_cost=plan.get('total_time_cost',0.),
                push_penalty=plan.get('push_penalty',.02))


def normalize(rows, seeds, *, free_time_cost=.02, total_time_cost=0.,
              push_penalty=.02, check_saved=True):
    if [r['seed'] for r in rows]!=seeds:raise ValueError('Mismatched evaluation goals')
    result=[]
    for r in rows:
        free=r['free_motion_seconds'] if 'free_motion_seconds' in r else r['phase_seconds']['free']
        legacy=r.get('unpenalized_return',r['return_'])
        valid=r['macros']-r['invalid']
        if not 0<=valid<=r['macros']:raise ValueError('Invalid action accounting')
        value=reward_from_legacy(legacy,valid,free,r['execution_seconds'],
            free_time_cost=free_time_cost,total_time_cost=total_time_cost,push_penalty=push_penalty)
        if check_saved and 'unpenalized_return' in r and not np.isclose(value,r['return_']):
            raise ValueError('Saved reward disagrees with objective')
        result.append(dict(seed=r['seed'],success=float(r['success']),reward=value,
            original_reward=legacy,pushes=r['macros'],distance=r['distance'],invalid=r['invalid'],
            free_seconds=free,execution_seconds=r['execution_seconds'],
            travel_m=r.get('finger_travel_m',r.get('travel_m'))))
    if not np.isfinite([[v for k,v in r.items() if k!='seed'] for r in result]).all():
        raise ValueError('Nonfinite evaluation')
    return result
