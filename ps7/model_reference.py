"""Data and reference scores for PS7's learned-model problem.

The transitions come from PS6's cautious-push demonstrations.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/cautious_model_based'
DEMONSTRATIONS = HERE.parent / 'ps6' / 'data/cautious_push'


def read(path):
    return json.loads(Path(path).read_text())


def states_from_observations(obs):
    """Decode PS6's 13-channel observations of a settled configuration.

    Channels 4:6 are finger xy, 6:10 the block quaternion (wxyz) and 10:13
    block xyz. Channels 0:4 are relative to the goal and are dropped, since
    the goal does not affect the physics. Returns an (N, 8) array in
    model_based_push.STATE_NAMES order. The quaternion conversion uses scipy,
    so this runs without Drake.
    """
    obs = np.asarray(obs, dtype=float)
    angles = Rotation.from_quat(obs[:, [7, 8, 9, 6]]).as_euler('xyz')
    return np.c_[obs[:, 10:12], angles[:, 2], obs[:, 4:6],
                 obs[:, 12], angles[:, :2]]


def model_data():
    """Return (transitions, splits) from PS6's cautious-push demonstrations.

    splits['train'] holds the 64 training episodes (886 transitions). PS6's
    16 validation episodes are divided between two splits, each episode
    staying whole: the first eight by episode id go to splits['validation'],
    which chooses the network checkpoint, and the last eight to
    splits['diagnostic'], used only to measure prediction error. transitions
    lists every record again, labeled 'train' or 'validation', for the affine
    fit and its check on all 221 validation transitions. Each record has
    state and next_state (8 entries, STATE_NAMES order), face, length (m),
    seconds, valid, reward, terminated, truncated, episode and split.
    """
    splits = {k: [] for k in ['train', 'validation', 'diagnostic']}
    for name in ['train', 'validation']:
        a = np.load(DEMONSTRATIONS / f'{name}.npz', allow_pickle=False)
        state = states_from_observations(a['observations'])
        nxt = states_from_observations(a['next_observations'])
        ids = sorted(set(map(int, a['episode_ids'])))
        assert len(ids) == (64 if name == 'train' else 16)
        for i, episode in enumerate(a['episode_ids']):
            split = 'train' if name == 'train' else (
                'validation' if int(episode) in ids[:8] else 'diagnostic')
            splits[split].append(dict(state=state[i].tolist(),
                next_state=nxt[i].tolist(), face=int(a['faces'][i]),
                length=float(a['distances'][i]), seconds=float(a['seconds'][i]),
                valid=bool(a['valid'][i]), reward=float(a['rewards'][i]),
                terminated=bool(a['terminated'][i]), truncated=bool(a['truncated'][i]),
                episode=int(episode), split=split))
    episodes = [set(r['episode'] for r in rows) for rows in splits.values()]
    assert all(not a & b for i, a in enumerate(episodes) for b in episodes[i+1:])
    transitions = [dict(r, split='train' if k == 'train' else 'validation')
                   for k, rows in splits.items() for r in rows]
    return transitions, splits


def reference_scores():
    """Supplied scores on PS6's 64 test goals, keyed by method.

    'bc' and 'residual_ppo' (the ten final PPO policies) come from PS6;
    'calibrated' and 'mppi' come from data/cautious_model_based/evaluation.
    """
    import sys
    shared = str(HERE.parent / 'ps6')
    if shared not in sys.path:
        sys.path.append(shared)
    from residual_analysis import reference
    ps6_reference = reference('confirmatory')
    groups = {'bc': [ps6_reference['baselines']['bc']], 'residual_ppo': [
        [dict(r, training_seed=run['config']['seed']) for r in run['evaluations']['final']]
        for run in ps6_reference['runs']]}
    for method in ['calibrated', 'mppi']:
        rows = read(DATA / 'evaluation' / f'{method}.json')['rows']
        if [r['seed'] for r in rows] != ps6_reference['plan']['test_goals']:
            raise ValueError("PS7's evaluation goals must match PS6's test goals, in order")
        groups[method] = [rows]
    return groups
