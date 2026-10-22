"""One-call Drake/Meshcat inspection of a policy, with explicit sampling mode."""
from dataclasses import dataclass
import html
from pathlib import Path
import time

import numpy as np
import torch

from cartpole import CartPole, Config
from reinforce import Actor

_meshcat = None


def policy_action(policy, config, rng, *, stochastic=True):
    """Return action(obs, step); 'random' samples either force with probability 1/2."""
    if isinstance(policy, str) and policy == 'random':
        return lambda obs, step: int(rng.random() < .5)
    if isinstance(policy, Actor):
        probability = policy.sampler()
        def act(obs, step):
            p = probability(obs)
            return int(rng.random() < p) if stochastic else int(p > .5)
        return act
    if callable(policy): return lambda obs, step: int(policy(obs))
    raise TypeError("Pass 'random', an Actor, checkpoint path, or action(obs) callable")


def load_policy(policy, config=None):
    """Resolve a checkpoint or the 'random' policy and its environment settings."""
    name = 'policy'
    if isinstance(policy, str) and policy == 'random':
        return policy, config or Config(), 'random_policy'
    if isinstance(policy, (str, Path)):
        path = Path(policy)
        name = path.stem
        data = torch.load(path, map_location='cpu', weights_only=True)
        saved_config = Config(**data['env_config'])
        if config is not None and config != saved_config:
            raise ValueError('Playback config must match the trained environment')
        config = saved_config
        if 'actor' in data:
            policy = Actor(data['config']['hidden'])
            policy.load_state_dict(data['actor'])
        else:
            raise ValueError('Expected a REINFORCE actor checkpoint')
    return policy, config or Config(), name


def rollout_policy(policy, *, seed=200_000, stochastic=True, config=None, env=None):
    """Run a reproducible episode. Headless unless a visualized env is supplied.

    Action random numbers match reinforce.evaluate for the same seed. The final
    state and boundary reason are returned, so a cap is not mislabeled a fall.
    """
    policy, config, name = load_policy(policy, config)
    env = CartPole(config) if env is None else env
    rng = np.random.default_rng(seed + 800_000)
    act = policy_action(policy, config, rng, stochastic=stochastic)
    obs, _ = env.reset(seed=seed)
    env.render()
    states, actions = [obs], []
    while True:
        action = act(obs, env.steps)
        obs, _, term, trunc, info = env.step(action)
        states.append(obs)
        actions.append(action)
        if term or trunc: break
    return dict(states=np.asarray(states), actions=np.asarray(actions), length=len(actions),
                terminated=term, truncated=trunc, end_reason=info['end_reason'],
                seed=seed, stochastic=(True if name == 'random_policy' else
                                      stochastic if isinstance(policy, Actor) else False), name=name)


def draw_track(meshcat, config):
    """Draw a rail and red cart-center boundary markers, without adding physics."""
    from pydrake.geometry import Box, Rgba
    from pydrake.math import RigidTransform
    limit = config.x_limit
    meshcat.SetObject('track/rail', Box(2*limit, .18, .025), Rgba(.35, .4, .45, 1.))
    meshcat.SetTransform('track/rail', RigidTransform([0., 0., -.075]))
    for name, x in [('left', -limit), ('right', limit)]:
        meshcat.SetObject(f'track/{name}/end', Box(.025, .3, .06), Rgba(.85, .12, .1, 1.))
        meshcat.SetTransform(f'track/{name}/end', RigidTransform([x, 0., -.06]))
        for side, y in [('front', -.14), ('back', .14)]:
            path = f'track/{name}/{side}'
            meshcat.SetObject(path, Box(.025, .025, .22), Rgba(.85, .12, .1, 1.))
            meshcat.SetTransform(path, RigidTransform([x, y, .035]))
    # Frame both ends, even for a saved policy with a non-default track limit.
    half_width = max(limit + .35, 1.)
    meshcat.SetCameraPose([0., -.9*half_width, .2 + .2*half_width], [0., 0., .2])


@dataclass
class PolicyPlayback:
    path: Path
    rollout: dict
    env: CartPole  # keep the live Meshcat session reachable in a notebook


def watch_policy(policy, *, seed=200_000, stochastic=True, live=False,
                 output_dir='results', show=True, config=None):
    """Visualize a random policy, trained actor, or checkpoint in one notebook call.

    watch_policy('random', seed=123) chooses either force with probability 1/2
    at every step, regardless of the observation. It needs no trained checkpoint;
    the seed controls the initial state and actions. stochastic=False still
    samples this explicitly random policy.
    watch_policy(actor) samples the trained policy in headless Drake, then records
    those exact states in Meshcat. Rendering cannot change the physics trajectory.
    displays an embedded animation, and saves a standalone HTML file.
    watch_policy('results/reinforce_baseline_seed100.pt', seed=123,
                 stochastic=False) explicitly inspects the greedy policy.
    live=True additionally paces the recorded states to real time in Meshcat.
    For Colab the saved srcdoc iframe needs no inbound connection to its VM.
    Red track ends mark the cart-center limits; crossing one ends the episode.
    These markers are visual guides, not physical collision stops.
    """
    global _meshcat
    from pydrake.geometry import StartMeshcat
    policy, config, name = load_policy(policy, config)
    # SceneGraph/contact and publish events can change a diagram's dynamics or
    # integration schedule. Always generate the trajectory with the SAME
    # headless system used for training, then animate its states without a
    # second physical simulation.
    result = rollout_policy(policy, seed=seed, stochastic=stochastic, config=config)
    result['name'] = name
    if _meshcat is None: _meshcat = StartMeshcat()
    _meshcat.Delete()
    env = CartPole(config, render_mode='human', meshcat=_meshcat)
    draw_track(_meshcat, config)
    env.visualizer.StartRecording()
    start = time.perf_counter()
    for step, state in enumerate(result['states']):
        if live:
            time.sleep(max(0., start + step*config.dt - time.perf_counter()))
        physical_state = state.astype(float).copy()
        physical_state[1] += np.pi
        env.context.SetTime(step*config.dt)
        env.plant.SetPositionsAndVelocities(env.plant_context, physical_state)
        env.render()
    env.visualizer.StopRecording()
    env.visualizer.PublishRecording()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = 'random' if name == 'random_policy' else 'sampled' if result['stochastic'] else 'greedy'
    path = output_dir / f'{name}_seed{seed}_{mode}.html'
    contents = env.meshcat.StaticHtml()
    # A tall, narrow perspective viewport crops the track horizontally. Keep
    # both the notebook and standalone replay at a wide aspect ratio.
    frame = (f'<iframe title="Cart-pole replay" srcdoc="{html.escape(contents, quote=True)}" '
             'style="display:block;width:100%;height:auto;aspect-ratio:16/9;'
             'max-height:520px;box-sizing:border-box;border:1px solid #ccc"></iframe>')
    caption = (f'<p>Red track ends mark the cart-center limits at &plusmn;{config.x_limit:g} m. '
               f'{result["length"]}/{config.horizon} steps; {result["end_reason"]}; '
               f'{mode}; seed {seed}.</p>')
    path.write_text('<!doctype html><html><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width,initial-scale=1">'
                    '<title>Cart-pole replay</title></head>'
                    '<body style="margin:8px;font-family:sans-serif">' + frame + caption + '</body></html>')
    print(f"{result['length']}/{config.horizon} steps; {result['end_reason']}; "
          f"{mode}; seed {seed}. Saved {path.resolve()}\n"
          f"Red track ends: cart-center limits at x = +/-{config.x_limit:g} m.")
    if show:
        from IPython.display import HTML, display
        display(HTML(frame + caption))
    return PolicyPlayback(path=path, rollout=result, env=env)
