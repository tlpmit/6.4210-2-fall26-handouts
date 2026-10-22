"""Supplied BC and scalar-residual PPO exercise. Default work uses packaged data.

Importing this module, fitting BC, and updating from the recorded batch require
only NumPy/PyTorch. Drake is imported only for an explicitly requested replay.
"""

from pathlib import Path
import json, time, hashlib
import numpy as np
import torch
from torch import nn
import push_math
from push_policy import HybridActorCritic, distance_from_raw
from run_cache import run_cached

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/cautious_push"
RESULTS = HERE / "results/cautious_push"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_bc(path=None):
    cp = torch.load(path or DATA / "bc.pt", weights_only=True, map_location="cpu")
    if (
        cp["plan"]["distance_bounds"] != [0.0, 0.125]
        or cp["plan"]["observation_version"] != "resting_configuration_v1"
    ):
        raise ValueError("Incompatible BC policy")
    model = HybridActorCritic(13)
    model.load_state_dict(cp["model"])
    return model


class ScalarResidualPolicy(nn.Module):
    """Only theta and the critic learn; the BC action heads remain frozen."""

    def __init__(self, bc_path=None, seed=150):
        super().__init__()
        self.base = load_bc(bc_path)
        self.base.requires_grad_(False)
        self.theta = nn.Parameter(torch.zeros(()))
        with torch.random.fork_rng():
            torch.manual_seed(seed + 100_000)
            self.critic = HybridActorCritic(13).critic

    def policy_parameters(self, obs):
        with torch.no_grad():
            logits, mu = self.base.policy_parameters(obs)
        gate = (
            ((obs[..., :2].abs() * 0.3 - 0.07) / 0.05)
            .clamp(0.0, 1.0)
            .repeat_interleave(2, dim=-1)
        )
        return logits, mu + self.theta * gate, 0.2 + 0.3 * gate

    def evaluate(self, obs, faces, raw):
        logits, mu, std = self.policy_parameters(obs)
        selected = lambda x: x.gather(-1, faces[..., None]).squeeze(-1)
        categorical = torch.distributions.Categorical(logits=logits)
        normal = torch.distributions.Normal(selected(mu), selected(std))
        return push_math.joint_log_prob(categorical, normal, faces, raw), self.critic(
            obs
        ).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, rng, greedy=False):
        x = torch.as_tensor(obs, dtype=torch.float32)[None]
        logits, mu, std = self.policy_parameters(x)
        probabilities = logits[0].softmax(-1).numpy().astype(float)
        face = (
            int(logits[0].argmax())
            if greedy
            else int(rng.choice(4, p=probabilities / probabilities.sum()))
        )
        z = float(mu[0, face]) + (
            0.0 if greedy else float(std[0, face]) * float(rng.normal())
        )
        lp, v = self.evaluate(x, torch.tensor([face]), torch.tensor([z]))
        return face, z, float(lp[0]), float(v[0])


def load_residual(path):
    cp = torch.load(path, weights_only=True, map_location="cpu")
    if cp.get("format") != "scalar_residual_v1":
        raise ValueError("Use a packaged scalar-residual checkpoint")
    model = ScalarResidualPolicy(seed=cp["config"]["seed"])
    model.load_state_dict(cp["model"])
    return model, cp


def _imitation(model, arrays):
    obs = torch.from_numpy(arrays["observations"])
    faces = torch.from_numpy(arrays["faces"])
    logits, mu = model.policy_parameters(obs)
    pred = distance_from_raw(mu.gather(-1, faces[:, None]).squeeze(-1))
    return push_math.bc_loss(logits, pred, faces, torch.from_numpy(arrays["distances"]))


def _metrics(model, arrays):
    with torch.no_grad():
        obs = torch.from_numpy(arrays["observations"])
        faces = torch.from_numpy(arrays["faces"])
        logits, mu = model.policy_parameters(obs)
        distances = distance_from_raw(mu.gather(-1, faces[:, None]).squeeze(-1))
        return dict(
            loss=float(_imitation(model, arrays)),
            face_accuracy=float((logits.argmax(-1) == faces).float().mean()),
            distance_mae_cm=float(
                100 * (distances - torch.from_numpy(arrays["distances"])).abs().mean()
            ),
        )


def clone_cautious(*, output=None, retrain=False):
    """One quick 400-epoch supervised fit, with episode-held-out validation."""
    out = Path(output or RESULTS / "cloning")

    def fit():
        torch.set_num_threads(1)
        torch.manual_seed(130)
        train = dict(np.load(DATA / "train.npz", allow_pickle=False))
        val = dict(np.load(DATA / "validation.npz", allow_pickle=False))
        if set(train["episode_ids"]) & set(val["episode_ids"]):
            raise ValueError("Episode splits overlap")
        model = HybridActorCritic(13)
        with torch.no_grad():
            model.log_std.fill_(np.log(0.15))
        parameters = [
            *model.actor.parameters(),
            *model.face.parameters(),
            *model.mean.parameters(),
        ]
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        history = []
        best = float("inf")
        start = time.perf_counter()
        for epoch in range(401):
            if epoch % 20 == 0:
                metrics = _metrics(model, val)
                history.append(dict(epoch=epoch, **metrics))
                if metrics["loss"] < best:
                    best = metrics["loss"]
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            obs_dim=13,
                            plan=dict(
                                observation_version="resting_configuration_v1",
                                distance_bounds=[0.0, 0.125],
                                horizon=32,
                            ),
                        ),
                        out / "bc.pt",
                    )
            if epoch == 400:
                break
            for indices in torch.randperm(len(train["faces"])).split(128):
                batch = {
                    k: v[indices.numpy()]
                    for k, v in train.items()
                    if k in ["observations", "faces", "distances"]
                }
                loss = _imitation(model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        best_model = load_bc(out / "bc.pt")
        result = dict(
            validation=_metrics(best_model, val),
            history=history,
            fit_seconds=time.perf_counter() - start,
            training_transitions=len(train["faces"]),
            validation_transitions=len(val["faces"]),
        )
        save(out / "fit.json", result)
        return result

    return run_cached(
        out,
        name="cautious_bc",
        settings=dict(seed=130, epochs=400),
        inputs=[
            HERE / n for n in ["push_math.py", "push_policy.py", "residual_push.py"]
        ]
        + [DATA / n for n in ["train.npz", "validation.npz"]],
        outputs=["bc.pt", "fit.json"],
        run=fit,
        retrain=retrain,
    )


def update_batch():
    """Recorded actions were sampled by update_start.pt, not by the teacher."""
    meta = read(DATA / "update_batch.json")
    if (
        hashlib.sha256((DATA / "update_start.pt").read_bytes()).hexdigest()
        != meta["checkpoint_sha256"]
    ):
        raise ValueError("Batch/checkpoint mismatch")
    arrays = dict(np.load(DATA / "update_batch.npz", allow_pickle=False))
    if (
        hashlib.sha256((DATA / "update_batch.npz").read_bytes()).hexdigest()
        != meta["batch_sha256"]
    ):
        raise ValueError("Batch checksum mismatch")
    return arrays, meta


def batch_targets(arrays):
    """Episode-local GAE, MC critic targets, then ONE batch standardization."""
    advantage = np.empty(len(arrays["rewards"]))
    returns = np.empty_like(advantage)
    for episode in np.unique(arrays["episode_ids"]):
        ix = np.flatnonzero(arrays["episode_ids"] == episode)
        if not np.all(np.diff(ix) == 1):
            raise ValueError("An episode is not contiguous")
        if not (arrays["terminated"][ix[-1]] or arrays["cutoff"][ix[-1]]):
            raise ValueError("Incomplete recorded episode")
        r, v, n = (
            arrays["rewards"][ix],
            arrays["old_values"][ix],
            arrays["next_values"][ix],
        )
        advantage[ix] = push_math.gae_advantages(r, v, n, 0.95)
        returns[ix] = np.cumsum(r[::-1])[::-1] + n[-1]
    return returns, (advantage - advantage.mean()) / (advantage.std() + 1e-8)


def update_from_batch(*, output=None, retrain=False):
    """One PPO iteration, always starting from the recorded collection policy.

    Four optimization epochs reuse ONE on-policy batch. Calling again with
    retrain=True starts from the same checkpoint; it is not another online
    training iteration. Full training histories come from actual fresh rollouts.
    """
    out = Path(output or RESULTS / "short_update")

    def run():
        torch.set_num_threads(1)
        torch.manual_seed(900)
        arrays, meta = update_batch()
        model, cp = load_residual(DATA / "update_start.pt")
        returns, advantages = batch_targets(arrays)
        obs = torch.tensor(arrays["observations"])
        faces = torch.tensor(arrays["faces"])
        raw = torch.tensor(arrays["raw"], dtype=torch.float32)
        old = torch.tensor(arrays["old_log_probs"], dtype=torch.float32)
        target = torch.tensor(returns, dtype=torch.float32)
        adv = torch.tensor(advantages, dtype=torch.float32)
        with torch.no_grad():
            initial_log, _ = model.evaluate(obs, faces, raw)
        if not torch.allclose(initial_log, old, atol=2e-5, rtol=1e-5):
            raise ValueError("Recorded actions must match their OLD checkpoint")
        optimizer = torch.optim.Adam(
            [
                dict(params=[model.theta], lr=0.01),
                dict(params=list(model.critic.parameters()), lr=0.0003),
            ]
        )
        start_theta = float(model.theta.detach())
        steps = 0
        stopped = False
        start = time.perf_counter()
        for _ in range(4):
            for ix in torch.randperm(len(raw)).split(64):
                lp, value = model.evaluate(obs[ix], faces[ix], raw[ix])
                lr = lp - old[ix]
                if float((lr.exp() - 1 - lr).detach().mean()) > 0.02:
                    stopped = True
                    break
                loss = push_math.ppo_actor_loss(
                    lp, old[ix], adv[ix], 0.1
                ) + 0.25 * push_math.critic_loss(value, target[ix])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([model.theta, *model.critic.parameters()], 0.5)
                optimizer.step()
                steps += 1
            if stopped:
                break
        assert all(p.grad is None for p in model.base.parameters())
        with torch.no_grad():
            lp, _ = model.evaluate(obs, faces, raw)
            lr = lp - old
        result = dict(
            initial_theta=start_theta,
            updated_theta=float(model.theta.detach()),
            transitions=len(raw),
            episodes=len(np.unique(arrays["episode_ids"])),
            optimizer_steps=steps,
            kl_stopped=stopped,
            approx_kl=float((lr.exp() - 1 - lr).mean()),
            update_seconds=time.perf_counter() - start,
            interpretation="One update on a fixed recorded on-policy batch; no claim of improved task performance from this one update.",
        )
        torch.save(dict(cp, model=model.state_dict()), out / "update.pt")
        save(out / "update.json", result)
        return result

    return run_cached(
        out,
        name="recorded_ppo_update",
        settings=dict(epochs=4, clip=0.1, gae_lambda=0.95),
        inputs=[
            HERE / n for n in ["push_math.py", "push_policy.py", "residual_push.py"]
        ]
        + [
            DATA / n
            for n in ["update_start.pt", "update_batch.json", "update_batch.npz"]
        ],
        outputs=["update.json", "update.pt"],
        run=run,
        retrain=retrain,
    )


def make_env(*, record=False):
    from push_env import LearningPush
    from push_physics import PushPhysics

    p = read(DATA / "plan.json")
    return LearningPush(
        horizon=32, timestep=0.001, physics=PushPhysics(**p["physics"]), record=record
    )


def cautious_action(obs):
    error = np.asarray(obs)[:2] * 0.3
    axis = int(np.argmax(abs(error)))
    face = axis * 2 + int(error[axis] < 0)
    calibration = read(DATA / "calibration.json")
    distance = float(
        np.clip(
            (min(abs(error[axis]), 0.04) - calibration["beta"]) / calibration["alpha"],
            0.0,
            0.125,
        )
    )
    return face, distance


def replay(which="final", *, seed=9460000, output=None, show=True):
    """Fresh greedy Drake episode, with metrics and a saved Meshcat recording.

    'best' uses replay_policy.json's monitoring-selected checkpoint. It is an
    illustration, not a replacement for the all-seed fixed-final comparison.
    Replays need no completed student mathematical functions.
    """
    from train_hybrid_push import hybrid_step
    from watch_push import watch_multi_push

    env = make_env(record=True)
    obs, _ = env.reset(seed=seed)
    if which == "bc":
        model = ScalarResidualPolicy()
    elif which == "teacher":
        model = None
    elif which in ("midpoint", "final"):
        model, _ = load_residual(DATA / "checkpoints" / f"seed150_{which}.pt")
    elif which == "best":
        selection = read(DATA / "replay_policy.json")
        model, _ = load_residual(DATA / selection["checkpoint"])
        print("Monitoring-selected replay policy:", selection)
    else:
        raise ValueError("Choose teacher, bc, midpoint, final or best")
    total = 0.0
    invalid = 0
    initial_observation = obs.copy()
    while not env.task.done:
        if model is None:
            face, length = cautious_action(obs)
            raw = float(torch.logit(torch.tensor(length / 0.125).clamp(1e-7, 1 - 1e-7)))
        else:
            with torch.no_grad():
                logits, mu, _ = model.policy_parameters(torch.as_tensor(obs)[None])
                face = int(logits[0].argmax())
                raw = float(mu[0, face])
        obs, r, _, info = hybrid_step(
            env, face, raw, free_time_cost=0.0, total_time_cost=0.02, push_penalty=0.0
        )
        total += r
        invalid += int(not info["valid"])
    summary = dict(
        policy=which,
        goal_seed=int(seed),
        success=info["success"],
        pushes=info["macros"],
        reward=total,
        execution_seconds=float(env.task.context.get_time()),
        distance=info["distance"],
        invalid=invalid,
        theta=float(model.theta.detach()) if model is not None else None,
    )
    print(summary)
    result = watch_multi_push(
        env.task,
        output_dir=Path(output or RESULTS / "replays") / f"{which}_{seed}",
        show=show,
    )
    result.update(rollout=summary, initial_observation=initial_observation)
    save(Path(result["path"]).with_suffix(".json"), summary)
    return result


def replay_pair(*, seed=9460000, bc_replay=None, output=None, show=True):
    """Compare BC and the monitoring-selected learned policy from identical starts."""
    import pandas as pd

    bc = (
        bc_replay
        if bc_replay is not None
        else replay("bc", seed=seed, output=output, show=False)
    )
    if bc["rollout"]["policy"] != "bc" or bc["rollout"]["goal_seed"] != seed:
        raise ValueError("Use the initial BC replay from this same goal")
    learned = replay("best", seed=seed, output=output, show=False)
    if not np.array_equal(bc["initial_observation"], learned["initial_observation"]):
        raise ValueError("Replay initial states must match exactly")
    table = pd.DataFrame([bc["rollout"], learned["rollout"]])
    if show:
        import html
        from IPython.display import HTML, display

        display(
            table[
                [
                    "policy",
                    "goal_seed",
                    "success",
                    "execution_seconds",
                    "pushes",
                    "reward",
                    "distance",
                ]
            ]
        )
        panels = []
        for title, result in [
            ("Initial BC", bc),
            ("Best monitoring-selected residual PPO", learned),
        ]:
            contents = html.escape(Path(result["path"]).read_text(), quote=True)
            panels.append(
                f'<section style="flex:1;min-width:340px"><h4>{title}</h4>'
                f'<iframe srcdoc="{contents}" title="{title}" width="100%" height="480"></iframe></section>'
            )
        display(
            HTML(
                '<div style="display:flex;flex-wrap:wrap;gap:12px">'
                + "".join(panels)
                + "</div>"
            )
        )
    return dict(bc=bc, learned=learned, scores=table)
