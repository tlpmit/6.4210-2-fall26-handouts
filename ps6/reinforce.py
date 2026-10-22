"""REINFORCE with an optional learned, action-independent baseline.

All trajectories are complete; the objective is undiscounted capped survival.
No advantage normalization, entropy bonus, or gradient clipping is hidden in
the comparison. The loss is an EPISODE average, with a fixed scale factor.

cartpole.ipynb runs the experiments and plots. The sections below separate
plumbing, supplied components, the learning loop, and the two exercises.
"""

######################################################################
## Code for nobody but Python to read
######################################################################

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from cartpole import CartPole, Config
from torch import nn


def tensors(episodes):
    """Concatenate episode records into state, action, return, and time tensors."""
    return (
        torch.tensor(
            np.concatenate([e["states"] for e in episodes]), dtype=torch.float32
        ),
        torch.tensor(
            np.concatenate([e["actions"] for e in episodes]), dtype=torch.long
        ),
        torch.tensor(
            np.concatenate([e["returns"] for e in episodes]), dtype=torch.float32
        ),
        torch.tensor(
            np.concatenate([np.arange(len(e["actions"])) for e in episodes]),
            dtype=torch.float32,
        ),
    )


######################################################################
## Code for students to be aware of
######################################################################

# TrainConfig holds defaults; the notebook supplies the assignment's budget.
# Actor maps (x, theta, xdot, thetadot) to two action logits. Value also receives
# the time index because expected capped return depends on time remaining.
# The network architectures and the NumPy inference helper are supplied.


@dataclass(frozen=True)
class TrainConfig:
    """Training and evaluation settings; the notebook overrides the defaults.

    Each update collects batch_episodes new episodes and takes ONE actor
    optimizer step. If baseline=True, it also takes value_epochs optimizer
    steps on the separate value network, using that same batch.
    """

    updates: int = 100
    batch_episodes: int = 16
    learning_rate: float = 0.01
    baseline: bool = False
    hidden: int = 16
    value_epochs: int = 30
    eval_every: int = 10
    eval_episodes: int = 20
    return_scale: float = (
        100.0  # Supplied numerical scaling for actor/baseline optimization.
    )


class Actor(nn.Module):
    """Two-action policy using the cart-pole's four physical state variables."""

    def __init__(self, hidden=16):
        """Build a policy with one hidden layer and initially uniform actions."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        # Start exactly at the uniform policy; initial features are random.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states):
        """Map state tensors of shape (..., 4) to action logits (..., 2)."""
        return self.net(states)

    def sampler(self):
        """Return a frozen function p(state) = probability of pushing right.

        Actions are 0 (left, -force) and 1 (right, +force). A state is a
        NumPy array (4,); p returns a scalar in [0, 1], not an action label.
        Batched states (..., 4) give probabilities (...,). The copy stays
        fixed while episodes are collected; forward is used for training.
        """
        w1, b1, w2, b2 = [p.detach().numpy().copy() for p in self.parameters()]

        def probability(states):
            """Return P(action=1 | state), the probability of a rightward force."""
            # einsum avoids spurious floating-point status warnings in some
            # macOS Accelerate float32 matrix-multiply implementations.
            hidden = np.tanh(np.einsum("...j,ij->...i", np.asarray(states), w1) + b1)
            logits = np.einsum("...j,ij->...i", hidden, w2) + b2
            difference = np.clip(logits[..., 0] - logits[..., 1], -80, 80)
            return 1.0 / (1.0 + np.exp(difference))

        return probability


class Value(nn.Module):
    """Action-independent baseline predicting return divided by return_scale."""

    def __init__(self, horizon):
        """Build a value network with physical-state and normalized-time inputs."""
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, states, times):
        """Map states (N, 4) and elapsed step indices (N,) to scaled values (N,)."""
        # Remaining time matters for the capped return, even at the same state.
        x = torch.cat([states, (times / self.horizon)[:, None]], dim=1)
        return self.net(x).squeeze(1)


######################################################################
## Code for students to study
######################################################################

# collect finishes each episode before calling your reward_to_go function.
# evaluate distinguishes sampling from greedy action selection.
# train calls your policy_loss, then fits the baseline for the NEXT batch.
# These supplied functions show how the mathematical steps fit together;
# no edits to the loop are required. The notebook calls train explicitly.


def collect(env, probability, rng, count):
    """Collect count complete episodes from the sampled policy.

    Inputs:
        env: CartPole instance, reset here at the start of every episode.
        probability: callable taking a float32 NumPy state array (4,) and
            returning a scalar p in [0, 1]. Choose action 1 (rightward force)
            with probability p and action 0 (leftward force) with 1 - p.
        rng: np.random.Generator controlling reset seeds and action sampling.
        count: int, number of episodes to collect.

    Returns:
        list[dict[str, np.ndarray]] of length count. For an episode of T
        transitions (T can differ between episodes), each dictionary contains:
          'states': float32 array (T, 4), states BEFORE the sampled actions;
          'actions': int64 array (T,), action labels 0 or 1;
          'returns': float64 array (T,), reward-to-go G_t from each action
              through this episode's end, computed by reward_to_go.
        Immediate rewards and the final post-action state are not stored.
        An episode ends at an angle/track limit or the configured time cap.
    """
    episodes = []
    for _ in range(count):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        states, actions, rewards = [], [], []
        while True:
            action = int(rng.random() < probability(obs))
            states.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        episodes.append(
            dict(
                states=np.asarray(states),
                actions=np.asarray(actions, dtype=np.int64),
                returns=reward_to_go(rewards),
            )
        )
    return episodes


def evaluate(actor, env_config, seeds, stochastic=False):
    """Evaluate a fixed actor; return a Python list[int] of episode lengths.

    Inputs:
        actor: trained or untrained Actor; its weights are not changed.
        env_config: cartpole.Config specifying dynamics, limits, and horizon.
        seeds: sequence of ints or 1-D NumPy integer array, one reset seed
            per episode. Reuse these seeds to compare the same initial states.
        stochastic: bool. True samples actions from the policy; False chooses
            its most probable action (ties choose action 0).

    Returns:
        A list of len(seeds) Python ints in the same order as seeds. Each is
        the number of env.step calls, including the final transition, between
        1 and env_config.horizon. It is also this episode's total reward
        because every transition earns +1. It is not a mean or success flag.
        Policy sampling uses a reproducible random generator for each seed.
    A fresh environment is created and closed here; no training takes place.
    """
    env = CartPole(env_config)
    probability = actor.sampler()
    scores = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed) + 800_000)
        for t in range(env_config.horizon):
            p = probability(obs)
            action = int(rng.random() < p) if stochastic else int(p > 0.5)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        scores.append(t + 1)
    env.close()
    return scores


def train(seed=0, config=None, env_config=None, output=None, *, eval_seeds=None):
    """Run REINFORCE and return (actor, history): (Actor, list[dict]).

    Inputs:
        seed: int controlling network initialization and training sampling.
        config: TrainConfig, or None for defaults (called cfg below).
        env_config: cartpole.Config, or None for defaults (ecfg below).
        output: str or pathlib.Path checkpoint filename, or None to skip saving.
        eval_seeds: sequence of ints or 1-D integer array, or None for the
            default starts 100_000 onward. Its length must be cfg.eval_episodes.

    One update collects cfg.batch_episodes complete episodes. The local
    episodes variable is collect's list of state/action/reward-to-go records;
    tensors concatenates them into a batch with N total transitions.
    There are cfg.updates actor optimizer steps, one per fresh batch. If the
    baseline is enabled, there are ALSO cfg.updates * cfg.value_epochs value-
    network optimizer steps. Update 0 only evaluates the initial policy.
    steps counts training env.step calls across all batches, not gradient
    steps; it excludes evaluation transitions. Episode lengths determine it.

    The loop below computes the current actor's log probability of each
    recorded action before policy_loss applies advantage weights and averages
    over episodes. Return and baseline inputs are fixed during that update.

    history contains one dictionary per monitoring point (including update 0):
        update, steps: int counters described above;
        seconds: float, elapsed wall time including monitoring;
        batch_return: float mean training episode length, or None at update 0;
        eval_mean, eval_success: float greedy mean length and fraction at cap;
        stochastic_mean: float sampled mean length;
        deterministic_scores, stochastic_scores: list[int] from evaluate.
    If output is provided, save the actor weights and configurations there.
    """
    cfg, ecfg = config or TrainConfig(), env_config or Config()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = Actor(cfg.hidden)
    value = Value(ecfg.horizon)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    voptimizer = torch.optim.Adam(value.parameters(), lr=0.003)
    env = CartPole(ecfg)
    rows, steps = [], 0  # Monitoring records; cumulative training transitions.
    start = time.perf_counter()
    if eval_seeds is None:
        eval_seeds = np.arange(100_000, 100_000 + cfg.eval_episodes)
    else:
        eval_seeds = np.asarray(eval_seeds, dtype=int)
        if len(eval_seeds) != cfg.eval_episodes:
            raise ValueError("eval_seeds must match config.eval_episodes")
    for update in range(cfg.updates + 1):
        # Update 0 is an initial evaluation; updates 1..cfg.updates train.
        if update:
            episodes = collect(env, actor.sampler(), rng, cfg.batch_episodes)
            states, actions, returns, times = tensors(episodes)
            steps += len(states)
            # This critic has only seen PREVIOUS batches: today's actor samples
            # cannot leak into their own baseline fit.
            # Supplied PyTorch bookkeeping: no_grad makes baseline predictions
            # fixed numbers for policy_loss, rather than part of the network
            # being optimized. Returns already come from fixed recorded data.
            # The student loss needs no .detach(); only actor predictions vary.
            with torch.no_grad():
                baseline = (
                    value(states, times) * cfg.return_scale
                    if cfg.baseline
                    else torch.zeros_like(returns)
                )
            # Supplied conversion: one current log probability per recorded
            # action. Keep its dependence on actor weights for backpropagation.
            log_probs = torch.log_softmax(actor(states), dim=-1)
            action_log_probs = log_probs[torch.arange(len(actions)), actions]
            loss = policy_loss(action_log_probs, returns, baseline, cfg.batch_episodes)
            # Numerical scaling for the existing optimizer settings, separate
            # from the student's mathematical objective. A positive constant
            # leaves the minimizer unchanged but rescales gradients.
            loss = loss / cfg.return_scale
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if cfg.baseline:
                for _ in range(cfg.value_epochs):
                    vloss = nn.functional.mse_loss(
                        value(states, times), returns / cfg.return_scale
                    )
                    voptimizer.zero_grad()
                    vloss.backward()
                    voptimizer.step()
            batch_return = float(np.mean([len(e["actions"]) for e in episodes]))
        else:
            batch_return = None
        if update % cfg.eval_every == 0 or update == cfg.updates:
            deterministic = evaluate(actor, ecfg, eval_seeds)
            stochastic = evaluate(actor, ecfg, eval_seeds, stochastic=True)
            row = dict(
                update=update,
                steps=steps,
                seconds=time.perf_counter() - start,
                batch_return=batch_return,
                eval_mean=float(np.mean(deterministic)),
                eval_success=float(np.mean(np.asarray(deterministic) == ecfg.horizon)),
                stochastic_mean=float(np.mean(stochastic)),
                stochastic_scores=stochastic,
                deterministic_scores=deterministic,
            )
            rows.append(row)
            print(
                f"seed={seed} baseline={cfg.baseline} "
                f"{ {k: v for k, v in row.items() if not k.endswith('_scores')} }",
                flush=True,
            )
    if output:
        torch.save(
            dict(
                actor=actor.state_dict(),
                config=asdict(cfg),
                env_config=asdict(ecfg),
                seed=seed,
            ),
            output,
        )
    env.close()
    return actor, rows


######################################################################
## Code for students to write
######################################################################

# Complete these two functions, then use the checks and runs in cartpole.ipynb.
# The pushing question imports reward_to_go from here rather than redoing it.
# reward_to_go uses ordinary NumPy. policy_loss receives PyTorch arrays so
# the training loop can compute how the loss depends on the actor's weights.
# They support NumPy-style arithmetic, .sum(), and .mean(). train (above)
# handles log probabilities and recorded-action indexing; no conversion to
# NumPy or special gradient bookkeeping is needed. Return the scalar array
# directly rather than calling float(), .item(), or np.asarray on the result.


def reward_to_go(rewards):
    """Compute the reward still to come after each action in one episode.

    Input:
        rewards: list/tuple of numbers or 1-D np.ndarray, length T.
            rewards[t] is earned by taking action t, including the final
            transition's reward. Supply ONE complete episode, not a batch.
    Output:
        np.ndarray, dtype np.float64, shape (T,). Entry t is the undiscounted
        sum G_t = rewards[t] + ... + rewards[T-1]. Empty input gives an empty
        array. The input is not modified.
    collect stores this array as 'returns'. train uses it both to weight each
    sampled action in policy_loss and as the value network's regression target
    (divided by return_scale there). PPO also reuses this sum on a single
    observed rollout prefix; its supplied caller adds a critic continuation
    estimate if that prefix ends at a computational cutoff.
    """
    raise NotImplementedError("your code here")


def policy_loss(action_log_probs, returns, baseline, episode_count):
    """Return a scalar loss whose minimization performs the REINFORCE update.

    Inputs (PyTorch arrays, described by their element types and shapes):
        action_log_probs: float32 array, shape (N,), with entry i equal to
            log pi(a_i | s_i) for the recorded action a_i at state s_i.
            train computes these using the CURRENT actor; they carry the
            connection to its weights.
        returns: float32 array, shape (N,), recorded reward-to-go G_t.
        baseline: float32 array, shape (N,), predicted returns at the
            same states/times, in reward units; all zeros if no baseline.
        episode_count: positive int B, number of complete episodes in this
            batch. The B episodes are concatenated, so N = T_1 + ... + T_B
            is their TOTAL number of transitions (episodes differ in length).
    Output:
        float32 scalar array, shape ()  ; do not return a Python float
    """
    raise NotImplementedError("your code here")
