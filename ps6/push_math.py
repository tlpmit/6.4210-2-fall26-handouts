"""Mathematical steps for the PPO pushing notebook.

The instructor checkout contains solutions between markers. The handout build
replaces those regions with NotImplementedError for students to complete.
"""

######################################################################
## Code for nobody but Python to read
######################################################################

import numpy as np
import torch
from torch import nn


######################################################################
## Code for students to be aware of
######################################################################

# ppo_push.ipynb runs cloning, one recorded update and reference analysis.
# The simulator, free-space motion planner, networks, and optimizers are
# supplied. The five mathematical functions below need implementation.
# For a batch of N transitions, face logits have shape (N, 4); selected
# faces, latent distances, log probabilities, returns, and values have (N,).
# Distances passed to bc_loss are in meters, not Gaussian latent coordinates.


######################################################################
## Code for students to study
######################################################################

# The supplied callers live in other files so this module stays small.
# Read these pieces alongside the function contracts below:
#
# residual_push.py and push_policy.py:
#   HybridActorCritic.distributions selects the Gaussian for the chosen face.
#   HybridActorCritic.evaluate calls joint_log_prob on that face and latent.
#   distance_from_raw maps the latent to a bounded physical push distance.
#   update_from_batch combines ppo_actor_loss and critic_loss; only theta
#   and a fresh critic change. BC faces and exploration variance are frozen.
# residual_push.py:
#   _imitation selects the DEMONSTRATED face's distance before bc_loss.
#   batch_targets calls gae_advantages separately for each complete episode.
#
# Stored old log probabilities, returns, and advantages are fixed targets
# during the PPO update. GAE receives ordinary NumPy arrays for ONE episode.


######################################################################
## Code for students to write
######################################################################

# Complete these functions in handout order; ppo_push.ipynb provides checks.
#
# Array guide: inputs use PyTorch's array type, with NumPy-style shapes and
# indexing. Ordinary +, -, *, /, **2, x.sum(), and x.mean() work as usual.
# Use torch.exp(x), torch.minimum(x, y), and torch.clamp(x, low, high) where
# NumPy would use np.exp, np.minimum, and np.clip. All are elementwise.
# nn.functional.cross_entropy(logits, labels) returns the mean negative log
# probability of the indicated labels, converting scores to probabilities.
# Return the resulting scalar array directly. Do not convert it with float(),
# .item(), or np.asarray: the training loop needs its dependence on the model.
# The supplied callers handle fixed targets; no .detach() is needed here.


def joint_log_prob(categorical, normal, faces, raw):
    """Return log p(face, latent distance | state), one value per observation.

    Inputs:
        categorical: supplied torch.distributions.Categorical object for N
            observations. categorical.log_prob(faces) returns a float32 array
            (N,) of log probabilities for the indicated face labels.
        normal: supplied torch.distributions.Normal object, already using the
            Gaussian for each chosen face. normal.log_prob(raw) returns a
            float32 array (N,) of log densities at the supplied latent values.
        faces: int64 array (N,), face labels 0, 1, 2, or 3.
        raw: float32 array (N,), recorded Gaussian samples BEFORE the sigmoid
            distance transform; these are not distances in meters.
    Output:
        float32 array (N,), joint log probability/density for each stored
        (face, raw) pair. Do not average the batch here. The distribution
        objects provide the log_prob operations; no formula for a Gaussian
        density needs to be implemented.
    """
    raise NotImplementedError("your code here")


def bc_loss(logits, predicted_distances, faces, distances):
    """Return scalar cross entropy plus mean((distance error / 0.1 m)^2).

    Inputs:
        logits: float32 array (N, 4), face scores BEFORE softmax.
        predicted_distances: float32 array (N,), model predictions in meters
            for the DEMONSTRATED faces, already selected by the supplied code.
        faces: int64 array (N,), demonstrated labels 0, 1, 2, or 3.
        distances: float32 array (N,), fixed demonstrated distances in meters.
    Output:
        float32 scalar array, shape (), the mean imitation loss over N
        demonstrations. Use nn.functional.cross_entropy(logits, faces) for
        the face term; ordinary arithmetic and .mean() suffice for distances.

    The 0.1 m reference length (the original action-range width) makes the
    distance error dimensionless. Add the two mean losses with no further
    weighting. A 1 cm error contributes 0.01 to the loss. Assess face accuracy
    and physical distance error separately on held-out demonstration episodes.
    """
    raise NotImplementedError("your code here")


def ppo_actor_loss(log_probs, old_log_probs, advantages, clip=.1):
    """Return the scalar negative mean of the clipped PPO surrogate.

    Inputs:
        log_probs: float32 array (N,), current joint log probabilities from
            joint_log_prob, one for each recorded action.
        old_log_probs: float32 array (N,), log probabilities of those SAME
            actions under the policy that collected them; fixed by the caller.
        advantages: float32 array (N,), fixed GAE advantage estimates,
            already standardized by batch_targets in residual_push.py; may be
            positive or negative.
            For raw estimates a, the caller supplies
            (a - a.mean()) / (a.std(unbiased=False) + 1e-8), computed once per
            collected batch. Use these weights as given; do not standardize again.
        clip: positive Python float, the tolerance in [1 - clip, 1 + clip].
    Output:
        float32 scalar array, shape (), the negative clipped surrogate averaged
        over N transitions. Return this array directly, without float() or
        .item(). The supplied code handles all fixed-target bookkeeping.

    Standardization is optional in PPO: dividing by the batch standard
    deviation controls the actor-loss scale; centering removes a common
    offset. Centering can change signs and hence clipping behavior, so this
    is an optimization convention, not an algebraically equivalent rewrite.
    Rewards and critic targets are not standardized.

    PyTorch operations (torch is already imported):
        torch.exp(x): exponentiate each entry, like np.exp(x).
        torch.clamp(x, min=low, max=high): replace entries below low by low
            and entries above high by high, like np.clip(x, low, high).
        torch.minimum(x, y): take the smaller of each corresponding pair of
            entries, like np.minimum(x, y); not the minimum of the whole batch.
        x.mean(): average all entries, returning a scalar array.
        The first three operations preserve the (N,) shape. Ordinary array
        subtraction, multiplication, and negation work as in NumPy. Use the
        PyTorch operations rather than converting these arrays to NumPy so
        the training loop can compute derivatives of your loss.
    """
    raise NotImplementedError("your code here")


def critic_loss(values, returns):
    """Return the mean squared error of the critic's return predictions.

    Inputs:
        values: float32 array (N,), current critic predictions in reward units.
        returns: float32 array (N,), fixed return targets in the same units.
            The caller sums observed rewards and, at a computational cutoff,
            adds the old critic's estimate of later reward. Success and
            physical failure have zero continuation. No gradient bookkeeping
            is needed in this function.
    Output:
        float32 scalar array, shape (), mean squared prediction error over N
        transitions. The caller applies the critic coefficient; do not include
        it here. Ordinary subtraction, **2, and .mean() are sufficient.
    """
    raise NotImplementedError("your code here")


def gae_advantages(rewards, values, next_values, lam=.95):
    """Return unstandardized GAE advantages for ONE episode, gamma=1.

    Inputs (ordinary NumPy float arrays of shape (T,), or lists of T numbers):
        rewards: rewards[t] is the reward after action t.
        values: values[t] is the OLD critic's prediction at the observation
            before action t.
        next_values: next_values[t] is the OLD critic's prediction at the
            observation after action t. The caller sets the last entry to
            zero at a true terminal and to V_old(final observation) at a
            computational cutoff.
        lam: float in [0, 1], the GAE lambda.
    Output:
        NumPy float array (T,), A, with
            delta[t] = rewards[t] + next_values[t] - values[t],
            A[t] = delta[t] + lam * A[t+1],  using A[T] = 0.

    This function handles one episode and returns raw advantages;
    batch_targets in residual_push.py calls it once per episode, so nothing
    carries into another episode, and standardizes the whole batch after.
    Use np.asarray(..., dtype=float), np.empty(T), and a backward loop
    over range(T-1, -1, -1). No PyTorch or gradient bookkeeping is needed.
    This is an advantage estimate, not the Monte Carlo critic target.
    """
    raise NotImplementedError("your code here")
