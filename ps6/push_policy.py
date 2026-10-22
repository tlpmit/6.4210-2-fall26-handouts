"""Pure PyTorch push policies; importing this module does not load Drake."""

from copy import deepcopy
import numpy as np
import torch
from torch import nn
from push_math import joint_log_prob

OBSERVATION_DIM = 13

DISTANCE_MIN = 0.0
DISTANCE_MAX = 0.125


def distance_from_raw(raw):
    """Map latent arrays to meters: no positive minimum, maximum 12.5 cm.

    Finite sigmoid inputs approach zero continuously; the executor also
    accepts an exact zero command from a heuristic or a direct caller.
    """
    return DISTANCE_MAX * torch.sigmoid(raw)


class HybridActorCritic(nn.Module):
    def __init__(self, obs_dim=OBSERVATION_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()
        )
        self.face = nn.Linear(32, 4)
        self.mean = nn.Linear(32, 4)
        self.log_std = nn.Parameter(torch.full((4,), np.log(0.6)))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        for layer in [self.face, self.mean]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def policy_parameters(self, obs):
        hidden = self.actor(obs)
        return self.face(hidden), self.mean(hidden)

    def distributions(self, obs, faces=None):
        logits, raw_means = self.policy_parameters(obs)
        categorical = torch.distributions.Categorical(logits=logits)
        if faces is None:
            faces = categorical.sample()
        means = raw_means.gather(-1, faces[..., None]).squeeze(-1)
        std = self.log_std.clamp(-3.0, 0.5).exp()[faces]
        return categorical, torch.distributions.Normal(means, std)

    def evaluate(self, obs, faces, raw):
        categorical, normal = self.distributions(obs, faces)
        return (
            joint_log_prob(categorical, normal, faces, raw),
            self.critic(obs).squeeze(-1),
            categorical.entropy(),
        )

    @torch.no_grad()
    def act(self, obs, *, greedy=False, rng=None):
        tensor = torch.as_tensor(obs, dtype=torch.float32)[None]
        logits, means = self.policy_parameters(tensor)
        logits = logits[0]
        if greedy:
            face = int(logits.argmax())
            raw = means[0, face]
        elif rng is not None:
            probabilities = logits.softmax(-1).numpy().astype(float)
            face = int(rng.choice(4, p=probabilities / probabilities.sum()))
            raw = means[0, face] + self.log_std.clamp(-3.0, 0.5).exp()[face] * float(
                rng.normal()
            )
        else:
            face = int(torch.distributions.Categorical(logits=logits).sample())
            raw = torch.distributions.Normal(
                means[0, face], self.log_std.clamp(-3.0, 0.5).exp()[face]
            ).sample()
        log_prob, value, _ = self.evaluate(tensor, torch.tensor([face]), raw[None])
        return face, float(raw), float(log_prob), float(value)


class ResidualHybridActorCritic(HybridActorCritic):
    """Frozen BC plus unconstrained corrections to face logits and raw distance.

    Zero output heads preserve the complete initial joint action distribution.
    The sigmoid still permits every legal distance, and every face can change.
    """

    def __init__(self, base):
        super().__init__(base.actor[0].in_features)
        self.base = deepcopy(base)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.log_std = nn.Parameter(base.log_std.detach().clone())
        self.critic = deepcopy(base.critic)
        for key in [
            "observation_version",
            "training_horizon",
            "free_time_cost",
            "total_time_cost",
            "push_penalty",
            "box_com_x",
            "orientation_features",
            "mass_model",
            "timestep",
            "physics",
        ]:
            if hasattr(base, key):
                setattr(self, key, getattr(base, key))

    def policy_parameters(self, obs):
        with torch.no_grad():
            logits, means = self.base.policy_parameters(obs)
        delta_logits, delta_means = super().policy_parameters(obs)
        return logits + delta_logits, means + delta_means
