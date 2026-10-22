# Cautious pushing: supplied data and reference runs

Use `ps6/ppo_push.ipynb` and the mathematical functions in `push_math.py`.
The required numerical exercise uses NumPy, PyTorch, SciPy and Matplotlib;
it does not require Drake or a long simulator run.

The handwritten teacher selects a body-fixed face and caps intended block
displacement at 4 cm. An affine calibration converts that displacement to
stick travel (about 8.35 cm at the cap). Behavior cloning imitates these
commands. PPO then freezes the BC weights and face probabilities and learns
one signed scalar correction to the latent push distance, tapered to zero
near the goal. The supplied full runs use GAE with lambda 0.95. See the
notebook and `plan.json` for the reward, taper, exploration and PPO settings.

| Asset | Contents |
|---|---|
| `train.npz`, `validation.npz` | 64/16 successful teacher episodes, 886/221 transitions; split by whole episode |
| `bc.pt` | Supplied frozen BC actor; all PPO reference runs share this one fit |
| `calibration.json` | Affine displacement calibration used by the teacher |
| `update_start.pt` | The exact policy that sampled the recorded update batch |
| `update_batch.npz`, `update_batch.json` | One genuine on-policy batch: 16 predetermined goals, 185 transitions, no filtering by outcome |
| `checkpoints/` | Initial, midpoint, final and monitoring-selected actors, named by training seed |
| `exploratory/` | Completed seeds 150–152: full monitoring histories and per-goal midpoint/final/selected greedy and sampled evaluations |
| `confirmatory/` | A separately planned fixed cohort of ten fresh training seeds 250–259; its status must be complete before analysis |
| `provenance.json`, `checksums.json` | Original source references and SHA-256 integrity checks |
| `replay_policy.json` | Best monitoring-selected checkpoint for the optional paired Drake demonstration; distinct from the fixed-final primary endpoint |

Each NPZ demonstration stores observations, face labels, commanded distances
and episode identifiers. The update batch additionally stores latent actions,
old joint log probabilities, old values, rewards, next-state values and
terminal/cutoff flags. At a physical terminal, the next value is zero; at
the 32-command computational cutoff it is the old critic prediction. GAE
must not cross episode boundaries. This cutoff differs from the historical
16-command pushing exercise.

The short update always starts from `update_start.pt`, even if you have just
fit another BC actor. Applying several optimization epochs to this batch is
one PPO iteration. Repeating the call does not provide fresh interaction or
another valid online training iteration. The full reference curves were
produced using fresh on-policy simulator interaction at each iteration.

The notebook can also run a fresh initial BC rollout before the coding exercise
and display it next to a learned policy at the end. Both use the same fixed
illustrative goal and exact initial state. Their score table reports success,
simulated seconds, push count and goal error; use equal playback speeds. The
learned checkpoint maximizes monitoring success then reward (earlier training
step and lower seed break ties), never a test-goal or replay outcome. This
illustration does not substitute for the all-run fixed-final comparison.

For the confirmatory comparison, the independent sampling unit is one PPO
training run's mean reward difference from BC on the shared 64-goal
benchmark. Use the fixed final checkpoint, include all ten runs, and do not
pool in exploratory seeds. The predeclared primary test is a two-sided
one-sample t-test of these ten differences; the whole-run bootstrap interval
and exact sign test are also supplied. These results describe PPO training
randomness conditional on this supplied BC and fixed goal benchmark. They
do not establish variation across new BC fits or arbitrary environments.
Goal bootstrap intervals instead condition on the fitted policies.

All reference variants include failed episodes as well as successes. A
failed prefix with fewer pushes is not a faster completion. Compare success
and reward before interpreting time or push counts. Historical raw/full
policy PPO data elsewhere in `ps6/data` use different setups; they are not
a controlled comparison with this restricted scalar actor and its learning
rate.

Checkpoints use PyTorch's `weights_only=True` load path. Residual optimizer
and RNG states remain in the instructor study; student checkpoints preserve
the model tensors and configuration. `verify_data()` checks the packaged
files. A running confirmatory status is a packaging/release gate, not a
license to analyze an incomplete favorable subset.
