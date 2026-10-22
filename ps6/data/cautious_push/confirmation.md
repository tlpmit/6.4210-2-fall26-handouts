# Homework 6: fixed ten-run confirmation

Use cautious BC followed by scalar residual PPO with GAE as the pushing exercise. The independent replication supports an improvement in expected final-policy reward.

Ten fresh PPO fits, seeds 250–259, each requested 8,192 pushes. All use one supplied BC fit and the same fixed 64-goal test benchmark. Earlier exploratory fits are excluded. The final greedy checkpoint is the prespecified primary endpoint; no seeds were added or dropped based on significance.

Mean reward gain **+0.479**, whole-training-run bootstrap 95% interval [0.457, 0.500]; t(9)=41.817, two-sided p=1.27529e-11. Training-run SD=0.0362; exact two-sided sign-test p=0.00195312.

These quantify PPO training randomness conditional on this BC and fixed goal benchmark. The t reference is exact for independent normal run differences and approximate otherwise. The sign test checks improvement direction. This study does not estimate variation across new BC fits and is not a controlled raw-PPO comparison.

| Policy | Success | Reward | Simulated seconds | Pushes | Goal error (cm) |
|---|---:|---:|---:|---:|---:|
| bc | 1.000 | 1.821 | 86.88 | 14.14 | 1.56 |
| bc_sampled | 1.000 | 1.836 | 86.08 | 14.36 | 1.57 |
| midpoint | 1.000 | 2.135 | 71.54 | 11.37 | 1.44 |
| final | 1.000 | 2.300 | 63.75 | 9.98 | 1.31 |
| sampled | 1.000 | 2.264 | 64.84 | 10.22 | 1.52 |
| selected | 1.000 | 2.300 | 63.72 | 9.97 | 1.32 |
| midpoint_sampled | 1.000 | 2.131 | 71.71 | 11.51 | 1.46 |

Final greedy time reduction versus BC: 26.62%. All episodes, including failures, contribute. Unfinished prefixes are not efficient completions.

| Seed | Actual pushes | Midpoint theta | Final theta | Final reward gain | Final success |
|---|---:|---:|---:|---:|---:|
| 250 | 8193 | 0.497 | 1.025 | +0.487 | 1.000 |
| 251 | 8196 | 0.481 | 1.037 | +0.491 | 1.000 |
| 252 | 8197 | 0.519 | 1.161 | +0.516 | 1.000 |
| 253 | 8199 | 0.643 | 1.209 | +0.524 | 1.000 |
| 254 | 8192 | 0.528 | 0.891 | +0.442 | 1.000 |
| 255 | 8199 | 0.658 | 0.933 | +0.446 | 1.000 |
| 256 | 8202 | 0.380 | 0.820 | +0.426 | 1.000 |
| 257 | 8193 | 0.507 | 1.113 | +0.506 | 1.000 |
| 258 | 8195 | 0.674 | 1.128 | +0.510 | 1.000 |
| 259 | 8196 | 0.502 | 0.893 | +0.442 | 1.000 |

[Full numerical analysis](confirmation_analysis.json) includes every seed's midpoint/final/selected greedy and matched sampled reward, success, time, push count and error, plus training-seed SDs kept separate from goal intervals.

Paired-goal intervals below condition on the fitted policies; they are not additional independent training-run evidence.

- final_minus_bc: reward +0.4791 [+0.4532, +0.5038]; success +0.0000 [+0.0000, +0.0000]; execution_seconds -23.1272 [-24.4680, -21.7462]; pushes -4.1594 [-4.3516, -3.9640]; distance -0.0025 [-0.0044, -0.0005]
- sampled_minus_bc: reward +0.4284 [+0.3912, +0.4657]; success +0.0000 [+0.0000, +0.0000]; execution_seconds -21.2325 [-23.1052, -19.4119]; pushes -4.1437 [-4.4406, -3.8594]; distance -0.0006 [-0.0020, +0.0009]
- final_minus_midpoint: reward +0.1644 [+0.1490, +0.1821]; success +0.0000 [+0.0000, +0.0000]; execution_seconds -7.7888 [-8.6806, -6.9570]; pushes -1.3922 [-1.5062, -1.2844]; distance -0.0013 [-0.0027, +0.0001]
- sampled_minus_midpoint: reward +0.1336 [+0.1101, +0.1552]; success +0.0000 [+0.0000, +0.0000]; execution_seconds -6.8665 [-7.9135, -5.7994]; pushes -1.2984 [-1.4422, -1.1484]; distance +0.0006 [-0.0009, +0.0020]

Costs and actual requested-command counts:

```json
{
  "actual_commands": {
    "250": 8193,
    "251": 8196,
    "252": 8197,
    "253": 8199,
    "254": 8192,
    "255": 8199,
    "256": 8202,
    "257": 8193,
    "258": 8195,
    "259": 8196
  },
  "elapsed_seconds": 26309.661336749792,
  "summed_worker_seconds": 134591.6992565822,
  "coordinator_command": [
    "/Users/lpk/Documents/lpkmac/git/fall26_64210/.venv/bin/python",
    "-u",
    "/Users/lpk/Documents/lpkmac/git/fall26_64210/misc/ps6_rl/results/push_cautious_confirm10/runner.py"
  ],
  "calendar_seconds": 35919.008591,
  "original_demonstrations_and_bc_seconds": {
    "training_data": 748.7774532088079,
    "validation_data": 239.18489591637626,
    "bc_fit": 0.615174874663353
  },
  "recorded_batch_collection_seconds": 172.60367929097265
}
```

The coordinator command is recorded in background_job.json. Summed worker time includes training and evaluation and is not elapsed time. Measured coordinator time and calendar time are reported separately; sleep and concurrency affect their relationship.

The student path fits BC in seconds and performs one PPO iteration on the supplied genuine on-policy batch. The full curves use fresh interaction at each iteration. A fixed instructor batch cannot replace fresh interaction after arbitrary policy changes. The current exercise uses the explicit 32-command computational cutoff rather than the old 16-command cutoff. The short-path integration and isolated student build must pass before release; standard laptop/Colab timing and dates remain instructor release checks.
