# SuperMimic

[SuperMimic](https://supermimic.github.io) trains one Unitree G1 whole-body motion tracker through a learned,
differentiable world model instead of model-free RL. This is the Eden port of the official implementation
([hybridrobotics/supermimic](https://github.com/hybridrobotics/supermimic)):

> Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath.
> SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model. CoRL 2026.
> [arXiv 2610.28480](https://arxiv.org/abs/2610.28480)

The task is `supermimic` (`eden/tasks/supermimic`) and the learner is `SuperMimicRunnerOptions`
(`eden/learning/wrappers/rsl_rl/supermimic`).

## Method

Each runner iteration steps 4096 simulated envs 10 times with the deterministic policy and stores the transitions
in a replay buffer of 1000 steps per env. On average 0.4 times per env step:

1. The world model fits 10 batches of 2048 replay windows of 8 steps. It predicts the change of the root twist and
   the joint velocities from the state, the commanded joint targets and the env's randomized parameters (foot
   friction, torso centre of mass, encoder bias). Poses integrate analytically.
2. The policy improves on 4096 rollouts of 32 steps that the frozen world model imagines from replay states. The
   loss is the negative of the task's tracking reward, computed analytically on the predicted state, discounted
   and weighted by the probability of not having fallen. A critic bootstraps the return beyond the horizon.

Resets concentrate on the 0.2 s bins of the motion library where the policy earns a low return.

## Train

LAFAN1 (40 clips, 2.45 h) downloads on first use:

```bash
eden train --task supermimic --processed_data_path /data/lafan1_g1.pt
```

BONES-SEED needs the G1 split of [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) (gated).
Write the clip list of the paper's filter once (129,514 clips, 262.3 h), then train on it:

```bash
python examples/projects/mimic/supermimic/bones_seed_clip_list.py --raw-data-path /data/bones_seed/g1/csv \
    -o bones_seed_clips.json
eden train --task supermimic --dataset bones_seed --raw_data_path /data/bones_seed/g1/csv \
    --processed_data_path /data/bones_seed_g1.pt --clip_list bones_seed_clips.json
```

The first run prepares the library into `--processed_data_path` (13.5 GB for BONES-SEED, root and joints only;
link states are computed by forward kinematics when an env reads them); later runs load it. Training takes about
13 GB of GPU memory on top of the library. One iteration takes about 2 s on an A100 and about 1.5 s on an L40S
(28,000 env steps per second, of which 0.4 s is collection and 1.0 s is learning).

Measured on LAFAN1 with the default options, one L40S and one seed:

| Iterations | Training time | Mean episode reward | Mean episode length (of 500 steps) |
|---|---|---|---|
| 100 | 2 min | 12.0 | 325 |
| 500 | 12 min | 37.6 | 480 |
| 1000 | 24 min | 39.6 | 472 |
| 2000 | 49 min | 42.1 | 482 |

The run took 53 minutes with startup. The default 20,000 iterations take about 8 hours at that rate.

## BlinkMimic recipe

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (F. Chen, 2026) trains the same tracker through the same
kind of world model, and its README reports SuperMimic's tracking quality after fewer updates. `--recipe blinkmimic`
selects the learner-side part of its released recipe (`PRESET[main]` in its `scripts/presets.sh`):

```bash
eden train --task supermimic --recipe blinkmimic --processed_data_path /data/lafan1_g1.pt
```

The recipe sets the options below, and `--override` sets any of them on its own.

| Option (under `runner_options`) | Default | Recipe | What it does |
|---|---|---|---|
| `policy.ema_decay` | 0 | 0.995 | Replay and evaluation run a Polyak average of the actor; collection keeps the trained actor |
| `policy.reference_action` | `false` | `true` | The actor's output is added to the action that commands the next reference pose, so the network learns a correction to it |
| `policy.lr_multiplier` | 1 | 4 | The actor's learning rate is this multiple of `policy.lr` |
| `policy.lr_decay_return`, `lr_decay_deadline`, `lr_decay_iterations` | none, none, 75 | 0.75, 350, 75 | Once `Train/mean_normalized_return` reaches 0.75, or at iteration 350, the multiplier ramps to 1 over 75 iterations |
| `policy.batch_size`, `policy.lr` | 4096, 2e-4 | 2048, 4e-4 | Half the imagined rollouts per actor step |
| `world_model.hidden_dims`, `num_batches`, `lr` | 3 × 2048, 10, 2e-4 | 3 × 1024, 5, 4e-4 | A smaller world model fit on half the batches |
| `world_model.mirror_share` | 0 | 0.5 | Half of every world-model batch is replaced by its left-right mirror image |
| `return_sampler.failure` | 0 | 0.3 | 30% of the resets start up to 1 s (`failure_lead_s`) before the frames of recent falls |
| `return_sampler.explore`, `keep` | 0.5, 0.1 | 0.35, 0.07 | The other 70% of the resets keep their split |
| `replay_ratio` | 16 | 8 | With 5 world-model batches, still 0.4 update rounds per env step |

`Train/mean_normalized_return` is the mean, over the episodes that ended in an iteration, of the episode's reward
per step over the largest reward of one step. BlinkMimic logs the same quantity as `ep_ret_norm`.

BlinkMimic counts iterations in update rounds. This task runs 4 rounds per iteration at 4096 envs, so BlinkMimic's
3500 iterations are 875 here, and its learning-rate deadline of 1400 and ramp of 300 are the 350 and 75 above. The
recipe leaves `max_iterations` alone; pass `--override "runner_options: {max_iterations: 875}"` for a run of
BlinkMimic's length.

The default recipe already optimizes with Muon, replays its updates as CUDA graphs, integrates poses from predicted
velocity changes and starts policy updates with the first world-model update, as BlinkMimic's recipe does.
BlinkMimic's world model also predicts a position correction on top of the integrated pose, which this one does
not. BlinkMimic's contact reward, its evaluation-guided resets and its capture of the simulator step are not ported.

> **TODO (timing):** training time and return of `--recipe blinkmimic` against the default recipe, on one GPU and
> one seed. Not measured yet.

## Evaluate

Play every clip once from its first frame, without reset noise, and count the falls:

```bash
python examples/projects/mimic/supermimic/evaluate.py --checkpoint logs/supermimic/<run>/model_20000.pt \
    --processed-data-path /data/lafan1_g1.pt
```

Replay (`eden inference` and this script) drives every environment through the motion encoder, whatever `slot_mix`
the checkpoint trained on, and reads the proprioception without the training sensor noise. `eden inference` rebuilds
the learner from the run's `config.json`. This script builds the task anew, so a checkpoint trained with
`--recipe blinkmimic` needs `--recipe blinkmimic` here too; without it the load stops and names the options that
differ.

## Options

`--override` reaches every learner option (`eden/learning/wrappers/rsl_rl/supermimic/options.py`). The ones the
official implementation exposes as variants:

| Option | Default | Variant |
|---|---|---|
| `runner_options.policy.algorithm` | `actor_critic` | `bptt` (no critic) |
| `runner_options.policy.observation` | `state` (the model state) | `proprio` (deployable proprioception with sensor noise) |
| `runner_options.policy.history_length` | 1 | more frames of proprioception |
| `runner_options.policy.slot_mix` | `{motion: 1.0, velocity: 0.0}` | a share for the velocity-command encoder |
| `runner_options.return_sampler.enabled` | `true` | `false`: uniform resets |

The official maximal-coordinate world-model state and the ONNX export are not ported: `eden export` and `eden deploy`
exit for this task before building anything. Replay a checkpoint with `eden inference --task supermimic`.
